#!/usr/bin/env python3
"""Channelizer front-end decimator + CIC droop-compensation FIR (§4.2 / §7 step 7).

Extends the time-multiplexed channelizer (``channelizer_lane.py``) with the two
filtering stages the lane prototype deferred:

  1. ``FrontEndDecimator`` -- a **shared** complex FIR low-pass decimator. The
     AD936x is run oversampled (e.g. ~61.44 MHz) and this one block decimates the
     whole capture down to the working rate (~the §8.2 window, ~14 MHz) with a
     *flat* passband across the window (a CIC here would droop the band edges,
     hurting the outer channels). It is amortized over all channels (one instance),
     so its cost is paid once -- not per channel / per lane.

  2. ``CompensationFIR`` -- a short per-channel FIR that **flattens the CIC droop**
     left by the per-channel decimation (``am_audio.CICDecimator``) and sharpens
     the channel selectivity. Designed as a least-squares inverse of the CIC
     passband. Runs at the (low) channel output rate, so in the full design it is
     time-multiplexed across channels like the CIC (one MAC engine).

Both are built on one generic integer building block, ``FIRStage`` (a direct-form
decimating FIR), which is verified bit-exact against a Python model. The composite
per-channel response (front-end -> NCO mix -> CIC -> compensation FIR) is then
swept to show a flat channel passband and strong adjacent-channel rejection.

Run:  python hdl/channelizer_chain.py
"""
from __future__ import annotations

import math
import pathlib

import numpy as np
import scipy.signal
from amaranth import Array, Cat, Const, Elaboratable, Module, Mux, Signal, signed
from amaranth.lib.memory import Memory
from amaranth.sim import Simulator

from am_audio import CICDecimator
from ddc_tune_decimate import SYNC_PERIOD

OUT_DIR = pathlib.Path(__file__).parent / "out"


# ---------------------------------------------------------------------------
# Generic integer decimating FIR
# ---------------------------------------------------------------------------

class FIRStage(Elaboratable):
    """Direct-form integer FIR with integer decimation.

    ``y[m] = ( sum_k coeffs[k] * x[n-k] ) >> out_shift`` produced once every
    ``decimation`` input samples (at input index n = decimation-1, 2*decimation-1,
    ...). The tap delay line advances one sample per ``clken``; the MAC is taken
    over [x_in, tap0, tap1, ...] so the just-arrived sample is included.

    Parameters
    ----------
    coeffs : sequence[int]   -- fixed-point taps.
    decimation : int         -- output every D-th input (1 = no decimation).
    in_width : int           -- signed input width.
    out_shift : int          -- arithmetic right shift after the MAC (renormalize).
    """

    def __init__(self, coeffs, decimation: int, in_width: int, out_shift: int):
        assert decimation >= 1
        self.coeffs = [int(c) for c in coeffs]
        self.ntaps = len(self.coeffs)
        self.decimation = decimation
        self.iw = in_width
        self.out_shift = out_shift
        cw = max((abs(c).bit_length() for c in self.coeffs), default=1) + 1
        self.coeff_width = cw
        self.acc_w = in_width + cw + math.ceil(math.log2(self.ntaps)) + 1
        self.ow = self.acc_w - out_shift + 1

        self.clken = Signal()
        self.x_in = Signal(signed(in_width))
        self.strobe_out = Signal()
        self.y_out = Signal(signed(self.ow), reset_less=True)

    @staticmethod
    def model(x, coeffs, decimation, out_shift):
        x = np.asarray(x, dtype=np.int64)
        h = [int(c) for c in coeffs]
        n_t = len(h)
        out = []
        for n in range(len(x)):
            if n % decimation == decimation - 1:
                acc = 0
                for k in range(n_t):
                    if n - k >= 0:
                        acc += h[k] * int(x[n - k])
                out.append(acc >> out_shift)         # arithmetic (floor) shift
        return np.array(out, dtype=np.int64)

    def elaborate(self, platform):
        m = Module()
        taps = [Signal(signed(self.iw), name=f"tap{k}")
                for k in range(self.ntaps - 1)]
        with m.If(self.clken):
            if taps:
                m.d.sync += taps[0].eq(self.x_in)
                for k in range(1, len(taps)):
                    m.d.sync += taps[k].eq(taps[k - 1])

        window = [self.x_in] + taps                  # x[n], x[n-1], ...
        acc = Signal(signed(self.acc_w))
        m.d.comb += acc.eq(sum(c * w for c, w in zip(self.coeffs, window)))

        cnt = Signal(range(self.decimation))
        dec_stb = Signal()
        m.d.comb += dec_stb.eq(self.clken & (cnt == self.decimation - 1))
        with m.If(self.clken):
            with m.If(cnt == self.decimation - 1):
                m.d.sync += cnt.eq(0)
            with m.Else():
                m.d.sync += cnt.eq(cnt + 1)

        with m.If(dec_stb):
            m.d.sync += self.y_out.eq(acc >> self.out_shift)
        m.d.sync += self.strobe_out.eq(dec_stb)
        return m


class FrontEndDecimator(Elaboratable):
    """Shared complex FIR low-pass decimator (one per receiver, not per channel)."""

    def __init__(self, coeffs, decimation: int, in_width: int, out_shift: int):
        self.i = FIRStage(coeffs, decimation, in_width, out_shift)
        self.q = FIRStage(coeffs, decimation, in_width, out_shift)
        self.clken = Signal()
        self.re_in = Signal(signed(in_width))
        self.im_in = Signal(signed(in_width))
        self.strobe_out = Signal()
        self.re_out = self.i.y_out
        self.im_out = self.q.y_out

    def elaborate(self, platform):
        m = Module()
        m.submodules.i = self.i
        m.submodules.q = self.q
        m.d.comb += [
            self.i.clken.eq(self.clken), self.i.x_in.eq(self.re_in),
            self.q.clken.eq(self.clken), self.q.x_in.eq(self.im_in),
            self.strobe_out.eq(self.i.strobe_out),
        ]
        return m


class MultiStageDecimator(Elaboratable):
    """Cascade of complex FIRStage decimators (the realistic shared front-end).

    A single long FIR at the working rate is ~43 DSP; a cascade of halfband
    decimate-by-2 stages does the same job far cheaper because each later stage
    runs at a lower rate and halfband filters null ~half their taps. ``stages`` is
    a list of dicts {coeffs, decim, out_shift}; each is built for I and Q.
    """

    def __init__(self, stages, in_width: int = 12):
        self.stages_spec = stages
        self.in_width = in_width
        self.i_chain, self.q_chain = [], []
        w = in_width
        for st in stages:
            fi = FIRStage(st["coeffs"], st["decim"], w, st["out_shift"])
            fq = FIRStage(st["coeffs"], st["decim"], w, st["out_shift"])
            self.i_chain.append(fi)
            self.q_chain.append(fq)
            w = fi.ow
        self.clken = Signal()
        self.re_in = Signal(signed(in_width))
        self.im_in = Signal(signed(in_width))
        self.strobe_out = Signal()
        self.re_out = self.i_chain[-1].y_out
        self.im_out = self.q_chain[-1].y_out

    @staticmethod
    def model(re, im, stages):
        for st in stages:
            re = FIRStage.model(re, st["coeffs"], st["decim"], st["out_shift"])
            im = FIRStage.model(im, st["coeffs"], st["decim"], st["out_shift"])
        return re, im

    def elaborate(self, platform):
        m = Module()
        for idx, (fi, fq) in enumerate(zip(self.i_chain, self.q_chain)):
            m.submodules[f"i{idx}"] = fi
            m.submodules[f"q{idx}"] = fq
        m.d.comb += [
            self.i_chain[0].clken.eq(self.clken), self.i_chain[0].x_in.eq(self.re_in),
            self.q_chain[0].clken.eq(self.clken), self.q_chain[0].x_in.eq(self.im_in),
        ]
        for idx in range(1, len(self.i_chain)):
            pi, ci = self.i_chain[idx - 1], self.i_chain[idx]
            pq, cq = self.q_chain[idx - 1], self.q_chain[idx]
            m.d.comb += [
                ci.clken.eq(pi.strobe_out), ci.x_in.eq(pi.y_out),
                cq.clken.eq(pq.strobe_out), cq.x_in.eq(pq.y_out),
            ]
        m.d.comb += self.strobe_out.eq(self.i_chain[-1].strobe_out)
        return m


class TdmFirEngine(Elaboratable):
    """Folded (time-multiplexed) FIR: one MAC serves N channels' cleanup filters.

    The per-channel selectivity/droop-comp FIR runs at the low channel rate, so a
    single multiply-accumulate iterated over the taps covers many channels. Each
    channel keeps its own ``ntaps`` delay line; presenting (in_chan, x_in) with
    ``in_valid`` runs an ``ntaps``-cycle pass and emits ``out`` for that channel.
    """

    def __init__(self, coeffs, n_channels: int, in_width: int, out_shift: int):
        self.coeffs = [int(c) for c in coeffs]
        self.ntaps = len(self.coeffs)
        self.n = n_channels
        self.iw = in_width
        self.out_shift = out_shift
        self.cw = max((abs(c).bit_length() for c in self.coeffs), default=1) + 1
        self.acc_w = in_width + self.cw + math.ceil(math.log2(self.ntaps)) + 2
        self.ow = self.acc_w - out_shift + 1

        self.in_valid = Signal()
        self.in_chan = Signal(range(max(2, n_channels)))
        self.x_in = Signal(signed(in_width))
        self.busy = Signal()
        self.out_valid = Signal()
        self.out_chan = Signal(range(max(2, n_channels)))
        self.out = Signal(signed(self.ow), reset_less=True)

    @staticmethod
    def model(samples, coeffs, n_channels, out_shift):
        """samples: list of (chan, x). Returns list of (chan, y), one per input."""
        h = [int(c) for c in coeffs]
        nt = len(h)
        hist = [[0] * nt for _ in range(n_channels)]
        out = []
        for ch, x in samples:
            line = hist[ch]
            line.insert(0, int(x))
            del line[-1]
            acc = sum(h[k] * line[k] for k in range(nt))
            out.append((ch, acc >> out_shift))
        return out

    def elaborate(self, platform):
        m = Module()
        nt = self.ntaps
        mem = Array(Signal(signed(self.iw), name=f"d{i}")
                    for i in range(self.n * nt))
        crom = Array(Const(c, signed(self.cw)) for c in self.coeffs)

        cnt = Signal(range(nt))
        chan = Signal.like(self.in_chan)
        xreg = Signal(signed(self.iw))
        acc = Signal(signed(self.acc_w))
        base = Signal(range(max(2, self.n * nt)))

        m.d.comb += base.eq(chan * nt)
        with m.FSM():
            with m.State("IDLE"):
                m.d.sync += self.out_valid.eq(0)
                with m.If(self.in_valid & ~self.busy):
                    m.d.sync += [chan.eq(self.in_chan), xreg.eq(self.x_in),
                                 cnt.eq(nt - 1), acc.eq(0), self.busy.eq(1)]
                    m.next = "RUN"
            with m.State("RUN"):
                with m.If(cnt != 0):
                    val = mem[base + cnt - 1]
                    m.d.sync += mem[base + cnt].eq(val)
                    m.d.sync += acc.eq(acc + crom[cnt] * val)
                    m.d.sync += cnt.eq(cnt - 1)
                with m.Else():
                    m.d.sync += mem[base].eq(xreg)
                    m.d.sync += acc.eq(acc + crom[0] * xreg)
                    m.next = "EMIT"
            with m.State("EMIT"):
                m.d.sync += [self.out.eq(acc >> self.out_shift),
                             self.out_chan.eq(chan), self.out_valid.eq(1),
                             self.busy.eq(0)]
                m.next = "IDLE"
        return m


class TdmFirEngineBRAM(TdmFirEngine):
    """BRAM-backed folded FIR: identical behaviour/interface to :class:`TdmFirEngine`
    (same ``model``), but the per-channel delay lines live in block RAM instead of a
    flat :class:`Array` of registers.

    The parent keeps ``n_channels * ntaps`` words as fabric flip-flops and shifts the
    whole line each sample -- fine for a tiny FIR, but the 119-tap cleanup over a few
    channels alone overflows the Z-7010's 35 k FFs (Vivado will not infer RAM from the
    shift pattern). This version stores each channel's history as a **circular buffer**
    in a single ``Memory`` (one R + one W port -> simple-dual-port BRAM) with a
    per-channel write pointer, so no data is shifted: each sample does one write plus
    ``ntaps`` reads. Bit-exact to the register version (verified against the shared
    ``model``).
    """

    def elaborate(self, platform):
        m = Module()
        nt, n = self.ntaps, self.n
        mem = Memory(shape=signed(self.iw), depth=n * nt, init=[0] * (n * nt))
        m.submodules.mem = mem
        rp = mem.read_port()
        wp = mem.write_port()
        crom = Array(Const(c, signed(self.cw)) for c in self.coeffs)
        wptr = Array(Signal(range(nt), name=f"wptr{c}") for c in range(n))

        chan = Signal.like(self.in_chan)
        xreg = Signal(signed(self.iw))
        p = Signal(range(nt))                 # write position for this channel
        base = Signal(range(max(2, n * nt)))
        acc = Signal(signed(self.acc_w))
        ii = Signal(range(nt + 1))            # tap currently being read-issued
        ad = Signal(range(nt + 1))            # tap currently being accumulated
        ad_v = Signal()                       # an accumulate is valid this cycle

        m.d.comb += base.eq(chan * nt)
        # read address for tap ``ii``: history position (p - ii) mod nt
        pos = Signal(range(nt))
        with m.If(p >= ii):
            m.d.comb += pos.eq(p - ii)
        with m.Else():
            m.d.comb += pos.eq(p + nt - ii)
        m.d.comb += [rp.addr.eq(base + pos), rp.en.eq(1)]

        with m.FSM():
            with m.State("IDLE"):
                m.d.sync += self.out_valid.eq(0)
                with m.If(self.in_valid & ~self.busy):
                    m.d.sync += [chan.eq(self.in_chan), xreg.eq(self.x_in),
                                 p.eq(wptr[self.in_chan]), acc.eq(0),
                                 self.busy.eq(1), ii.eq(nt - 1), ad_v.eq(0)]
                    m.next = "RUN"
            with m.State("RUN"):
                # accumulate the tap whose read was issued last cycle
                with m.If(ad_v):
                    m.d.sync += acc.eq(acc + crom[ad] * rp.data)
                with m.If(ii >= 1):
                    m.d.sync += [ad.eq(ii), ad_v.eq(1), ii.eq(ii - 1)]
                with m.Else():
                    m.d.sync += ad_v.eq(0)
                    m.next = "FINISH"
            with m.State("FINISH"):
                # tap 0 is the freshest sample (still in xreg, not yet in RAM)
                m.d.sync += acc.eq(acc + crom[0] * xreg)
                m.d.comb += [wp.addr.eq(base + p), wp.data.eq(xreg), wp.en.eq(1)]
                m.d.sync += wptr[chan].eq(Mux(p == nt - 1, 0, p + 1))
                m.next = "EMIT"
            with m.State("EMIT"):
                m.d.sync += [self.out.eq(acc >> self.out_shift),
                             self.out_chan.eq(chan), self.out_valid.eq(1),
                             self.busy.eq(0)]
                m.next = "IDLE"
        return m


# ---------------------------------------------------------------------------
# Coefficient design
# ---------------------------------------------------------------------------

def design_lowpass(ntaps: int, cutoff: float, coeff_width: int = 16):
    """Unity-DC-gain low-pass; return (int taps, out_shift) for ~unity gain."""
    h = scipy.signal.firwin(ntaps, cutoff)               # sum(h) = 1
    scale = (2 ** (coeff_width - 1) - 1) / np.max(np.abs(h))
    h_int = np.round(h * scale).astype(int)
    out_shift = max(0, int(round(math.log2(abs(int(np.sum(h_int)))))))
    return h_int, out_shift


def design_halfband(ntaps: int, coeff_width: int = 16):
    """Halfband decimate-by-2 low-pass (cutoff = quarter sample rate).

    Every other tap (bar the centre) is forced to zero, so only ~half the taps
    cost a multiply -- the key saving of a multistage decimator. Returns
    (int taps, out_shift, n_nonzero_taps)."""
    assert ntaps % 2 == 1, "halfband length must be odd"
    h = scipy.signal.firwin(ntaps, 0.5)
    c = ntaps // 2
    for n in range(ntaps):
        if n != c and (n - c) % 2 == 0:
            h[n] = 0.0
    h = h / h.sum()                                      # restore unity DC gain
    scale = (2 ** (coeff_width - 1) - 1) / np.max(np.abs(h))
    h_int = np.round(h * scale).astype(int)
    out_shift = max(0, int(round(math.log2(abs(int(np.sum(h_int)))))))
    return h_int, out_shift, int(np.count_nonzero(h_int))


def front_end_stages():
    """Two-stage halfband decimate-by-4 front end (the cheap multistage option).

    Each stage is a halfband decimate-by-2: stage 1 at the high (oversampled) rate
    is short and nulls ~half its taps; stage 2 runs at half the rate. Returns a list
    of {coeffs, decim, out_shift, nnz} usable by MultiStageDecimator and the folded
    cost model."""
    h1, s1, nnz1 = design_halfband(11)
    h2, s2, nnz2 = design_halfband(31)
    return [
        {"coeffs": h1, "decim": 2, "out_shift": s1, "nnz": nnz1},
        {"coeffs": h2, "decim": 2, "out_shift": s2, "nnz": nnz2},
    ]


def cic_response(fo, R, S):
    """CIC magnitude response at output-normalized frequency ``fo`` in [0, 0.5]."""
    fo = np.asarray(fo, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = (np.sin(np.pi * fo) / (R * np.sin(np.pi * fo / R))) ** S
    h = np.where(fo == 0, 1.0, h)
    return np.abs(h)


def design_cic_compensation(R: int, S: int, ntaps: int, fp: float, fs: float,
                            coeff_width: int = 16):
    """FIR that inverts the CIC passband droop (fp/fs are fractions of Nyquist)."""
    freq = np.linspace(0.0, 1.0, 1025)
    fo = freq * 0.5
    inv = 1.0 / cic_response(fo, R, S)
    desired = np.zeros_like(freq)
    ft = fp + 0.7 * (fs - fp)                 # transition ends here; full stop by fs
    pb = freq <= fp
    sb = freq >= ft
    tb = (~pb) & (~sb)
    desired[pb] = inv[pb]
    edge = inv[pb][-1] if pb.any() else 1.0
    desired[tb] = edge * (ft - freq[tb]) / (ft - fp)
    h = scipy.signal.firwin2(ntaps, freq, desired, window=("kaiser", 8.6))
    scale = (2 ** (coeff_width - 1) - 1) / np.max(np.abs(h))
    h_int = np.round(h * scale).astype(int)
    out_shift = max(0, int(round(math.log2(abs(int(np.sum(h_int)))))))
    return h_int, out_shift


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _run_fir(dut: FIRStage, x):
    got = []

    async def bench(ctx):
        for v in x:
            ctx.set(dut.x_in, int(v))
            ctx.set(dut.clken, 1)
            await ctx.tick()
            ctx.set(dut.clken, 0)
            if ctx.get(dut.strobe_out):
                got.append(ctx.get(dut.y_out))
            await ctx.tick()                          # idle gap (clken low)
            if ctx.get(dut.strobe_out):
                got.append(ctx.get(dut.y_out))

    sim = Simulator(dut)
    sim.add_clock(SYNC_PERIOD)
    sim.add_testbench(bench)
    sim.run()
    return np.array(got, dtype=np.int64)


def _verify_firstage():
    rng = np.random.default_rng(11)
    results = []
    for decim in (1, 4):
        coeffs, out_shift = design_lowpass(23, 0.3, coeff_width=12)
        dut = FIRStage(coeffs, decim, in_width=14, out_shift=out_shift)
        n = 500
        lim = 2 ** 13 - 1
        x = rng.integers(-lim, lim, size=n).astype(np.int64)
        got = _run_fir(dut, x)
        exp = FIRStage.model(x, coeffs, decim, out_shift)
        k = min(len(got), len(exp))
        np.testing.assert_array_equal(got[:k], exp[:k])
        results.append((decim, k))
    return results


def _front_end_response():
    """Front-end keeps the whole capture window flat and rejects beyond it."""
    R_fe = 4                                  # oversample factor (e.g. 61.44->15.36)
    # Flat passband over the channel region, transition in the guard, brickwall by
    # the output Nyquist (0.5/R_fe) to prevent decimation aliasing. The channels
    # fill the central ~77% of the band (capture_window.py), so the guard above
    # them gives the FIR room to roll off.
    coeffs, out_shift = design_lowpass(95, 0.95 / R_fe, coeff_width=16)
    # sweep complex tones across +/- input Nyquist; measure output magnitude
    def amp_at(f):
        n = 1500
        nn = np.arange(n)
        amp = 1500
        iq = amp * np.exp(1j * 2 * np.pi * f * nn)
        di = FIRStage(coeffs, R_fe, in_width=12, out_shift=out_shift)
        dq = FIRStage(coeffs, R_fe, in_width=12, out_shift=out_shift)
        gi = _run_fir(di, np.round(iq.real))
        gq = _run_fir(dq, np.round(iq.imag))
        k = min(len(gi), len(gq))
        z = gi[:k] + 1j * gq[:k]
        return np.sqrt(np.mean(np.abs(z[len(z) // 3:]) ** 2))
    # channel region (central ~77% of the no-alias edge 0.5/R_fe) should be ~flat;
    # beyond the output Nyquist (alias region) strongly rejected
    edge = 0.5 / R_fe
    f_pass = [0.0, edge * 0.4, edge * 0.7]
    f_stop = [edge * 1.3, 0.35]
    pass_amps = [amp_at(f) for f in f_pass]
    stop_amps = [amp_at(f) for f in f_stop]
    ref = pass_amps[0]
    pass_db = [20 * np.log10(a / ref + 1e-12) for a in pass_amps]
    stop_db = [20 * np.log10(a / ref + 1e-12) for a in stop_amps]
    return pass_db, stop_db


def _compensation_response():
    """CIC + compensation FIR: flatten the passband, keep stopband rejection."""
    # CIC droops gently and rolls off slowly near Nyquist, so the compensation FIR
    # does double duty: invert the passband droop AND provide the sharp channel
    # selectivity (adjacent-channel rejection) the CIC cannot. Parameters chosen so
    # the droop is real (a wide final-stage passband) -- where compensation matters.
    R_ch, S = 8, 5
    fp, fs = 0.35, 0.60                       # fraction of channel Nyquist
    comp, comp_shift = design_cic_compensation(R_ch, S, 119, fp, fs, coeff_width=16)

    # model responses on an output-normalized grid
    freq = np.linspace(0.0, 1.0, 513)
    fo = freq * 0.5
    cic = cic_response(fo, R_ch, S)
    w, hcomp = scipy.signal.freqz(comp, worN=freq * np.pi)
    hcomp = np.abs(hcomp)
    hcomp /= hcomp[0]                          # normalize DC to 1
    combined = cic * hcomp

    pb = freq <= fp
    sb = freq >= fs
    cic_ripple_db = 20 * np.log10(cic[pb].max() / cic[pb].min())
    comb_ripple_db = 20 * np.log10(combined[pb].max() / combined[pb].min())
    stop_atten_db = -20 * np.log10(combined[sb].max() + 1e-12)
    return (cic_ripple_db, comb_ripple_db, stop_atten_db,
            (freq, cic, hcomp, combined, fp, fs))


def _run_stream(dut, x):
    """Drive a clken/x_in -> strobe_out/y_out module continuously; collect outputs."""
    got = []

    async def bench(ctx):
        ctx.set(dut.clken, 1)
        for v in x:
            ctx.set(dut.x_in, int(v))
            await ctx.tick()
            if ctx.get(dut.strobe_out):
                got.append(ctx.get(dut.y_out))
        for _ in range(4):                            # flush trailing registered output
            await ctx.tick()
            if ctx.get(dut.strobe_out):
                got.append(ctx.get(dut.y_out))

    sim = Simulator(dut)
    sim.add_clock(SYNC_PERIOD)
    sim.add_testbench(bench)
    sim.run()
    return np.array(got, dtype=np.int64)


def _chain_output_rms(f_tone, f_mix, *, R_fe, fe_coeffs, fe_shift,
                      R_ch, S, comp, comp_shift):
    """Full HW pipeline: front-end decimate -> NCO mix -> CIC -> compensation FIR.

    Returns steady-state output RMS for a single complex input tone at ``f_tone``
    (cyc/sample at the ADC rate), tuned by an NCO at ``f_mix``.
    """
    n_adc = 4096
    n = np.arange(n_adc)
    amp = 1500
    iq = amp * np.exp(1j * 2 * np.pi * f_tone * n)

    # 1) shared front-end decimator (HW, two real FIRStage)
    fe_i = _run_stream(FIRStage(fe_coeffs, R_fe, 12, fe_shift), np.round(iq.real))
    fe_q = _run_stream(FIRStage(fe_coeffs, R_fe, 12, fe_shift), np.round(iq.imag))
    k = min(len(fe_i), len(fe_q))
    fe = fe_i[:k] + 1j * fe_q[:k]

    # 2) NCO mix to baseband (frequency shift; f_mix referred to the working rate)
    nn = np.arange(k)
    mixed = fe * np.exp(-1j * 2 * np.pi * (f_mix * R_fe) * nn)
    mi = np.round(mixed.real).astype(np.int64)
    mq = np.round(mixed.imag).astype(np.int64)
    iw = int(max(np.abs(np.concatenate([mi, mq])).max(), 1)).bit_length() + 2

    # 3) per-channel CIC decimator (HW, two real)
    ci_dut, cq_dut = CICDecimator(iw, R_ch, S), CICDecimator(iw, R_ch, S)
    cw = ci_dut.acc_w
    ci = _run_stream(ci_dut, mi)
    cq = _run_stream(cq_dut, mq)
    k2 = min(len(ci), len(cq))

    # 4) per-channel compensation FIR (HW, decim=1, two real)
    oi = _run_stream(FIRStage(comp, 1, cw, comp_shift), ci[:k2])
    oq = _run_stream(FIRStage(comp, 1, cw, comp_shift), cq[:k2])
    k3 = min(len(oi), len(oq))
    z = oi[:k3] + 1j * oq[:k3]
    z = z[len(z) // 2:]                               # discard transient
    return np.sqrt(np.mean(np.abs(z) ** 2))


def _end_to_end_chain():
    R_fe = 4
    fe_coeffs, fe_shift = design_lowpass(95, 0.95 / R_fe, coeff_width=16)
    R_ch, S = 8, 5
    fp, fs = 0.35, 0.60
    comp, comp_shift = design_cic_compensation(R_ch, S, 119, fp, fs, coeff_width=16)

    f0 = 0.03                                         # channel center (cyc/sample @ ADC)
    common = dict(R_fe=R_fe, fe_coeffs=fe_coeffs, fe_shift=fe_shift,
                  R_ch=R_ch, S=S, comp=comp, comp_shift=comp_shift)
    # on-channel: tone tuned to DC -> full passband gain
    on = _chain_output_rms(f0, f0, **common)
    # adjacent channel: tone one channel-spacing away -> lands in comp stopband
    adj_off = 0.40 / (R_fe * R_ch)                    # 0.40 of channel Nyquist
    adj = _chain_output_rms(f0 + adj_off, f0, **common)
    rej_db = 20 * np.log10(on / (adj + 1e-12))
    return on, adj, rej_db


def _run_cplx(dut, re, im):
    """Drive a complex clken/re_in/im_in -> strobe_out/re_out/im_out module."""
    got_re, got_im = [], []

    async def bench(ctx):
        ctx.set(dut.clken, 1)
        for r, i in zip(re, im):
            ctx.set(dut.re_in, int(r))
            ctx.set(dut.im_in, int(i))
            await ctx.tick()
            if ctx.get(dut.strobe_out):
                got_re.append(ctx.get(dut.re_out))
                got_im.append(ctx.get(dut.im_out))
        for _ in range(8):
            await ctx.tick()
            if ctx.get(dut.strobe_out):
                got_re.append(ctx.get(dut.re_out))
                got_im.append(ctx.get(dut.im_out))

    sim = Simulator(dut)
    sim.add_clock(SYNC_PERIOD)
    sim.add_testbench(bench)
    sim.run()
    k = min(len(got_re), len(got_im))
    return np.array(got_re[:k]) + 1j * np.array(got_im[:k])


def _multistage_front_end():
    """Multistage (halfband cascade) front end: bit-exact, flat window, cheap."""
    stages = front_end_stages()
    R_fe = 4

    # 1) bit-exact: random complex input, HW cascade == cascaded model
    rng = np.random.default_rng(7)
    n = 2000
    lim = 2 ** 11 - 1
    re = rng.integers(-lim, lim, size=n).astype(np.int64)
    im = rng.integers(-lim, lim, size=n).astype(np.int64)
    dut = MultiStageDecimator(stages, in_width=12)
    got = _run_cplx(dut, re, im)
    mre, mim = MultiStageDecimator.model(re, im, stages)
    kk = min(len(got), len(mre))
    exact = np.array_equal(got[:kk].real.astype(np.int64), mre[:kk]) and \
        np.array_equal(got[:kk].imag.astype(np.int64), mim[:kk])

    # 2) flatness/rejection: sweep complex tones through the HW cascade
    def amp_at(f):
        nn = np.arange(2400)
        iq = 1500 * np.exp(1j * 2 * np.pi * f * nn)
        z = _run_cplx(MultiStageDecimator(stages, 12),
                      np.round(iq.real), np.round(iq.imag))
        return np.sqrt(np.mean(np.abs(z[len(z) // 3:]) ** 2))

    edge = 0.5 / R_fe                              # final no-alias edge (cyc/sample)
    f_pass = [0.0, edge * 0.4, edge * 0.7]         # channel region (central ~77%)
    f_stop = [edge * 1.3, 0.30]                    # alias region
    ref = amp_at(f_pass[0])
    pass_db = [20 * np.log10(amp_at(f) / ref + 1e-12) for f in f_pass]
    stop_db = [20 * np.log10(amp_at(f) / ref + 1e-12) for f in f_stop]

    # folded DSP cost: only nonzero halfband taps cost a multiply.
    f_clk, fs_adc = 62.5e6, 56e6                   # ~oversampled capture (4x 14 MHz)
    rate = fs_adc
    folded = 0.0
    for st in stages:
        rate /= st["decim"]                        # stage output rate
        folded += 2 * st["nnz"] * rate / f_clk     # x2 = complex
    return exact, pass_db, stop_db, stages, folded


def _run_tdm_fir(eng, samples, coeffs, out_shift):
    got = []

    async def bench(ctx):
        for ch, x in samples:
            ctx.set(eng.in_chan, int(ch))
            ctx.set(eng.x_in, int(x))
            ctx.set(eng.in_valid, 1)
            await ctx.tick()
            ctx.set(eng.in_valid, 0)
            while True:
                await ctx.tick()
                if ctx.get(eng.out_valid):
                    got.append((ctx.get(eng.out_chan), ctx.get(eng.out)))
                    break

    sim = Simulator(eng)
    sim.add_clock(SYNC_PERIOD)
    sim.add_testbench(bench)
    sim.run()
    exp = TdmFirEngine.model(samples, coeffs, eng.n, out_shift)
    return all(g[0] == e[0] and g[1] == e[1] for g, e in zip(got, exp)), len(got)


def _verify_tdm_fir():
    """Folded TDM FIR engines (register + BRAM) == per-channel parallel FIR."""
    coeffs, out_shift = design_lowpass(9, 0.3, coeff_width=12)
    n_ch = 4
    rng = np.random.default_rng(3)
    n = 120
    chans = rng.integers(0, n_ch, size=n)
    lim = 2 ** 11 - 1
    xs = rng.integers(-lim, lim, size=n).astype(np.int64)
    samples = list(zip(chans.tolist(), xs.tolist()))

    eng = TdmFirEngine(coeffs, n_ch, in_width=12, out_shift=out_shift)
    ok_ff, n_out = _run_tdm_fir(eng, samples, coeffs, out_shift)
    eng_b = TdmFirEngineBRAM(coeffs, n_ch, in_width=12, out_shift=out_shift)
    ok_bram, _ = _run_tdm_fir(eng_b, samples, coeffs, out_shift)
    return ok_ff and ok_bram, n_out, eng.ntaps, n_ch


def main():
    res = _verify_firstage()
    for decim, k in res:
        print(f"[FIRStage decim={decim}] HW == model: PASS ({k} outputs)")

    pass_db, stop_db = _front_end_response()
    print("\n[FrontEndDecimator] shared complex FIR decimator (oversample 4x):")
    print(f"  in-window passband ripple : "
          f"{max(pass_db) - min(pass_db):.2f} dB  (tones {pass_db})")
    print(f"  out-of-window rejection   : {-max(stop_db):.1f} dB worst-case")
    assert max(pass_db) - min(pass_db) < 1.0, "front-end window not flat"
    assert max(stop_db) < -40, "front-end does not reject out-of-window"
    print("  => front-end holds the whole window flat and rejects beyond it.")

    cic_r, comb_r, stop_a, plotdata = _compensation_response()
    print("\n[CompensationFIR] CIC droop correction (R=8, 5 stages):")
    print(f"  CIC-only passband droop   : {cic_r:.2f} dB")
    print(f"  CIC + comp passband ripple: {comb_r:.2f} dB")
    print(f"  combined stopband rejection: {stop_a:.1f} dB")
    assert comb_r < cic_r, "compensation did not improve passband flatness"
    assert comb_r < 0.5, "compensated passband not flat enough"
    assert stop_a > 40, "insufficient channel selectivity"
    print(f"  => droop {cic_r:.2f} dB -> {comb_r:.2f} dB flat, "
          f"{stop_a:.0f} dB adjacent-channel rejection.")

    on, adj, rej = _end_to_end_chain()
    print("\n[end-to-end HW] front-end -> NCO mix -> per-channel CIC -> comp FIR:")
    print(f"  on-channel output RMS     : {on:.1f}")
    print(f"  adjacent-channel RMS      : {adj:.1f}")
    print(f"  measured rejection        : {rej:.1f} dB")
    assert rej > 40, "end-to-end channel does not reject the adjacent channel"
    print("  => a tone on the channel passes; one channel-spacing away is rejected.")

    exact, ms_pass, ms_stop, stages, folded = _multistage_front_end()
    print("\n[MultiStageDecimator] halfband decimate-by-4 front end (the cheap option "
          "if we oversample; the AD936x can also deliver the working rate directly):")
    taps = "+".join(str(len(s["coeffs"])) for s in stages)
    nnz = "+".join(str(s["nnz"]) for s in stages)
    print(f"  stages: {taps} taps ({nnz} nonzero after halfband nulling)")
    print(f"  HW == cascaded model: {'PASS' if exact else 'FAIL'}")
    print(f"  channel-region ripple : {max(ms_pass) - min(ms_pass):.2f} dB")
    print(f"  out-of-window rejection: {-max(ms_stop):.1f} dB")
    print(f"  folded DSP48E1 (complex, @62.5 MHz): ~{folded:.0f}  "
          f"(vs ~43 for one long FIR)")
    assert exact, "multistage front end not bit-exact"
    assert max(ms_pass) - min(ms_pass) < 1.0, "multistage window not flat"
    assert max(ms_stop) < -40, "multistage does not reject out-of-window"

    ok, n_out, ntaps, n_ch = _verify_tdm_fir()
    print(f"\n[TdmFirEngine] folded cleanup FIR ({ntaps} taps, {n_ch} channels, "
          f"one MAC):")
    print(f"  HW == per-channel parallel FIR: {'PASS' if ok else 'FAIL'} "
          f"({n_out} interleaved outputs; register + BRAM delay-line variants)")
    assert ok, "folded TDM FIR not bit-exact to per-channel reference"
    print("  => one multiply-accumulate, iterated over taps, serves all channels; "
          "BRAM variant keeps per-channel delay lines in block RAM.")

    print("\nPASS: shared front-end decimator + per-channel CIC droop compensation "
          "verified (bit-exact FIR; flat window; flat channel passband + rejection); "
          "multistage front end + folded TDM cleanup FIR verified.")
    _save_plot(plotdata)


def _save_plot(plotdata):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    freq, cic, hcomp, combined, fp, fs = plotdata
    OUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    db = lambda h: 20 * np.log10(h + 1e-12)
    ax.plot(freq, db(cic), label="CIC only (droops)")
    ax.plot(freq, db(hcomp), label="compensation FIR", ls=":")
    ax.plot(freq, db(combined), label="CIC + comp (flat)", lw=2)
    ax.axvspan(0, fp, color="tab:green", alpha=0.10, label="channel passband")
    ax.axvline(fs, color="tab:red", ls="--", alpha=0.6, label="stopband edge")
    ax.set_xlabel("normalized frequency (channel output rate; 1.0 = Nyquist)")
    ax.set_ylabel("dB")
    ax.set_ylim(-80, 10)
    ax.set_title("Per-channel CIC droop vs CIC + compensation FIR")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "channelizer_chain.png"
    fig.savefig(path, dpi=120)
    print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
