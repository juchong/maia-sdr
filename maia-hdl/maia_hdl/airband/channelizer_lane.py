#!/usr/bin/env python3
"""Time-multiplexed channelizer lane prototype (handoff §4.2 / §7 step 7).

The feasibility study (``feasibility_25ch.py``) showed that the AM channels fit the
Z-7010 only if the per-channel datapaths are **time-multiplexed**: the PL runs at
62.5 MHz while each channel needs only ~tens of ksps, so one physical datapath can
serve many channels by iterating over them between input samples (§4.2).

This module builds one such **lane** and proves the idea in simulation. A single
NCO + complex-mixer + CIC-decimator datapath is shared across ``n_channels``
channels; all per-channel state (NCO phase, CIC integrator/comb registers,
decimation counter) lives in addressable arrays indexed by the channel counter.
One wideband input sample is broadcast to every channel; the lane processes the
channels one-per-cycle, so it sustains the channel set as long as

    n_channels  <=  cycles_between_input_samples = F_clk / R_in.

Datapath per channel:

    wideband IQ ─► [NCO mix: × e^(-jθ_c)] ─► [CIC decimate by R] ─► baseband IQ_c

The NCO uses a shared signed sine ROM (quarter-wave offset for cosine). The mixer
is a complex multiply (downconversion). The CIC is the same multiplier-free
cascaded-integrator-comb used in the AM back-end (``am_audio.CICDecimator``), but
with per-channel state so the one datapath is reused.

What this prototype is / isn't:
  * IS: a bit-exact, resource-measurable realization of the time-multiplexed
    channelization the feasibility model assumed (validates the LUT/FF/DSP/BRAM
    per-lane numbers in real HDL, via ``synth_estimate``-style Yosys).
  * ISN'T: the final channel filter. A CIC has passband droop; the real lane adds
    a short cleanup/compensation FIR (the ``FIR_CLEANUP`` term in the model) and a
    shared front-end decimator. Those are the next increment.

Run:  python hdl/channelizer_lane.py
"""
from __future__ import annotations

import math
import pathlib

import numpy as np
from amaranth import Array, Cat, Const, Elaboratable, Module, Signal, signed
from amaranth.lib.memory import Memory
from amaranth.sim import Simulator

from ddc_tune_decimate import SYNC_PERIOD

OUT_DIR = pathlib.Path(__file__).parent / "out"


def sine_rom(addr_bits: int, width: int) -> list[int]:
    """Signed full-wave sine table: ``rom[k] = round(A*sin(2*pi*k/2**addr_bits))``."""
    n = 1 << addr_bits
    amp = (1 << (width - 1)) - 1
    return [int(round(amp * math.sin(2 * math.pi * k / n))) for k in range(n)]


class TdmDdcLane(Elaboratable):
    """One time-multiplexed DDC lane: NCO + complex mixer + CIC, shared over N ch.

    Control / data interface (all in the ``sync`` domain):

    Config (load a channel's NCO frequency word):
        freq_wren  : Signal(), in
        freq_waddr : channel index, in
        freq_wdata : signed(nco_width), in   -- cycles/sample × 2**nco_width

    Input (one wideband complex sample, broadcast to all channels):
        in_valid   : Signal(), in    -- pulse to start a processing pass
        re_in,im_in: signed(in_width), in
        busy       : Signal(), out   -- high while iterating channels (ignore in_valid)

    Output (decimated baseband sample for one channel):
        out_valid  : Signal(), out
        out_chan   : channel index, out
        out_re,out_im : signed(acc_width), out   -- full-CIC-gain (caller scales)

    Parameters
    ----------
    n_channels : number of channels multiplexed onto this datapath.
    decimation : CIC decimation factor R (per channel).
    in_width   : signed wideband input sample width.
    nco_width  : NCO phase-accumulator width.
    lut_addr_bits : sine ROM address bits (phase quantization).
    lut_width  : signed sine ROM sample width (sets mixer normalization shift).
    stages     : CIC order N.
    """

    def __init__(self, *, n_channels: int, decimation: int, in_width: int = 12,
                 nco_width: int = 24, lut_addr_bits: int = 10, lut_width: int = 12,
                 stages: int = 3):
        assert n_channels >= 1 and decimation >= 2 and stages >= 1
        assert lut_addr_bits >= 2
        self.n_channels = n_channels
        self.decimation = decimation
        self.in_width = in_width
        self.nco_width = nco_width
        self.lut_addr_bits = lut_addr_bits
        self.lut_width = lut_width
        self.stages = stages

        self.mix_shift = lut_width - 1
        self.mix_width = in_width + 2                  # |x·cos+y·sin|/A <= ~sqrt2·|x|
        self.growth = math.ceil(stages * math.log2(decimation))
        self.acc_w = self.mix_width + self.growth

        self.freq_wren = Signal()
        self.freq_waddr = Signal(range(n_channels))
        self.freq_wdata = Signal(signed(nco_width))

        self.in_valid = Signal()
        self.re_in = Signal(signed(in_width))
        self.im_in = Signal(signed(in_width))
        self.busy = Signal()

        self.out_valid = Signal()
        self.out_chan = Signal(range(n_channels))
        self.out_re = Signal(signed(self.acc_w), reset_less=True)
        self.out_im = Signal(signed(self.acc_w), reset_less=True)

    # -- bit-exact reference model -----------------------------------------
    def model(self, samples, freqs):
        """Reference output per channel for a wideband IQ input sequence.

        ``samples`` : iterable of (re, im) ints (the broadcast wideband stream).
        ``freqs``   : per-channel signed NCO frequency words (cycles/sample
                      × 2**nco_width).
        Returns ``{chan: (out_re_array, out_im_array)}`` matching the hardware,
        using exactly the hardware's fixed-point arithmetic.
        """
        rom = sine_rom(self.lut_addr_bits, self.lut_width)
        nmask = (1 << self.nco_width) - 1
        amask = (1 << self.lut_addr_bits) - 1
        quarter = 1 << (self.lut_addr_bits - 2)
        shift_phase = self.nco_width - self.lut_addr_bits
        out = {}
        for c, f in enumerate(freqs):
            phase = 0
            mi, mq = [], []
            for (re, im) in samples:
                a = (phase >> shift_phase) & amask
                s = rom[a]
                co = rom[(a + quarter) & amask]
                # downconvert: (re+j im)·(cos - j sin)
                oi = (re * co + im * s) >> self.mix_shift
                oq = (im * co - re * s) >> self.mix_shift
                mi.append(oi)
                mq.append(oq)
                phase = (phase + (f & nmask)) & nmask
            out[c] = (self._cic_model(mi), self._cic_model(mq))
        return out

    def _cic_model(self, x):
        acc = np.asarray(x, dtype=np.int64)
        for _ in range(self.stages):
            acc = np.cumsum(acc)
        dec = acc[self.decimation - 1::self.decimation].copy()
        for _ in range(self.stages):
            dec = dec - np.concatenate(([np.int64(0)], dec[:-1]))
        return dec

    # -- hardware ----------------------------------------------------------
    def elaborate(self, platform):
        m = Module()
        C, S = self.n_channels, self.stages

        # Shared sine ROM (read-only Array of constants).
        rom = Array(Const(v, signed(self.lut_width))
                    for v in sine_rom(self.lut_addr_bits, self.lut_width))

        # Per-channel state (addressable by the channel counter).
        phase = Array(Signal(self.nco_width, name=f"phase{c}") for c in range(C))
        freq = Array(Signal(signed(self.nco_width), name=f"freq{c}")
                     for c in range(C))
        deccnt = Array(Signal(range(self.decimation), name=f"deccnt{c}")
                       for c in range(C))
        integ_i = [Array(Signal(signed(self.acc_w), name=f"ii{st}_{c}")
                         for c in range(C)) for st in range(S)]
        integ_q = [Array(Signal(signed(self.acc_w), name=f"iq{st}_{c}")
                         for c in range(C)) for st in range(S)]
        comb_i = [Array(Signal(signed(self.acc_w), name=f"ci{st}_{c}")
                        for c in range(C)) for st in range(S)]
        comb_q = [Array(Signal(signed(self.acc_w), name=f"cq{st}_{c}")
                        for c in range(C)) for st in range(S)]

        # NCO frequency config write.
        with m.If(self.freq_wren):
            m.d.sync += freq[self.freq_waddr].eq(self.freq_wdata)

        # Latched wideband sample + channel iterator.
        xi = Signal(signed(self.in_width))
        xq = Signal(signed(self.in_width))
        chan = Signal(range(C))
        proc = Signal()                      # processing a channel this cycle

        m.d.comb += self.busy.eq(proc)
        m.d.sync += self.out_valid.eq(0)     # default; pulsed below

        with m.If(~proc & self.in_valid):
            m.d.sync += [xi.eq(self.re_in), xq.eq(self.im_in),
                         chan.eq(0), proc.eq(1)]

        # ---- the shared datapath, evaluated for channel `chan` ----
        amask = (1 << self.lut_addr_bits) - 1
        quarter = 1 << (self.lut_addr_bits - 2)
        sin_addr = Signal(self.lut_addr_bits)
        cos_addr = Signal(self.lut_addr_bits)
        m.d.comb += [
            sin_addr.eq(phase[chan][self.nco_width - self.lut_addr_bits:]),
            cos_addr.eq(sin_addr + quarter),     # +90 deg, wraps mod 2**addr_bits
        ]
        sin_v = Signal(signed(self.lut_width))
        cos_v = Signal(signed(self.lut_width))
        m.d.comb += [sin_v.eq(rom[sin_addr]), cos_v.eq(rom[cos_addr])]

        # complex downconversion, normalized by the ROM amplitude (>> mix_shift)
        prod_i = Signal(signed(self.in_width + self.lut_width + 1))
        prod_q = Signal(signed(self.in_width + self.lut_width + 1))
        m.d.comb += [
            prod_i.eq(xi * cos_v + xq * sin_v),
            prod_q.eq(xq * cos_v - xi * sin_v),
        ]
        mix_i = Signal(signed(self.mix_width))
        mix_q = Signal(signed(self.mix_width))
        m.d.comb += [mix_i.eq(prod_i >> self.mix_shift),
                     mix_q.eq(prod_q >> self.mix_shift)]

        # integrator cascade (combinational chain over this channel's state)
        new_ii, new_iq = [], []
        chain_i, chain_q = mix_i, mix_q
        for st in range(S):
            ni = Signal(signed(self.acc_w), name=f"new_ii{st}")
            nq = Signal(signed(self.acc_w), name=f"new_iq{st}")
            m.d.comb += [ni.eq(integ_i[st][chan] + chain_i),
                         nq.eq(integ_q[st][chan] + chain_q)]
            new_ii.append(ni)
            new_iq.append(nq)
            chain_i, chain_q = ni, nq

        # decimation strobe for this channel
        dec_stb = Signal()
        m.d.comb += dec_stb.eq(proc & (deccnt[chan] == self.decimation - 1))

        # comb cascade (only meaningful on dec_stb), fed by the newest integrator
        comb_out_i, comb_out_q = chain_i, chain_q   # = new_ii[-1], new_iq[-1]
        cin_i, cin_q = [], []
        for st in range(S):
            di = Signal(signed(self.acc_w), name=f"comb_di{st}")
            dq = Signal(signed(self.acc_w), name=f"comb_dq{st}")
            m.d.comb += [di.eq(comb_out_i - comb_i[st][chan]),
                         dq.eq(comb_out_q - comb_q[st][chan])]
            cin_i.append(comb_out_i)
            cin_q.append(comb_out_q)
            comb_out_i, comb_out_q = di, dq

        with m.If(proc):
            # advance NCO and write back integrator state for this channel
            m.d.sync += phase[chan].eq(phase[chan] + freq[chan])
            for st in range(S):
                m.d.sync += [integ_i[st][chan].eq(new_ii[st]),
                             integ_q[st][chan].eq(new_iq[st])]
            with m.If(dec_stb):
                m.d.sync += deccnt[chan].eq(0)
                for st in range(S):
                    m.d.sync += [comb_i[st][chan].eq(cin_i[st]),
                                 comb_q[st][chan].eq(cin_q[st])]
                m.d.sync += [self.out_valid.eq(1), self.out_chan.eq(chan),
                             self.out_re.eq(comb_out_i),
                             self.out_im.eq(comb_out_q)]
            with m.Else():
                m.d.sync += deccnt[chan].eq(deccnt[chan] + 1)

            # advance / finish the channel sweep
            with m.If(chan == C - 1):
                m.d.sync += proc.eq(0)
            with m.Else():
                m.d.sync += chan.eq(chan + 1)

        return m


class TdmDdcLaneBRAM(TdmDdcLane):
    """BRAM-backed, fully pipelined TDM DDC lane: identical *values* to
    :class:`TdmDdcLane` (same ``model``), but the per-channel state lives in
    ``amaranth.lib.memory.Memory`` (block/distributed RAM, not fan-out flip-flops)
    and the datapath is split into a four-stage pipeline so it closes timing at the
    62.5 MHz PL clock.

    The single-cycle parent computes ROM mix + complex multiply + a 6-deep
    integrator/comb adder cascade in one clock -- a ~19 ns critical path that misses
    62.5 MHz. Here each channel flows through:

        READ : present read address = chan (memories registered, 1-cycle latency).
        MIX  : sin/cos ROM + complex downconvert multiply; write back NCO phase
               and the decimation counter.
        INTEG: 3-stage integrator cascade; write back integrators.
        COMB : 3-stage comb cascade; on a decimation strobe write back the combs
               and emit the channel's output.

    One channel enters per clock, so throughput is unchanged (one channel/cycle).
    Each channel appears once per input-sample sweep, and a channel is only revisited
    on the next input sample (>= many cycles later because the PL clock far exceeds
    the per-channel rate), so the read-then-write-back-3-cycles-later pipeline never
    hazards on its own state. Bit-exact to the parent (extra latency only; verified).
    """

    def elaborate(self, platform):
        m = Module()
        C, S = self.n_channels, self.stages
        aw = self.acc_w

        # Shared sine ROM (small, read combinationally by the phase MSBs).
        rom = Array(Const(v, signed(self.lut_width))
                    for v in sine_rom(self.lut_addr_bits, self.lut_width))

        # ---- per-channel state in RAM (one R + one W port each) ----
        def make(shape):
            return Memory(shape=shape, depth=C, init=[0] * C)

        phase_m = make(self.nco_width)
        freq_m = make(signed(self.nco_width))
        dec_m = make(range(self.decimation))
        ii_m = [make(signed(aw)) for _ in range(S)]
        iq_m = [make(signed(aw)) for _ in range(S)]
        ci_m = [make(signed(aw)) for _ in range(S)]
        cq_m = [make(signed(aw)) for _ in range(S)]
        all_m = [phase_m, freq_m, dec_m] + ii_m + iq_m + ci_m + cq_m
        for idx, mem in enumerate(all_m):
            m.submodules[f"mem{idx}"] = mem

        phase_rp, freq_rp, dec_rp = (phase_m.read_port(), freq_m.read_port(),
                                     dec_m.read_port())
        ii_rp = [mem.read_port() for mem in ii_m]
        iq_rp = [mem.read_port() for mem in iq_m]
        ci_rp = [mem.read_port() for mem in ci_m]
        cq_rp = [mem.read_port() for mem in cq_m]

        phase_wp, dec_wp = phase_m.write_port(), dec_m.write_port()
        ii_wp = [mem.write_port() for mem in ii_m]
        iq_wp = [mem.write_port() for mem in iq_m]
        ci_wp = [mem.write_port() for mem in ci_m]
        cq_wp = [mem.write_port() for mem in cq_m]
        freq_wp = freq_m.write_port()        # config only

        m.d.comb += [freq_wp.addr.eq(self.freq_waddr),
                     freq_wp.data.eq(self.freq_wdata),
                     freq_wp.en.eq(self.freq_wren)]

        # ---- sweep control: a channel index + valid flag flow down the pipe ----
        xi = Signal(signed(self.in_width))
        xq = Signal(signed(self.in_width))
        chan_r = Signal(range(C))
        proc_r = Signal()                    # READ stage active
        chan_x = Signal(range(C))            # MIX stage
        proc_x = Signal()
        chan_g = Signal(range(C))            # INTEG stage
        proc_g = Signal()
        chan_c = Signal(range(C))            # COMB stage
        proc_c = Signal()

        m.d.sync += [chan_x.eq(chan_r), proc_x.eq(proc_r),
                     chan_g.eq(chan_x), proc_g.eq(proc_x),
                     chan_c.eq(chan_g), proc_c.eq(proc_g)]
        m.d.comb += self.busy.eq(proc_r | proc_x | proc_g | proc_c)
        m.d.sync += self.out_valid.eq(0)

        with m.If(~self.busy & self.in_valid):
            m.d.sync += [xi.eq(self.re_in), xq.eq(self.im_in),
                         chan_r.eq(0), proc_r.eq(1)]
        with m.Elif(proc_r):
            with m.If(chan_r == C - 1):
                m.d.sync += proc_r.eq(0)
            with m.Else():
                m.d.sync += chan_r.eq(chan_r + 1)

        for rp in [phase_rp, freq_rp, dec_rp, *ii_rp, *iq_rp, *ci_rp, *cq_rp]:
            m.d.comb += [rp.addr.eq(chan_r), rp.en.eq(1)]

        # ===== MIX stage: read data is valid for chan_x =====
        quarter = 1 << (self.lut_addr_bits - 2)
        sin_addr = Signal(self.lut_addr_bits)
        cos_addr = Signal(self.lut_addr_bits)
        m.d.comb += [sin_addr.eq(phase_rp.data[self.nco_width - self.lut_addr_bits:]),
                     cos_addr.eq(sin_addr + quarter)]
        sin_v = Signal(signed(self.lut_width))
        cos_v = Signal(signed(self.lut_width))
        m.d.comb += [sin_v.eq(rom[sin_addr]), cos_v.eq(rom[cos_addr])]

        prod_i = Signal(signed(self.in_width + self.lut_width + 1))
        prod_q = Signal(signed(self.in_width + self.lut_width + 1))
        m.d.comb += [prod_i.eq(xi * cos_v + xq * sin_v),
                     prod_q.eq(xq * cos_v - xi * sin_v)]

        # registered MIX outputs + integrator state carried to INTEG
        mix_i = Signal(signed(self.mix_width))
        mix_q = Signal(signed(self.mix_width))
        ii_x = [Signal(signed(aw), name=f"ii_x{st}") for st in range(S)]
        iq_x = [Signal(signed(aw), name=f"iq_x{st}") for st in range(S)]
        # comb state must reach COMB (two more cycles): register twice
        ci_x = [Signal(signed(aw), name=f"ci_x{st}") for st in range(S)]
        cq_x = [Signal(signed(aw), name=f"cq_x{st}") for st in range(S)]
        dec_stb_x = Signal()
        m.d.comb += dec_stb_x.eq(proc_x & (dec_rp.data == self.decimation - 1))

        m.d.sync += [mix_i.eq(prod_i >> self.mix_shift),
                     mix_q.eq(prod_q >> self.mix_shift)]
        for st in range(S):
            m.d.sync += [ii_x[st].eq(ii_rp[st].data), iq_x[st].eq(iq_rp[st].data),
                         ci_x[st].eq(ci_rp[st].data), cq_x[st].eq(cq_rp[st].data)]

        # MIX write-backs: NCO phase advance + decimation counter
        m.d.comb += [
            phase_wp.addr.eq(chan_x),
            phase_wp.data.eq(phase_rp.data + freq_rp.data),
            phase_wp.en.eq(proc_x),
            dec_wp.addr.eq(chan_x),
            dec_wp.data.eq(dec_rp.data + 1),
            dec_wp.en.eq(proc_x),
        ]
        with m.If(dec_stb_x):
            m.d.comb += dec_wp.data.eq(0)

        # ===== INTEG stage: 3-stage integrator cascade for chan_g =====
        new_ii, new_iq = [], []
        chain_i, chain_q = mix_i, mix_q
        for st in range(S):
            ni = Signal(signed(aw), name=f"new_ii{st}")
            nq = Signal(signed(aw), name=f"new_iq{st}")
            m.d.comb += [ni.eq(ii_x[st] + chain_i), nq.eq(iq_x[st] + chain_q)]
            new_ii.append(ni)
            new_iq.append(nq)
            chain_i, chain_q = ni, nq

        for st in range(S):
            m.d.comb += [
                ii_wp[st].addr.eq(chan_g), ii_wp[st].data.eq(new_ii[st]),
                ii_wp[st].en.eq(proc_g),
                iq_wp[st].addr.eq(chan_g), iq_wp[st].data.eq(new_iq[st]),
                iq_wp[st].en.eq(proc_g),
            ]

        # registered INTEG outputs + comb state carried to COMB
        cin_in_i = Signal(signed(aw))        # newest integrator output -> comb input
        cin_in_q = Signal(signed(aw))
        ci_g = [Signal(signed(aw), name=f"ci_g{st}") for st in range(S)]
        cq_g = [Signal(signed(aw), name=f"cq_g{st}") for st in range(S)]
        dec_stb_g = Signal()                 # strobe aligned to INTEG stage
        m.d.sync += [cin_in_i.eq(chain_i), cin_in_q.eq(chain_q),
                     dec_stb_g.eq(dec_stb_x)]
        for st in range(S):
            m.d.sync += [ci_g[st].eq(ci_x[st]), cq_g[st].eq(cq_x[st])]

        # ===== COMB stage: 3-stage comb cascade for chan_c =====
        comb_out_i, comb_out_q = cin_in_i, cin_in_q
        cin_i, cin_q = [], []
        for st in range(S):
            di = Signal(signed(aw), name=f"comb_di{st}")
            dq = Signal(signed(aw), name=f"comb_dq{st}")
            m.d.comb += [di.eq(comb_out_i - ci_g[st]), dq.eq(comb_out_q - cq_g[st])]
            cin_i.append(comb_out_i)
            cin_q.append(comb_out_q)
            comb_out_i, comb_out_q = di, dq

        # strobe aligned to COMB stage (one more register: dec_stb_x already folds
        # in proc_x, which flows down the pipe, so dec_stb_c implies a valid channel)
        dec_stb_c = Signal()
        m.d.sync += dec_stb_c.eq(dec_stb_g)
        for st in range(S):
            m.d.comb += [
                ci_wp[st].addr.eq(chan_c), ci_wp[st].data.eq(cin_i[st]),
                ci_wp[st].en.eq(dec_stb_c),
                cq_wp[st].addr.eq(chan_c), cq_wp[st].data.eq(cin_q[st]),
                cq_wp[st].en.eq(dec_stb_c),
            ]
        with m.If(dec_stb_c):
            m.d.sync += [self.out_valid.eq(1), self.out_chan.eq(chan_c),
                         self.out_re.eq(comb_out_i), self.out_im.eq(comb_out_q)]

        return m


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _run_lane(dut: TdmDdcLane, samples, freqs):
    """Drive the lane over a wideband IQ stream; collect per-channel outputs."""
    out = {c: ([], []) for c in range(dut.n_channels)}

    async def bench(ctx):
        # load NCO frequencies
        ctx.set(dut.freq_wren, 1)
        for c, f in enumerate(freqs):
            ctx.set(dut.freq_waddr, c)
            ctx.set(dut.freq_wdata, int(f))
            await ctx.tick()
        ctx.set(dut.freq_wren, 0)
        await ctx.tick()

        for (re, im) in samples:
            ctx.set(dut.re_in, int(re))
            ctx.set(dut.im_in, int(im))
            ctx.set(dut.in_valid, 1)
            await ctx.tick()
            ctx.set(dut.in_valid, 0)
            # let the lane sweep all channels (busy de-asserts when done)
            while ctx.get(dut.busy):
                if ctx.get(dut.out_valid):
                    cc = ctx.get(dut.out_chan)
                    out[cc][0].append(ctx.get(dut.out_re))
                    out[cc][1].append(ctx.get(dut.out_im))
                await ctx.tick()
            if ctx.get(dut.out_valid):     # last channel's strobe lands here
                cc = ctx.get(dut.out_chan)
                out[cc][0].append(ctx.get(dut.out_re))
                out[cc][1].append(ctx.get(dut.out_im))

    sim = Simulator(dut)
    sim.add_clock(SYNC_PERIOD)
    sim.add_testbench(bench)
    sim.run()
    return {c: (np.array(i, np.int64), np.array(q, np.int64))
            for c, (i, q) in out.items()}


def _run_lane_paced(dut: TdmDdcLane, samples, freqs, gap: int):
    """Drive the lane feeding one input every ``gap`` cycles (>= the per-sample
    channel sweep + pipeline drain), collecting every ``out_valid`` strobe. Works
    for both the register lane and the pipelined BRAM lane."""
    out = {c: ([], []) for c in range(dut.n_channels)}

    async def bench(ctx):
        ctx.set(dut.freq_wren, 1)
        for c, f in enumerate(freqs):
            ctx.set(dut.freq_waddr, c)
            ctx.set(dut.freq_wdata, int(f))
            await ctx.tick()
        ctx.set(dut.freq_wren, 0)
        await ctx.tick()

        async def step():
            if ctx.get(dut.out_valid):
                cc = ctx.get(dut.out_chan)
                out[cc][0].append(ctx.get(dut.out_re))
                out[cc][1].append(ctx.get(dut.out_im))
            await ctx.tick()

        for (re, im) in samples:
            ctx.set(dut.re_in, int(re))
            ctx.set(dut.im_in, int(im))
            ctx.set(dut.in_valid, 1)
            await step()
            ctx.set(dut.in_valid, 0)
            for _ in range(gap - 1):
                await step()
        for _ in range(gap + 4):       # drain the pipeline tail
            await step()

    sim = Simulator(dut)
    sim.add_clock(SYNC_PERIOD)
    sim.add_testbench(bench)
    sim.run()
    return {c: (np.array(i, np.int64), np.array(q, np.int64))
            for c, (i, q) in out.items()}


def _verify_bram_bitexact():
    """The BRAM-backed lane must be bit-exact vs the same reference model used for
    the register lane (state moved FF->BRAM, datapath unchanged)."""
    params = dict(n_channels=5, decimation=8, in_width=12, nco_width=24,
                  lut_addr_bits=10, lut_width=12, stages=3)
    dut = TdmDdcLaneBRAM(**params)
    rng = np.random.default_rng(11)
    n = 600
    lim = 2 ** (dut.in_width - 1) - 1
    samples = rng.integers(-lim, lim, size=(n, 2)).astype(np.int64)
    fr = [int(round(f * 2 ** dut.nco_width))
          for f in (0.05, -0.1, 0.2, 0.0, -0.17)]

    hw = _run_lane_paced(dut, samples, fr, gap=dut.n_channels + 6)
    ref = dut.model([tuple(s) for s in samples], fr)
    nout = 0
    for c in range(dut.n_channels):
        for k in (0, 1):
            g, e = hw[c][k], ref[c][k]
            kk = min(len(g), len(e))
            assert kk > 0, f"no output for channel {c}"
            np.testing.assert_array_equal(g[:kk], e[:kk])
            nout = max(nout, kk)
    return dut.n_channels, dut.acc_w, nout


def _verify_bitexact():
    dut = TdmDdcLane(n_channels=4, decimation=8, in_width=12, nco_width=24,
                     lut_addr_bits=10, lut_width=12, stages=3)
    rng = np.random.default_rng(7)
    n = 800
    lim = 2 ** (dut.in_width - 1) - 1
    samples = rng.integers(-lim, lim, size=(n, 2)).astype(np.int64)
    # four arbitrary channel tunings (cycles/sample × 2**nco_width)
    fr = [int(round(f * 2 ** dut.nco_width)) for f in (0.05, -0.1, 0.2, 0.0)]

    hw = _run_lane(dut, samples, fr)
    ref = dut.model([tuple(s) for s in samples], fr)
    for c in range(dut.n_channels):
        for k in (0, 1):
            g, e = hw[c][k], ref[c][k]
            kk = min(len(g), len(e))
            assert kk > 0, f"no output for channel {c}"
            np.testing.assert_array_equal(g[:kk], e[:kk])
    return dut.n_channels, dut.acc_w, len(ref[0][0])


def _demo_tuning():
    """Functional check: each channel tunes to its own tone -> baseband; the lane
    rejects the other channels' tones (CIC selectivity)."""
    dut = TdmDdcLane(n_channels=4, decimation=16, in_width=12, nco_width=24,
                     lut_addr_bits=10, lut_width=12, stages=3)
    tones = [0.03, 0.11, -0.07, 0.22]          # cyc/sample, one per channel
    amps = [1500, 1500, 1500, 1500]
    n = 4096
    nn = np.arange(n)
    iq = np.zeros(n, complex)
    for f, a in zip(tones, amps):
        iq += a * np.exp(1j * 2 * np.pi * f * nn)
    iq /= len(tones)                            # keep within input range
    samples = np.stack([np.round(iq.real), np.round(iq.imag)], 1).astype(np.int64)
    fr = [int(round(f * 2 ** dut.nco_width)) for f in tones]

    hw = _run_lane(dut, samples, fr)
    # for each channel, |mean baseband| (tuned tone -> DC) vs AC ripple
    stats = {}
    for c in range(dut.n_channels):
        i, q = hw[c]
        z = (i + 1j * q)[len(i) // 4:]          # drop startup transient
        dc = np.mean(z)
        ripple = np.std(z) / (np.abs(dc) + 1e-9)
        stats[c] = (np.abs(dc), ripple)
    return tones, stats, hw


def main():
    C, acc_w, nout = _verify_bitexact()
    print(f"[bit-exact] TDM lane HW == reference model: PASS  "
          f"({C} channels share one datapath, {nout} decimated samples/ch, "
          f"acc width {acc_w} b)")

    Cb, accb, noutb = _verify_bram_bitexact()
    print(f"[bit-exact] BRAM-backed lane HW == reference model: PASS  "
          f"({Cb} ch, per-channel state in block RAM, 4-stage pipeline, "
          f"{noutb} samples/ch)")

    tones, stats, hw = _demo_tuning()
    print("\n[tuning] each channel tunes its own tone to baseband (CIC selectivity):")
    ok = True
    for c, (mag, ripple) in stats.items():
        print(f"  ch{c}  tone {tones[c]:+.3f} cyc/s  ->  |DC| {mag:8.0f}  "
              f"ripple/|DC| {ripple*100:5.1f}%")
        ok = ok and ripple < 0.25
    assert ok, "a channel did not produce a clean baseband (tuning/selectivity)"
    print("\nPASS: one time-multiplexed datapath channelizes N channels "
          "(bit-exact) and tunes each to baseband.")

    _save_plot(tones, hw)


def _save_plot(tones, hw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(exist_ok=True)
    n = len(hw)
    fig, ax = plt.subplots(n, 1, figsize=(9, 2.0 * n), sharex=True)
    for c in range(n):
        i, q = hw[c]
        ax[c].plot(i, label="I")
        ax[c].plot(q, label="Q")
        ax[c].set_ylabel(f"ch{c}\n{tones[c]:+.3f}")
        ax[c].grid(True, alpha=0.3)
        if c == 0:
            ax[c].set_title("TDM channelizer lane: per-channel baseband output "
                            "(one shared datapath)")
        ax[c].legend(loc="upper right", fontsize=8)
    ax[-1].set_xlabel("decimated sample")
    fig.tight_layout()
    path = OUT_DIR / "channelizer_lane.png"
    fig.savefig(path, dpi=110)
    print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
