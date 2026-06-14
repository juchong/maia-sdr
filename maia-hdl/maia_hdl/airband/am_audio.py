#!/usr/bin/env python3
"""AM audio back-end: DC-block + audio decimation, wired after the envelope
detector (handoff doc §7 step 5; completes the AM chain).

The envelope detector (``am_demod.EnvelopeMagnitude``) outputs ``|I+jQ|`` at the
DDC output rate. For airband AM that signal is::

    carrier_amplitude (DC)  +  voice audio (~0.3-3.4 kHz)

So the back-end is:

  1. ``DCBlock`` -- a one-pole high-pass (leaky-integrator DC estimate, then
     subtract). Strips the large carrier-amplitude DC term, leaving signed audio.
     Multiplier-free (one subtract + one arithmetic shift); cheap per channel.
  2. ``CICDecimator`` -- a multiplier-free CIC (cascaded-integrator-comb)
     decimator that low-pass filters and drops the rate down to the audio rate
     (8/16 ksps). No DSP48 / no coefficient memory -> very cheap to replicate
     across many channels on the Z-7010 (§4.2).

We put the DC-block *before* decimation so the big carrier DC does not inflate
the CIC integrator word growth.

This module also wires the whole single-channel chain together for an
end-to-end simulation::

    DDC (NCO mixer + FIR decimate) -> EnvelopeMagnitude -> DCBlock -> CICDecimator

and verifies that a frequency-offset AM tone is tuned to baseband, demodulated,
and recovered as clean low-rate audio.

Run:  python hdl/am_audio.py
"""
from __future__ import annotations

import math
import pathlib

import numpy as np
from amaranth import Cat, Elaboratable, Module, Mux, Signal, signed
from amaranth.sim import Simulator

from maia_hdl.ddc import DDC

from am_demod import EnvelopeMagnitude
from ddc_tune_decimate import (
    CLK3X_PERIOD,
    SYNC_PERIOD,
    CommonEdgeTb,
    IN_WIDTH,
    NCO_WIDTH,
    design_stage1,
)

OUT_DIR = pathlib.Path(__file__).parent / "out"


class DCBlock(Elaboratable):
    """One-pole DC blocker (high-pass): subtract a leaky-integrator DC estimate.

    ``dc[n]   = s[n] >> k``           (running mean estimate)
    ``y[n]    = x[n] - dc[n]``
    ``s[n+1]  = s[n] + y[n]``

    Steady state for a constant input drives ``y -> 0``; the -3 dB corner is
    roughly ``fs / (2*pi*2**k)``. Multiplier-free (one shift, two adds).

    Parameters
    ----------
    width : int
        Signed input/output width.
    k : int
        Pole parameter (larger k => lower corner, longer settling ~ 2**k samples).

    Attributes
    ----------
    clken : Signal(), in        -- advance state on this sample strobe
    x_in  : signed(width), in
    y_out : signed(width), out  -- high-passed output (delay = 1 strobe)
    """

    delay = 1

    def __init__(self, width: int, k: int = 7):
        self.w = width
        self.k = k
        self.clken = Signal()
        self.x_in = Signal(signed(width))
        self.y_out = Signal(signed(width), reset_less=True)

    @staticmethod
    def model(x, k: int = 7):
        x = np.asarray(x, dtype=np.int64)
        y = np.empty_like(x)
        s = np.int64(0)
        for n in range(len(x)):
            dc = s >> k                 # arithmetic shift (floor toward -inf)
            y[n] = x[n] - dc
            s = s + y[n]
        return y

    def elaborate(self, platform):
        m = Module()
        s = Signal(signed(self.w + self.k + 2))
        dc = Signal(signed(self.w + 2))
        m.d.comb += dc.eq(s >> self.k)
        y = Signal(signed(self.w + 1))
        m.d.comb += y.eq(self.x_in - dc)
        with m.If(self.clken):
            m.d.sync += [
                self.y_out.eq(y),
                s.eq(s + y),
            ]
        return m


class CICDecimator(Elaboratable):
    """Multiplier-free CIC decimator (cascaded integrator-comb), diff delay M=1.

    ``stages`` integrators at the input rate, decimate by ``decimation``, then
    ``stages`` comb (differentiator) stages at the output rate. No multipliers,
    no coefficient memory. DC gain is ``decimation**stages`` (full word growth
    is preserved on ``y_out``; the caller can right-shift to normalize).

    The integrator and comb cascades are evaluated combinationally within a
    single (gated) clock so the hardware matches the cumulative-sum / difference
    reference model exactly.

    Parameters
    ----------
    in_width : int
        Signed input width.
    decimation : int
        Decimation factor R (output strobes every R input strobes).
    stages : int
        Number of integrator/comb stages N (order).

    Attributes
    ----------
    clken      : Signal(), in            -- one pulse per input sample
    x_in       : signed(in_width), in
    strobe_out : Signal(), out           -- pulses when y_out is a new sample
    y_out      : signed(acc_width), out  -- full-gain decimated output
    """

    def __init__(self, in_width: int, decimation: int, stages: int = 3):
        assert decimation >= 2 and stages >= 1
        self.iw = in_width
        self.decimation = decimation
        self.stages = stages
        self.growth = math.ceil(stages * math.log2(decimation))
        self.acc_w = in_width + self.growth
        self.clken = Signal()
        self.x_in = Signal(signed(in_width))
        self.strobe_out = Signal()
        self.y_out = Signal(signed(self.acc_w), reset_less=True)

    @staticmethod
    def model(x, decimation: int, stages: int = 3):
        acc = np.asarray(x, dtype=np.int64)
        for _ in range(stages):
            acc = np.cumsum(acc)
        dec = acc[decimation - 1::decimation].copy()
        for _ in range(stages):
            delayed = np.concatenate(([np.int64(0)], dec[:-1]))
            dec = dec - delayed
        return dec

    def elaborate(self, platform):
        m = Module()

        integ = [Signal(signed(self.acc_w), name=f"integ{i}")
                 for i in range(self.stages)]
        chain = self.x_in
        new_integ = []
        for i in range(self.stages):
            nxt = Signal(signed(self.acc_w), name=f"integ{i}_n")
            m.d.comb += nxt.eq(integ[i] + chain)
            new_integ.append(nxt)
            chain = nxt
        with m.If(self.clken):
            for reg, nxt in zip(integ, new_integ):
                m.d.sync += reg.eq(nxt)

        cnt = Signal(range(self.decimation))
        dec_stb = Signal()
        m.d.comb += dec_stb.eq(self.clken & (cnt == self.decimation - 1))
        with m.If(self.clken):
            with m.If(cnt == self.decimation - 1):
                m.d.sync += cnt.eq(0)
            with m.Else():
                m.d.sync += cnt.eq(cnt + 1)

        comb_delay = [Signal(signed(self.acc_w), name=f"comb{i}_d")
                      for i in range(self.stages)]
        chain = new_integ[-1] if self.stages else self.x_in
        diffs = []
        for i in range(self.stages):
            d = Signal(signed(self.acc_w), name=f"comb{i}")
            m.d.comb += d.eq(chain - comb_delay[i])
            diffs.append((chain, d))
            chain = d
        with m.If(dec_stb):
            for (cin, _), dreg in zip(diffs, comb_delay):
                m.d.sync += dreg.eq(cin)
            m.d.sync += self.y_out.eq(chain)

        m.d.sync += self.strobe_out.eq(dec_stb)
        return m


class AMChannel(Elaboratable):
    """Single airband channel: DDC -> envelope magnitude -> DC block -> CIC.

    Exposes the DDC control/IQ-in ports plus an ``audio_out``/``audio_strobe``
    at the decimated audio rate. The envelope detector runs free (memoryless
    pipeline); the stateful DC-block and CIC advance only on validated sample
    strobes derived from ``ddc.strobe_out``.
    """

    def __init__(self, *, audio_decim: int, cic_stages: int = 3,
                 dcblock_k: int = 8):
        self.audio_decim = audio_decim
        self.ddc = DDC("clk3x", in_width=IN_WIDTH, nco_width=NCO_WIDTH)
        self.mag = EnvelopeMagnitude(self.ddc.ow[-1])   # DDC out width
        self.dcblk = DCBlock(self.mag.ow + 1, k=dcblock_k)
        self.cic = CICDecimator(self.dcblk.w, audio_decim, stages=cic_stages)
        self.audio_out = self.cic.y_out
        self.audio_strobe = self.cic.strobe_out
        # Expose the DDC's multi-rate common_edge so CommonEdgeTb can drive it.
        self.common_edge = self.ddc.common_edge

    def elaborate(self, platform):
        m = Module()
        m.submodules.ddc = ddc = self.ddc
        m.submodules.mag = mag = self.mag
        m.submodules.dcblk = dcblk = self.dcblk
        m.submodules.cic = cic = self.cic

        # Envelope detector runs every cycle on the (held) DDC outputs.
        m.d.comb += [
            mag.clken.eq(1),
            mag.re_in.eq(ddc.re_out),
            mag.im_in.eq(ddc.im_out),
        ]
        # Valid for mag = DDC strobe delayed by mag.delay cycles (ungated).
        vmag = Signal(mag.delay)
        m.d.sync += vmag.eq(Cat(ddc.strobe_out, vmag[:-1]))
        mag_valid = vmag[-1]

        # DC block advances on each validated magnitude sample.
        m.d.comb += [
            dcblk.clken.eq(mag_valid),
            dcblk.x_in.eq(mag.mag_out),
        ]
        # Its result is ready one clock after the strobe (delay = 1).
        dc_valid = Signal()
        m.d.sync += dc_valid.eq(mag_valid)

        m.d.comb += [
            cic.clken.eq(dc_valid),
            cic.x_in.eq(dcblk.y_out),
        ]
        return m


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _verify_dcblock(width=18, k=6, n=4000):
    dut = DCBlock(width, k=k)
    rng = np.random.default_rng(2)
    lim = 2 ** (width - 1) - 1
    bias = lim // 3
    x = (rng.integers(-lim // 4, lim // 4, size=n) + bias).astype(np.int64)
    expected = dut.model(x, k=k)
    got = np.zeros(n, dtype=np.int64)

    async def bench(ctx):
        ctx.set(dut.clken, 1)
        for j in range(n):
            ctx.set(dut.x_in, int(x[j]))
            await ctx.tick()
            got[j] = ctx.get(dut.y_out)

    sim = Simulator(dut)
    sim.add_clock(SYNC_PERIOD)
    sim.add_testbench(bench)
    sim.run()

    np.testing.assert_array_equal(got, expected)
    # residual DC after settling should be a tiny fraction of the input bias
    resid = np.mean(expected[n // 2:])
    return bias, float(resid)


def _verify_cic(in_width=18, decimation=6, stages=3, n=3000, gap=2):
    dut = CICDecimator(in_width, decimation, stages=stages)
    rng = np.random.default_rng(3)
    lim = 2 ** (in_width - 1) - 1
    x = rng.integers(-lim, lim, size=n).astype(np.int64)
    expected = dut.model(x, decimation, stages=stages)
    got: list[int] = []

    async def bench(ctx):
        for j in range(n):
            ctx.set(dut.x_in, int(x[j]))
            ctx.set(dut.clken, 1)
            await ctx.tick()
            ctx.set(dut.clken, 0)
            if ctx.get(dut.strobe_out):
                got.append(ctx.get(dut.y_out))
            for _ in range(gap):                 # idle cycles between samples
                await ctx.tick()
                if ctx.get(dut.strobe_out):
                    got.append(ctx.get(dut.y_out))

    sim = Simulator(dut)
    sim.add_clock(SYNC_PERIOD)
    sim.add_testbench(bench)
    sim.run()

    got = np.array(got, dtype=np.int64)
    k = min(len(got), len(expected))
    np.testing.assert_array_equal(got[:k], expected[:k])
    return dut.growth, k


def _run_chain(freq_reg, samples, *, ddc_decim, num_mult, cutoff,
               audio_decim, cic_stages, dcblock_k, gap=6):
    top = AMChannel(audio_decim=audio_decim, cic_stages=cic_stages,
                    dcblock_k=dcblock_k)
    ddc = top.ddc
    dut = CommonEdgeTb(top, [("clk3x", 3, "common_edge")])
    coeffs1, op1, odd1 = design_stage1(ddc_decim, num_mult, cutoff)
    audio: list[int] = []

    async def driver(ctx):
        ctx.set(ddc.enable_input, 0)
        ctx.set(ddc.frequency, freq_reg)
        ctx.set(ddc.decimation1, ddc_decim)
        ctx.set(ddc.decimation2, 1)
        ctx.set(ddc.decimation3, 1)
        ctx.set(ddc.bypass2, 1)
        ctx.set(ddc.bypass3, 1)
        ctx.set(ddc.operations_minus_one1, op1 - 1)
        ctx.set(ddc.operations_minus_one2, 0)
        ctx.set(ddc.operations_minus_one3, 0)
        ctx.set(ddc.odd_operations1, int(odd1))
        ctx.set(ddc.odd_operations3, 0)
        ctx.set(ddc.coeff_wren, 1)
        for addr, coeff in enumerate(coeffs1):
            ctx.set(ddc.coeff_waddr, int(addr))
            ctx.set(ddc.coeff_wdata, int(coeff))
            await ctx.tick()
        ctx.set(ddc.coeff_wren, 0)
        await ctx.tick()
        ctx.set(ddc.enable_input, 1)
        for re, im in samples:
            ctx.set(ddc.re_in, int(re))
            ctx.set(ddc.im_in, int(im))
            ctx.set(ddc.strobe_in, 1)
            await ctx.tick()
            ctx.set(ddc.strobe_in, 0)
            await ctx.tick().repeat(gap)

    async def capturer(ctx):
        budget = len(samples) * (gap + 2) + 4000
        while budget > 0:
            await ctx.tick()
            budget -= 1
            if ctx.get(top.audio_strobe):
                audio.append(ctx.get(top.audio_out))

    sim = Simulator(dut)
    sim.add_clock(SYNC_PERIOD)
    sim.add_clock(CLK3X_PERIOD, domain="clk3x", phase=SYNC_PERIOD / 2)
    sim.add_testbench(driver)
    sim.add_testbench(capturer)
    sim.run()
    return np.array(audio, dtype=np.int64)


def _demo_chain():
    """End-to-end: AM tone offset in the channel -> tune -> demod -> audio."""
    ddc_decim = 8
    audio_decim = 6              # DDC-rate / 6 -> audio rate
    num_mult = 8
    cutoff = 0.2
    cic_stages = 3
    dcblock_k = 8

    n_in = 3600
    f0 = 0.05                    # carrier offset in the wideband input (cyc/sample)
    fa = 0.0015                  # audio modulation rate at the *input* sample rate
    depth = 0.6
    amp = 1200                   # *(1+depth) < 2**(IN_WIDTH-1)-1 = 2047 (AM headroom)

    n = np.arange(n_in)
    env = amp * (1 + depth * np.cos(2 * np.pi * fa * n))
    carrier = np.exp(1j * 2 * np.pi * f0 * n)
    iq = env * carrier
    samples = np.stack([np.round(iq.real), np.round(iq.imag)], axis=1)

    freq_reg = int(round(f0 * 2 ** NCO_WIDTH)) & (2 ** NCO_WIDTH - 1)
    audio = _run_chain(freq_reg, samples, ddc_decim=ddc_decim, num_mult=num_mult,
                       cutoff=cutoff, audio_decim=audio_decim,
                       cic_stages=cic_stages, dcblock_k=dcblock_k)

    # Expected audio tone in cycles/output-sample at the audio rate.
    audio_rate_div = ddc_decim * audio_decim
    fa_out = fa * audio_rate_div

    ss = audio[len(audio) // 4:].astype(float)     # drop startup transient
    ss = ss - np.mean(ss)
    win = np.hanning(len(ss))
    spec = np.abs(np.fft.rfft(ss * win))
    fb = np.fft.rfftfreq(len(ss))
    peak = fb[np.argmax(spec)]
    return audio, peak, fa_out


def main():
    bias, resid = _verify_dcblock()
    print("[DCBlock] HW == model: PASS  "
          f"(input bias {bias} -> residual DC {resid:+.1f})")
    assert abs(resid) < bias * 0.05, "DC block did not remove the DC bias"

    growth, k = _verify_cic()
    print(f"[CICDecimator] HW == model: PASS  "
          f"({k} decimated samples, word growth +{growth} bits)")

    audio, peak, expected = _demo_chain()
    print(f"[chain] DDC->mag->DCblock->CIC produced {len(audio)} audio samples")
    print(f"[chain] recovered audio tone {peak:.4f} cyc/sample "
          f"(expected {expected:.4f})")
    assert abs(peak - expected) < 0.01, "did not recover the AM audio tone"
    print("\nPASS: AM chain tunes, demodulates, DC-blocks, and decimates to audio.")

    _save_plot(audio, peak, expected)


def _save_plot(audio, peak, expected):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(exist_ok=True)
    ss = audio[len(audio) // 4:].astype(float)
    ss = ss - np.mean(ss)
    fig, ax = plt.subplots(2, 1, figsize=(9, 7))
    ax[0].plot(ss)
    ax[0].set_title("Recovered audio (DDC -> |.| -> DC block -> CIC decimate)")
    ax[0].set_xlabel("audio sample")
    ax[0].set_ylabel("amplitude")
    ax[0].grid(True, alpha=0.3)

    spec = np.abs(np.fft.rfft(ss * np.hanning(len(ss))))
    fb = np.fft.rfftfreq(len(ss))
    ax[1].plot(fb, 20 * np.log10(spec / np.max(spec) + 1e-12))
    ax[1].axvline(expected, color="r", ls="--", alpha=0.6, label="expected tone")
    ax[1].set_title(f"Audio spectrum (peak {peak:.4f} cyc/sample)")
    ax[1].set_xlabel("normalized frequency (audio rate)")
    ax[1].set_ylabel("dB")
    ax[1].set_ylim(-80, 5)
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = OUT_DIR / "am_audio.png"
    fig.savefig(path, dpi=110)
    print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
