#!/usr/bin/env python3
"""AM envelope detector for the airband channelizer (handoff doc §7 step 5).

Airband voice is AM (A3E), so demodulation is the magnitude of the complex
baseband: audio(t) = |I(t) + jQ(t)|, followed by a DC block (remove the carrier
amplitude) and audio-rate decimation. This module implements the magnitude with
a cheap, multiplier-free **alpha-max-beta-min** approximation:

    |z| ~= max(|I|,|Q|) + (3/8) * min(|I|,|Q|)

which costs only abs/compare/add/shift (no DSP48), has bounded error (~ +6.8%
worst case at 45 degrees), and is plenty accurate for AM voice. This matters
because the channelizer must fit many channels in the small Z-7010 (§4.2).

Pipeline latency: 2 cycles (gated by ``clken``).

Run:  python hdl/am_demod.py   (verifies HW == integer model, error bound, and
                                demonstrates AM-tone recovery; writes a plot)
"""
from __future__ import annotations

import pathlib

import numpy as np
from amaranth import Module, Mux, Signal, signed, Elaboratable
from amaranth.sim import Simulator

OUT_DIR = pathlib.Path(__file__).parent / "out"


class EnvelopeMagnitude(Elaboratable):
    """Approximate complex magnitude via alpha-max-beta-min (alpha=1, beta=3/8).

    Parameters
    ----------
    in_width : int
        Width of the signed I/Q inputs.

    Attributes
    ----------
    clken : Signal(), in       -- clock enable
    re_in, im_in : signed(in_width), in
    mag_out : unsigned(in_width+1), out   -- approximate magnitude (delay=2)
    """

    delay = 2

    def __init__(self, in_width: int):
        self.iw = in_width
        self.ow = in_width + 1
        self.clken = Signal()
        self.re_in = Signal(signed(in_width))
        self.im_in = Signal(signed(in_width))
        self.mag_out = Signal(self.ow, reset_less=True)

    @staticmethod
    def model(re, im):
        re = np.asarray(re, dtype=np.int64)
        im = np.asarray(im, dtype=np.int64)
        a, b = np.abs(re), np.abs(im)
        mx, mn = np.maximum(a, b), np.minimum(a, b)
        return mx + ((mn * 3) >> 3)

    def elaborate(self, platform):
        m = Module()
        re_abs = Signal(self.iw)
        im_abs = Signal(self.iw)
        with m.If(self.clken):
            m.d.sync += [
                re_abs.eq(Mux(self.re_in[-1], -self.re_in, self.re_in)),
                im_abs.eq(Mux(self.im_in[-1], -self.im_in, self.im_in)),
            ]
        bigger = re_abs > im_abs
        mx = Mux(bigger, re_abs, im_abs)
        mn = Mux(bigger, im_abs, re_abs)
        mag = mx + ((mn + (mn << 1)) >> 3)   # mx + (3*mn)>>3
        with m.If(self.clken):
            m.d.sync += self.mag_out.eq(mag)
        return m


def _verify_hw(in_width=16, n=2000):
    """Feed random IQ; check HW output matches the integer model exactly."""
    dut = EnvelopeMagnitude(in_width)
    lim = 2 ** (in_width - 1) - 1
    rng = np.random.default_rng(1)
    re = rng.integers(-lim, lim, size=n)
    im = rng.integers(-lim, lim, size=n)
    expected = dut.model(re, im)
    got = np.zeros(n, dtype=np.int64)

    async def bench(ctx):
        ctx.set(dut.clken, 1)
        for j in range(n):
            ctx.set(dut.re_in, int(re[j]))
            ctx.set(dut.im_in, int(im[j]))
            await ctx.tick()
            # mag_out read just after edge j reflects model(sample[j-1]):
            # inputs are set before the edge and the read happens after it,
            # which already absorbs one of the two pipeline cycles.
            got[j] = ctx.get(dut.mag_out)

    sim = Simulator(dut)
    sim.add_clock(12e-9)
    sim.add_testbench(bench)
    sim.run()

    np.testing.assert_array_equal(got[1:], expected[:n - 1])

    # approximation error vs true magnitude (ignore tiny vectors)
    true = np.hypot(re, im)
    mask = true > lim * 0.05
    err = (expected[mask] - true[mask]) / true[mask]
    return float(err.min()), float(err.max())


def _demo_am_recovery(in_width=16):
    """Numpy demo: envelope detector recovers an AM tone from complex baseband."""
    fs = 16000          # audio-band complex baseband rate (already channelized)
    t = np.arange(4000) / fs
    fa = 1000.0         # 1 kHz "voice" tone
    depth = 0.6
    carrier_amp = 0.7 * (2 ** (in_width - 1) - 1)
    # AM baseband: carrier at DC, amplitude = A*(1+m*cos), arbitrary fixed phase
    env = carrier_amp * (1 + depth * np.cos(2 * np.pi * fa * t))
    phase = 0.9
    re = np.round(env * np.cos(phase)).astype(int)
    im = np.round(env * np.sin(phase)).astype(int)

    mag = EnvelopeMagnitude.model(re, im).astype(float)
    audio = mag - np.mean(mag)          # DC block (remove carrier amplitude)

    # recovered tone amplitude vs expected (depth*carrier_amp)
    spec = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    fbins = np.fft.rfftfreq(len(audio), 1 / fs)
    peak_hz = fbins[np.argmax(spec)]
    return t, env, mag, audio, fs, fa, peak_hz


def main():
    emin, emax = _verify_hw()
    print(f"[HW vs model] exact match over all samples: PASS")
    print(f"[approx error vs true |z|] min={emin*100:+.2f}%  max={emax*100:+.2f}%")
    assert -0.03 < emin and emax < 0.07, "approximation error out of expected band"

    t, env, mag, audio, fs, fa, peak_hz = _demo_am_recovery()
    print(f"[AM recovery] injected tone {fa:.0f} Hz; recovered peak {peak_hz:.0f} Hz")
    assert abs(peak_hz - fa) < fs / len(audio) * 3, "did not recover AM tone"
    print("\nPASS: envelope magnitude matches model, error bounded, AM tone recovered.")

    _save_plot(t, env, mag, audio, fs)


def _save_plot(t, env, mag, audio, fs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(2, 1, figsize=(9, 7))
    ms = slice(0, 400)
    ax[0].plot(t[ms] * 1e3, env[ms], label="true envelope A(1+m)")
    ax[0].plot(t[ms] * 1e3, mag[ms], "--", label="alpha-max-beta-min |z|")
    ax[0].set_title("AM envelope detection (complex baseband -> magnitude)")
    ax[0].set_xlabel("time (ms)")
    ax[0].set_ylabel("amplitude")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    spec = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    fbins = np.fft.rfftfreq(len(audio), 1 / fs)
    ax[1].plot(fbins, 20 * np.log10(spec / np.max(spec) + 1e-12))
    ax[1].set_title("Recovered audio spectrum after DC block (peak = AM tone)")
    ax[1].set_xlabel("Hz")
    ax[1].set_ylabel("dB")
    ax[1].set_xlim(0, 4000)
    ax[1].set_ylim(-80, 5)
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = OUT_DIR / "am_demod.png"
    fig.savefig(path, dpi=110)
    print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
