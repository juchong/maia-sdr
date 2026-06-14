#!/usr/bin/env python3
"""Standalone DDC experiment: tune the NCO to a complex tone and decimate.

Builds on maia-hdl's DDC (NCO mixer -> 3-stage FIR decimator) used as a library.
We exercise a single decimation stage (stages 2 and 3 bypassed) to keep the
coefficient bookkeeping simple, then:

  * Run A -- tune the NCO so the input tone lands at DC. The decimated output
    should be (nearly) a constant complex value (baseband).
  * Run B -- detune the NCO so the same tone lands outside the low-pass
    passband. The decimated output should be strongly attenuated.

This demonstrates the core DDC function (frequency translation + decimation +
channel selectivity) that the airband channelizer is built from.

Run:  python hdl/ddc_tune_decimate.py
"""
from __future__ import annotations

import pathlib

import numpy as np
import scipy.signal
from amaranth import Cat, Elaboratable, Module, Signal
from amaranth.sim import Simulator

from maia_hdl.ddc import DDC

OUT_DIR = pathlib.Path(__file__).parent / "out"

# DDC fixed parameters (maia-hdl defaults)
IN_WIDTH = 12
NCO_WIDTH = 28
COEFF_WIDTH = 18
MACC_TRUNC0 = 17          # truncation of stage-1 output (maia-hdl default)
WIDTH_GROWTH0 = 4         # for in_width=12, out_width=16 (per maia-hdl test)

# Clock periods for the Amaranth simulator (1x "sync" and 3x "clk3x").
SYNC_PERIOD = 12e-9
CLK3X_PERIOD = 4e-9


class CommonEdgeTb(Elaboratable):
    """Test harness that drives a DUT's ``common_edge`` input(s).

    Vendored from maia-hdl's test/common_edge.py (MIT, (C) Daniel Estevez) so we
    do not depend on maia-hdl's (non-installed) test package.
    """

    def __init__(self, dut, domains):
        self.dut = dut
        self.domains = domains

    def elaborate(self, platform):
        m = Module()
        m.submodules.dut = self.dut
        for domain, nx, name in self.domains:
            if hasattr(self.dut, name):
                common_edge_del = Signal(nx, init=1,
                                         name=f"common_edge_del_{domain}")
                m.d[domain] += common_edge_del.eq(
                    Cat(common_edge_del[-1], common_edge_del))
                m.d.comb += getattr(self.dut, name).eq(common_edge_del[1])
        return m


def design_stage1(decimation: int, num_mult: int, cutoff: float):
    """Design + scale + pack stage-1 (FIR4DSP) coefficients.

    Returns (coeffs1, operations1, odd1) where coeffs1 is the 256-entry packed
    coefficient array expected by FIR4DSP's polyphase memory layout.

    ``num_taps = decimation * num_mult``; ``cutoff`` is normalized to Nyquist.
    """
    num_taps = decimation * num_mult
    # Linear-phase low-pass; firwin gives a symmetric, unity-DC-gain prototype.
    h = scipy.signal.firwin(num_taps, cutoff)

    # Fixed-point scaling, mirroring maia-hdl's FIRDecimator3Stage test.
    stage_growth = MACC_TRUNC0 + WIDTH_GROWTH0
    max_coeff = 2 ** (COEFF_WIDTH - 1) - 1
    scale_desired = 2 ** stage_growth / np.sum(np.abs(h))
    scale_max = max_coeff / np.max(np.abs(h))
    h_int = np.round(h * min(scale_desired, scale_max)).astype(int)

    # operations / odd flag (see maia-hdl test_fir.py): num_mult = 2*op - odd
    operations = num_mult // 2 + (num_mult % 2)
    odd = (num_mult % 2) == 1

    # Pack into the 256-entry stage-1 coefficient memory.
    coeffs = np.zeros(256, dtype=int)
    op, dec = operations, decimation
    for j in range(op):
        coeffs[j::op][:dec] = h_int[2 * j * dec:][:dec][::-1]
        if not odd or j != op - 1:
            coeffs[128 + j::op][:dec] = h_int[(2 * j + 1) * dec:][:dec][::-1]
    return coeffs, operations, odd


def run_ddc(freq_reg: int, samples: np.ndarray, *, decimation: int,
            num_mult: int, cutoff: float, gap: int = 6) -> np.ndarray:
    """Simulate the DDC over a complex input sequence; return complex output."""
    ddc = DDC("clk3x", in_width=IN_WIDTH, nco_width=NCO_WIDTH)
    dut = CommonEdgeTb(ddc, [("clk3x", 3, "common_edge")])
    coeffs1, op1, odd1 = design_stage1(decimation, num_mult, cutoff)

    n_in = len(samples)
    want = n_in // decimation - 12
    out: list[complex] = []

    async def driver(ctx):
        ctx.set(ddc.enable_input, 0)
        ctx.set(ddc.frequency, freq_reg)
        ctx.set(ddc.decimation1, decimation)
        ctx.set(ddc.decimation2, 1)
        ctx.set(ddc.decimation3, 1)
        ctx.set(ddc.bypass2, 1)
        ctx.set(ddc.bypass3, 1)
        ctx.set(ddc.operations_minus_one1, op1 - 1)
        ctx.set(ddc.operations_minus_one2, 0)
        ctx.set(ddc.operations_minus_one3, 0)
        ctx.set(ddc.odd_operations1, int(odd1))
        ctx.set(ddc.odd_operations3, 0)
        # load stage-1 coefficients (addr space 0..255)
        ctx.set(ddc.coeff_wren, 1)
        for addr, coeff in enumerate(coeffs1):
            ctx.set(ddc.coeff_waddr, int(addr))
            ctx.set(ddc.coeff_wdata, int(coeff))
            await ctx.tick()
        ctx.set(ddc.coeff_wren, 0)
        await ctx.tick()
        # feed samples, one strobe pulse per sample with idle gap between
        ctx.set(ddc.enable_input, 1)
        for re, im in samples:
            ctx.set(ddc.re_in, int(re))
            ctx.set(ddc.im_in, int(im))
            ctx.set(ddc.strobe_in, 1)
            await ctx.tick()
            ctx.set(ddc.strobe_in, 0)
            if gap:
                await ctx.tick().repeat(gap)

    async def capturer(ctx):
        budget = n_in * (gap + 2) + 2000
        while len(out) < want and budget > 0:
            await ctx.tick()
            budget -= 1
            if ctx.get(ddc.strobe_out):
                out.append(ctx.get(ddc.re_out) + 1j * ctx.get(ddc.im_out))

    sim = Simulator(dut)
    sim.add_clock(SYNC_PERIOD)
    sim.add_clock(CLK3X_PERIOD, domain="clk3x", phase=SYNC_PERIOD / 2)
    sim.add_testbench(driver)
    sim.add_testbench(capturer)
    sim.run()
    return np.array(out)


def main():
    decimation = 8
    num_mult = 8           # 64 taps
    cutoff = 0.1           # normalized to Nyquist
    amplitude = 2000       # < 2**(IN_WIDTH-1)-1 = 2047
    n_in = 2500
    f0 = 0.05              # tone frequency, cycles/sample

    n = np.arange(n_in)
    tone = amplitude * np.exp(1j * 2 * np.pi * f0 * n)
    samples = np.stack([np.round(tone.real), np.round(tone.imag)], axis=1)

    def freq_reg(f):
        return int(round(f * 2 ** NCO_WIDTH)) & (2 ** NCO_WIDTH - 1)

    # Run A: tune NCO to the tone -> lands at DC.
    out_a = run_ddc(freq_reg(f0), samples,
                    decimation=decimation, num_mult=num_mult, cutoff=cutoff)
    # Run B: detune by 0.2 cyc/sample -> tone lands in the stopband.
    out_b = run_ddc(freq_reg(f0 - 0.2), samples,
                    decimation=decimation, num_mult=num_mult, cutoff=cutoff)

    ss_a = out_a[50:]      # drop filter/pipeline transient
    ss_b = out_b[50:]
    dc = np.mean(ss_a)
    ripple = np.std(ss_a) / np.abs(dc)
    rms_a = np.sqrt(np.mean(np.abs(ss_a) ** 2))
    rms_b = np.sqrt(np.mean(np.abs(ss_b) ** 2))
    rejection_db = 20 * np.log10(rms_a / rms_b)

    print(f"Outputs captured: A={len(out_a)} B={len(out_b)} "
          f"(decimation {decimation}, {n_in} input samples)")
    print(f"[Run A: tuned to tone -> DC]")
    print(f"  output DC value : {dc.real:.0f} + {dc.imag:.0f}j "
          f"(|DC| = {np.abs(dc):.0f})")
    print(f"  AC ripple/|DC|  : {ripple*100:.2f} %  (small => clean baseband)")
    print(f"  RMS             : {rms_a:.0f}")
    print(f"[Run B: tone detuned into stopband]")
    print(f"  RMS             : {rms_b:.0f}")
    print(f"[Channel selectivity] rejection A/B = {rejection_db:.1f} dB")

    assert ripple < 0.05, f"tuned output not DC-like (ripple {ripple:.3f})"
    assert rejection_db > 30, f"insufficient rejection ({rejection_db:.1f} dB)"
    print("\nPASS: NCO tuning brings the tone to baseband; off-tune is rejected.")

    _save_plot(out_a, out_b, decimation, f0)


def _save_plot(out_a, out_b, decimation, f0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(2, 1, figsize=(9, 7))
    ax[0].plot(out_a.real, label="I (tuned)")
    ax[0].plot(out_a.imag, label="Q (tuned)")
    ax[0].plot(np.abs(out_b), label="|out| (detuned)", alpha=0.7)
    ax[0].set_title(f"DDC output (decim {decimation}, tone f0={f0} cyc/sample)")
    ax[0].set_xlabel("output sample")
    ax[0].set_ylabel("amplitude")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    ss = out_a[50:]
    spec = np.fft.fftshift(np.abs(np.fft.fft(ss * np.hanning(len(ss)))))
    fb = np.fft.fftshift(np.fft.fftfreq(len(ss)))
    ax[1].plot(fb, 20 * np.log10(spec / np.max(spec) + 1e-12))
    ax[1].set_title("Run A output spectrum (energy at DC = tuned to baseband)")
    ax[1].set_xlabel("normalized frequency (output rate)")
    ax[1].set_ylabel("dB")
    ax[1].set_ylim(-80, 5)
    ax[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = OUT_DIR / "ddc_tune_decimate.png"
    fig.savefig(path, dpi=110)
    print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
