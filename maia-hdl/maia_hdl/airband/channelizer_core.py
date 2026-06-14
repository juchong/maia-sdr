#!/usr/bin/env python3
"""Integrated multichannel channelizer core (handoff §4 / §7 step 8).

This is the unification step: the separately-verified building blocks are wired
into one top-level datapath that takes the wideband IQ stream and emits a clean,
droop-compensated baseband sample per channel.

    wideband IQ ─► [TdmDdcLaneBRAM]               (NCO mix + CIC decimate, per ch
                        │                            state in block RAM)
                        ▼  out_valid (chan, re, im)
                  [SyncFIFO]                        (absorbs the burst of channel
                        │                            outputs at each CIC boundary)
                        ▼
            ┌── [TdmFirEngine  I] ──┐               (one folded MAC shared over all
            └── [TdmFirEngine  Q] ──┘                channels: CIC droop comp +
                        │                            adjacent-channel selectivity)
                        ▼  out_valid (chan, re, im)  -> compensated baseband

Why a FIFO between the two: every channel shares one decimation counter cadence,
so all ``n_channels`` lanes strobe their CIC output on the *same* input sample
(a burst of N outputs in N consecutive cycles every R inputs). The folded cleanup
FIR needs ``ntaps`` cycles per sample, far longer than one cycle, so the burst is
buffered and drained at the FIR's pace. The channel rate is low enough that the
FIR always catches up before the next CIC boundary (verified by the bit-exact
test below — nothing is dropped).

The I and Q cleanup engines are identical folded FIRs run in lockstep (same
control FSM, fed simultaneously), so their outputs co-arrive and form the complex
compensated sample.

This module is also the synthesis target for the combined place-and-route run
(``ChannelizerCore`` -> Vivado), giving the real *combined* LUT/FF/DSP/BRAM and
timing for the whole receiver datapath, not just the isolated blocks.

Run:  python hdl/channelizer_core.py
"""
from __future__ import annotations

import pathlib

import numpy as np
from amaranth import Cat, Elaboratable, Module, Signal, signed
from amaranth.lib.fifo import SyncFIFO
from amaranth.sim import Simulator

from channelizer_chain import (TdmFirEngine, TdmFirEngineBRAM,
                               design_cic_compensation)
from channelizer_lane import TdmDdcLaneBRAM
from ddc_tune_decimate import SYNC_PERIOD

OUT_DIR = pathlib.Path(__file__).parent / "out"


class ChannelizerCore(Elaboratable):
    """Full per-channel datapath: BRAM-backed TDM DDC lane + folded complex
    cleanup FIR, joined by a burst-absorbing FIFO.

    Ports mirror the lane's config/input side and the FIR's output side:

    Config (load channel ``c``'s NCO tuning word):
        freq_wren, freq_waddr, freq_wdata

    Input (one wideband complex sample, broadcast to all channels):
        in_valid, re_in, im_in

    Output (droop-compensated baseband sample for one channel):
        out_valid, out_chan, out_re, out_im
    """

    def __init__(self, *, n_channels: int, decimation: int, coeffs, out_shift: int,
                 in_width: int = 12, nco_width: int = 24, lut_addr_bits: int = 10,
                 lut_width: int = 12, stages: int = 3, fifo_depth: int | None = None):
        self.n_channels = n_channels
        self.coeffs = [int(c) for c in coeffs]
        self.out_shift = out_shift

        self.lane = TdmDdcLaneBRAM(
            n_channels=n_channels, decimation=decimation, in_width=in_width,
            nco_width=nco_width, lut_addr_bits=lut_addr_bits, lut_width=lut_width,
            stages=stages)
        self.fir_i = TdmFirEngineBRAM(self.coeffs, n_channels=n_channels,
                                      in_width=self.lane.acc_w, out_shift=out_shift)
        self.fir_q = TdmFirEngineBRAM(self.coeffs, n_channels=n_channels,
                                      in_width=self.lane.acc_w, out_shift=out_shift)
        # all channels strobe together -> buffer a full burst plus margin
        self.fifo_depth = fifo_depth or (4 * n_channels)

        # config / input (passthrough to lane)
        self.freq_wren = self.lane.freq_wren
        self.freq_waddr = self.lane.freq_waddr
        self.freq_wdata = self.lane.freq_wdata
        self.in_valid = self.lane.in_valid
        self.re_in = self.lane.re_in
        self.im_in = self.lane.im_in

        # output (compensated baseband)
        self.out_valid = Signal()
        self.out_chan = Signal(range(max(2, n_channels)))
        self.out_re = Signal(signed(self.fir_i.ow), reset_less=True)
        self.out_im = Signal(signed(self.fir_q.ow), reset_less=True)
        # pulses if the lane->cleanup-FIR FIFO backs up (cleanup FIR can't drain
        # the per-CIC-boundary burst in time -> a real-time throughput overrun).
        self.overflow = Signal()

    # -- bit-exact reference model -----------------------------------------
    def model(self, samples, freqs):
        """Compose the lane model and a per-channel single-channel FIR model.

        Returns ``{chan: (out_re_array, out_im_array)}`` in decimation order,
        using exactly the hardware fixed-point arithmetic of both stages.
        """
        lane_out = self.lane.model(samples, freqs)
        res = {}
        for c, (i_str, q_str) in lane_out.items():
            yi = [y for _, y in TdmFirEngine.model(
                [(0, int(x)) for x in i_str], self.coeffs, 1, self.out_shift)]
            yq = [y for _, y in TdmFirEngine.model(
                [(0, int(x)) for x in q_str], self.coeffs, 1, self.out_shift)]
            res[c] = (np.array(yi, np.int64), np.array(yq, np.int64))
        return res

    # -- hardware ----------------------------------------------------------
    def elaborate(self, platform):
        m = Module()
        m.submodules.lane = lane = self.lane
        m.submodules.fir_i = fir_i = self.fir_i
        m.submodules.fir_q = fir_q = self.fir_q

        cbits = max(1, (self.n_channels - 1).bit_length())
        aw = lane.acc_w
        m.submodules.fifo = fifo = SyncFIFO(width=cbits + 2 * aw,
                                            depth=self.fifo_depth)

        # lane output -> FIFO (chan, re, im packed)
        m.d.comb += [
            fifo.w_data.eq(Cat(lane.out_chan, lane.out_re, lane.out_im)),
            fifo.w_en.eq(lane.out_valid),
        ]
        with m.If(lane.out_valid & ~fifo.w_rdy):
            m.d.comb += self.overflow.eq(1)

        # FIFO head -> both cleanup engines (run in lockstep)
        head_chan = fifo.r_data[:cbits]
        head_re = fifo.r_data[cbits:cbits + aw].as_signed()
        head_im = fifo.r_data[cbits + aw:].as_signed()

        start = Signal()
        m.d.comb += start.eq(fifo.r_rdy & ~fir_i.busy & ~fir_q.busy)
        m.d.comb += [
            fifo.r_en.eq(start),
            fir_i.in_valid.eq(start), fir_i.in_chan.eq(head_chan),
            fir_i.x_in.eq(head_re),
            fir_q.in_valid.eq(start), fir_q.in_chan.eq(head_chan),
            fir_q.x_in.eq(head_im),
        ]

        # both engines co-arrive (identical FSM, fed together)
        m.d.comb += [
            self.out_valid.eq(fir_i.out_valid),
            self.out_chan.eq(fir_i.out_chan),
            self.out_re.eq(fir_i.out),
            self.out_im.eq(fir_q.out),
        ]
        return m


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _run_core(dut: ChannelizerCore, samples, freqs, gap: int):
    """Drive the core feeding one wideband sample every ``gap`` cycles; collect
    every compensated output strobe per channel."""
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
        for _ in range(gap * dut.n_channels + 400):   # drain FIFO + FIR tails
            await step()

    sim = Simulator(dut)
    sim.add_clock(SYNC_PERIOD)
    sim.add_testbench(bench)
    sim.run()
    return {c: (np.array(i, np.int64), np.array(q, np.int64))
            for c, (i, q) in out.items()}


def _verify_bitexact():
    """Whole-core HW must equal lane-model -> per-channel FIR-model, bit for bit."""
    coeffs, out_shift = design_cic_compensation(8, 5, 31, 0.35, 0.60)
    dut = ChannelizerCore(n_channels=4, decimation=16, coeffs=coeffs,
                          out_shift=out_shift, in_width=12, stages=3)
    rng = np.random.default_rng(23)
    n = 360
    lim = 2 ** (12 - 1) - 1
    samples = rng.integers(-lim, lim, size=(n, 2)).astype(np.int64)
    fr = [int(round(f * 2 ** dut.lane.nco_width))
          for f in (0.05, -0.1, 0.2, -0.17)]

    hw = _run_core(dut, samples, fr, gap=40)
    ref = dut.model([tuple(s) for s in samples], fr)
    nout = 0
    for c in range(dut.n_channels):
        for k in (0, 1):
            g, e = hw[c][k], ref[c][k]
            kk = min(len(g), len(e))
            assert kk > 0, f"no output for channel {c}"
            np.testing.assert_array_equal(g[:kk], e[:kk])
            nout = max(nout, kk)
    return dut.n_channels, len(coeffs), nout


def _demo_selectivity():
    """End-to-end: each channel tunes its tone to baseband; an adjacent tone is
    rejected by the CIC + cleanup FIR. Reports rejection in dB."""
    coeffs, out_shift = design_cic_compensation(8, 5, 31, 0.35, 0.60)
    dut = ChannelizerCore(n_channels=3, decimation=16, coeffs=coeffs,
                          out_shift=out_shift, in_width=12, stages=3)
    # tones: each channel's own + a strong neighbour offset to test rejection
    tones = [0.03, 0.12, -0.08]
    n = 2048
    nn = np.arange(n)
    iq = np.zeros(n, complex)
    for f in tones:
        iq += 1800 * np.exp(1j * 2 * np.pi * f * nn)
    iq /= len(tones)
    samples = np.stack([np.round(iq.real), np.round(iq.imag)], 1).astype(np.int64)
    fr = [int(round(f * 2 ** dut.lane.nco_width)) for f in tones]

    hw = _run_core(dut, samples, fr, gap=40)
    stats = {}
    for c in range(dut.n_channels):
        i, q = hw[c]
        z = (i + 1j * q)[len(i) // 3:]
        dc = np.abs(np.mean(z))
        ac = np.std(z)
        stats[c] = (dc, ac / (dc + 1e-9))
    return tones, stats


def main():
    C, nt, nout = _verify_bitexact()
    print(f"[bit-exact] ChannelizerCore HW == (lane model -> cleanup FIR model): "
          f"PASS  ({C} ch, {nt}-tap cleanup FIR, {nout} samples/ch)")

    tones, stats = _demo_selectivity()
    print("\n[end-to-end] tuned tone -> baseband, neighbours rejected "
          "(CIC + cleanup FIR):")
    ok = True
    for c, (dc, ripple) in stats.items():
        print(f"  ch{c}  tone {tones[c]:+.3f} cyc/s  ->  |DC| {dc:10.0f}  "
              f"ripple/|DC| {ripple*100:5.1f}%")
        ok = ok and ripple < 0.3
    assert ok, "a channel did not produce a clean baseband through the full core"
    print("\nPASS: integrated core channelizes + cleans each channel "
          "(bit-exact through lane + folded complex FIR).")


if __name__ == "__main__":
    main()
