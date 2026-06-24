#!/usr/bin/env python3
"""ReceiverTop: the full multichannel airband receiver datapath (handoff §7).

Wires the separately-verified blocks into the complete receiver that sits on top
of the Maia base platform:

    wideband IQ (rxiq, sync/62.5 MHz, 12-bit)
        │  (broadcast to every lane)
        ├─► [ChannelizerCore lane 0]  (≤~4 channels: NCO mix + CIC + cleanup FIR)
        ├─► [ChannelizerCore lane 1]
        │        ...                   one lane sweeps all its channels per input
        └─► [ChannelizerCore lane L]      sample, so ~ceil(N / (Fpl/Fs)) lanes
                 │  per-lane (local_chan, re, im) bursts
                 ▼
            [round-robin collector]    serialize lanes -> one global-channel stream
                 ▼
        [TdmAmBackend]                 folded |.| -> DC block -> audio CIC (all chans)
                 ▼  (global_chan, audio)
        [AudioFramer]                  64-bit {seq, chan, sample} records
                 ▼
        stream_data/valid/ready  ─►  maia_hdl.dma.DmaStreamWrite  ─►  HP DMA -> PS

One lane time-multiplexes up to ``chans_per_lane`` channels; ``n_channels`` total
are spread across ``ceil(n_channels / chans_per_lane)`` identical lanes, all fed
the same wideband stream. Per-channel NCO tuning words are written through a flat
register interface (``freq_wren``/``freq_waddr``=global channel/``freq_wdata``),
routed to the owning lane.

Run:  python hdl/receiver_top.py
"""
from __future__ import annotations

import math

import numpy as np
from amaranth import Cat, Elaboratable, Module, Signal, signed
from amaranth.lib.fifo import SyncFIFO
from amaranth.sim import Simulator

from am_backend_tdm import TdmAmBackend
from audio_framer import AudioFramer
from channelizer_core import ChannelizerCore
from channelizer_chain import design_cic_compensation
from ddc_tune_decimate import SYNC_PERIOD


class ReceiverTop(Elaboratable):
    """Complete N-channel AM receiver: lanes -> collector -> AM back-end -> framer.

    Parameters
    ----------
    n_channels : total channels across all lanes.
    chans_per_lane : channels time-multiplexed onto one ChannelizerCore lane.
    decimation : CIC decimation inside each lane (working rate -> channel rate).
    coeffs, out_shift : cleanup FIR taps / renormalization (design_cic_compensation).
    audio_decim, cic_stages, dcblock_k : AM back-end audio decimation params.
    in_width, nco_width, stages : lane DDC params.
    audio_sample_w : signed width of the framed audio sample field (<= 32).
    """

    def __init__(self, *, n_channels, chans_per_lane, decimation, coeffs, out_shift,
                 audio_decim, cic_stages=3, dcblock_k=8, in_width=12, nco_width=24,
                 stages=3, audio_sample_w=24):
        self.n_channels = n_channels
        self.chans_per_lane = chans_per_lane
        self.n_lanes = math.ceil(n_channels / chans_per_lane)
        self.in_width = in_width
        self.nco_width = nco_width

        # Balance channels across lanes so counts differ by at most 1 (and are
        # never a pathological single channel when avoidable): e.g. 21 channels
        # over 5 lanes -> [5, 4, 4, 4, 4]. Lane budget (cycles/input) is bounded
        # by ceil(n_channels / n_lanes) <= chans_per_lane.
        base, rem = divmod(n_channels, self.n_lanes)
        self.sizes = [base + 1] * rem + [base] * (self.n_lanes - rem)
        self.bases = [sum(self.sizes[:L]) for L in range(self.n_lanes)]
        if n_channels > 1 and min(self.sizes) < 2:
            raise ValueError(
                f"lane sizing {self.sizes} has a single-channel lane; pick a "
                f"chans_per_lane that divides {n_channels} more evenly")

        self.lanes = []
        for c in self.sizes:
            self.lanes.append(ChannelizerCore(
                n_channels=c, decimation=decimation, coeffs=coeffs,
                out_shift=out_shift, in_width=in_width, nco_width=nco_width,
                stages=stages))
        self.base_w = len(self.lanes[0].out_re)         # complex baseband width

        self.am = TdmAmBackend(n_channels=n_channels, in_width=self.base_w,
                               audio_decim=audio_decim, cic_stages=cic_stages,
                               dcblock_k=dcblock_k)
        self.audio_sample_w = audio_sample_w
        self.audio_shift = max(0, self.am.acc_w - audio_sample_w)
        self.framer = AudioFramer(n_channels=n_channels, sample_w=audio_sample_w)

        # ---- ports ----
        self.freq_wren = Signal()
        self.freq_waddr = Signal(range(max(2, n_channels)))
        self.freq_wdata = Signal(nco_width)
        self.in_valid = Signal()
        self.re_in = Signal(signed(in_width))
        self.im_in = Signal(signed(in_width))
        # overflow pulses if any stage cannot keep up at the real-time cadence:
        # a lane output FIFO backs up (cleanup FIR too slow) or the framer FIFO
        # backs up (AM back-end / DMA too slow). Real-time-valid => stays 0.
        self.overflow = Signal()
        self.stream_data = self.framer.stream_data
        self.stream_valid = self.framer.stream_valid
        self.stream_ready = self.framer.stream_ready

    def elaborate(self, platform):
        m = Module()
        for L, lane in enumerate(self.lanes):
            m.submodules[f"lane{L}"] = lane
        m.submodules.am = am = self.am
        m.submodules.framer = framer = self.framer

        # ---- broadcast wideband input + route NCO register writes ----
        for L, lane in enumerate(self.lanes):
            base = self.bases[L]
            local = self.freq_waddr - base
            sel = (self.freq_waddr >= base) & (self.freq_waddr < base + lane.n_channels)
            m.d.comb += [
                lane.in_valid.eq(self.in_valid),
                lane.re_in.eq(self.re_in),
                lane.im_in.eq(self.im_in),
                lane.freq_wren.eq(self.freq_wren & sel),
                lane.freq_waddr.eq(local),
                lane.freq_wdata.eq(self.freq_wdata),
            ]

        # ---- per-lane output staging FIFOs (local_chan, re, im) ----
        clbits = [max(1, (l.n_channels - 1).bit_length()) for l in self.lanes]
        fifos = []
        lane_ovf = Signal()
        for L, lane in enumerate(self.lanes):
            f = SyncFIFO(width=clbits[L] + 2 * self.base_w, depth=4 * lane.n_channels)
            m.submodules[f"cfifo{L}"] = f
            m.d.comb += [
                f.w_data.eq(Cat(lane.out_chan, lane.out_re, lane.out_im)),
                f.w_en.eq(lane.out_valid),
            ]
            with m.If(lane.out_valid & ~f.w_rdy):
                m.d.comb += lane_ovf.eq(1)
            with m.If(lane.overflow):                  # internal lane->FIR FIFO
                m.d.comb += lane_ovf.eq(1)
            fifos.append(f)
        m.d.comb += self.overflow.eq(self.framer.overflow | lane_ovf)

        # ---- round-robin collector: serialize lanes into the AM back-end ----
        rr = Signal(range(max(2, self.n_lanes)))
        can_start = Signal()
        m.d.comb += can_start.eq(~am.busy)

        # rotating-priority select: first ready lane at/after rr (explicit wrap).
        sel_lane = Signal(range(max(2, self.n_lanes)))
        sel_valid = Signal()
        ready_rot = []
        idx_rot = []
        for i in range(self.n_lanes):
            idx = Signal(range(max(2, self.n_lanes)), name=f"idx_rot{i}")
            with m.If(rr + i >= self.n_lanes):
                m.d.comb += idx.eq(rr + i - self.n_lanes)
            with m.Else():
                m.d.comb += idx.eq(rr + i)
            idx_rot.append(idx)
            rdy = Signal(name=f"rdy_rot{i}")
            # mux: is fifo[idx] ready?
            r = 0
            for L in range(self.n_lanes):
                r = r | ((idx == L) & fifos[L].r_rdy)
            m.d.comb += rdy.eq(r)
            ready_rot.append(rdy)

        # pick the first ready in rotated order
        for i in reversed(range(self.n_lanes)):
            with m.If(ready_rot[i]):
                m.d.comb += [sel_lane.eq(idx_rot[i]), sel_valid.eq(1)]

        start = Signal()
        m.d.comb += start.eq(can_start & sel_valid)

        # mux the selected lane's fifo head into the AM back-end
        head_local = Signal(max(clbits))
        head_re = Signal(signed(self.base_w))
        head_im = Signal(signed(self.base_w))
        for L, f in enumerate(fifos):
            lc = f.r_data[:clbits[L]]
            lre = f.r_data[clbits[L]:clbits[L] + self.base_w].as_signed()
            lim = f.r_data[clbits[L] + self.base_w:].as_signed()
            with m.If(sel_lane == L):
                m.d.comb += [head_local.eq(lc), head_re.eq(lre), head_im.eq(lim),
                             f.r_en.eq(start)]

        # global channel index = lane_base + local
        global_chan = Signal(range(max(2, self.n_channels)))
        with m.Switch(sel_lane):
            for L in range(self.n_lanes):
                with m.Case(L):
                    m.d.comb += global_chan.eq(self.bases[L] + head_local)

        m.d.comb += [
            am.in_valid.eq(start), am.in_chan.eq(global_chan),
            am.re_in.eq(head_re), am.im_in.eq(head_im),
        ]
        with m.If(start):
            with m.If(rr == self.n_lanes - 1):
                m.d.sync += rr.eq(0)
            with m.Else():
                m.d.sync += rr.eq(sel_lane + 1)

        # ---- AM audio -> framer (scaled into the sample field) ----
        m.d.comb += [
            framer.in_valid.eq(am.audio_valid),
            framer.in_chan.eq(am.audio_chan),
            framer.in_sample.eq(am.audio_out >> self.audio_shift),
            framer.in_carrier.eq(am.carrier_out),
        ]
        return m

    # -- reference model (per-channel, bit-exact through lane+FIR+AM) -------
    def model(self, samples, freqs):
        """Returns {global_chan: audio_array} as the framer would carry it
        (after the >> audio_shift scale)."""
        out = {}
        for L, lane in enumerate(self.lanes):
            base = self.bases[L]
            sub_fr = freqs[base:base + lane.n_channels]
            core_out = lane.model([tuple(s) for s in samples], sub_fr)
            for c, (i_arr, q_arr) in core_out.items():
                seq = [(0, int(i_arr[k]), int(q_arr[k])) for k in range(len(i_arr))]
                am_ref = self.am.model([(0, re, im) for _, re, im in seq])[0]
                out[base + c] = np.array(
                    [int(v) >> self.audio_shift for v in am_ref], np.int64)
        return out


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _run(dut: ReceiverTop, samples, freqs, gap):
    words = []

    async def bench(ctx):
        ctx.set(dut.stream_ready, 1)
        ctx.set(dut.freq_wren, 1)
        for c, f in enumerate(freqs):
            ctx.set(dut.freq_waddr, c)
            ctx.set(dut.freq_wdata, int(f) & (2 ** dut.nco_width - 1))
            await ctx.tick()
        ctx.set(dut.freq_wren, 0)
        await ctx.tick()

        async def step():
            if ctx.get(dut.stream_valid):
                words.append(ctx.get(dut.stream_data))
            assert ctx.get(dut.overflow) == 0, "framer overflow"
            await ctx.tick()

        for (re, im) in samples:
            ctx.set(dut.re_in, int(re))
            ctx.set(dut.im_in, int(im))
            ctx.set(dut.in_valid, 1)
            await step()
            ctx.set(dut.in_valid, 0)
            for _ in range(gap - 1):
                await step()
        for _ in range(gap * dut.n_channels + 4000):
            await step()

    sim = Simulator(dut)
    sim.add_clock(SYNC_PERIOD)
    sim.add_testbench(bench)
    sim.run()
    return words


def _verify():
    coeffs, out_shift = design_cic_compensation(8, 5, 31, 0.35, 0.60)
    dut = ReceiverTop(n_channels=6, chans_per_lane=2, decimation=16, coeffs=coeffs,
                      out_shift=out_shift, audio_decim=4, cic_stages=3, dcblock_k=8,
                      in_width=12, stages=3, audio_sample_w=24)
    rng = np.random.default_rng(31)
    n = 700
    lim = 2 ** (12 - 1) - 1
    samples = rng.integers(-lim, lim, size=(n, 2)).astype(np.int64)
    freqs = [int(round(f * 2 ** dut.nco_width))
             for f in (0.05, -0.1, 0.18, -0.17, 0.09, 0.22)]

    words = _run(dut, samples, freqs, gap=40)
    assert words, "no framed audio produced"

    # demux framed records by channel; verify seq monotonic + sample bit-exact
    per = {c: [] for c in range(dut.n_channels)}
    last_seq = {}
    for w in words:
        seq, chan, sample, _carrier = AudioFramer.unpack(w)
        assert chan < dut.n_channels, f"bad channel {chan}"
        if chan in last_seq:
            assert seq == last_seq[chan] + 1, f"ch{chan} seq gap {last_seq[chan]}->{seq}"
        last_seq[chan] = seq
        per[chan].append(sample)

    ref = dut.model(samples, freqs)
    nout = 0
    for c in range(dut.n_channels):
        g = np.array(per[c], np.int64)
        e = ref[c]
        kk = min(len(g), len(e))
        assert kk > 0, f"no audio for channel {c}"
        np.testing.assert_array_equal(g[:kk], e[:kk])
        nout = max(nout, kk)
    return dut.n_channels, dut.n_lanes, len(words), nout


def main():
    nch, nlanes, nwords, nout = _verify()
    print(f"[ReceiverTop] HW == per-channel (lane -> cleanup FIR -> AM) model, "
          f"framed: PASS")
    print(f"[ReceiverTop] {nch} channels across {nlanes} lanes -> {nwords} framed "
          f"64-bit audio records ({nout} samples/ch), per-channel seq monotonic")
    print("\nPASS: end-to-end receiver channelizes, AM-demodulates, and frames "
          "all channels (bit-exact), ready for DmaStreamWrite.")


if __name__ == "__main__":
    main()
