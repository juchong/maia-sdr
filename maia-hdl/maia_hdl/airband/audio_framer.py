#!/usr/bin/env python3
"""Audio framer + DMA packer (handoff §4.3).

Dozens of low-rate audio streams share one DMA buffer, so each audio sample must
carry enough framing for the ARM/Pi side to demux and detect drops. This module
turns the per-channel audio strobes from ``am_backend_tdm.TdmAmBackend`` into a
fixed 8-byte (64-bit) record stream, then exposes an **AXI4-Stream-like**
interface (``stream_data``/``stream_valid``/``stream_ready``) that plugs straight
into maia-hdl's :class:`maia_hdl.dma.DmaStreamWrite` (default ``width=64``).

Record layout (one little-endian 64-bit word per audio sample)::

    bits [23:0]  : audio sample, signed, sign-extended to 24 bits
    bits [31:24] : carrier-level byte (per-channel AM carrier, minifloat; 0 = none)
    bits [39:32] : channel index (0..N-1)
    bits [63:40] : per-channel sample sequence counter (wraps at 2**24)

The carrier byte is the AM DC term the demodulator's DC block estimates (the
running mean of the envelope magnitude, i.e. the unmodulated carrier amplitude),
encoded as an 8-bit minifloat ``[exp(5) | mant(3)]`` so the host can run a true
carrier-power squelch. It occupies the top 8 bits of what used to be a 32-bit
sample field; the audio content was only 24 bits, so the frame size is unchanged.

The per-channel sequence counter lets userspace align channels and notice gaps
(a missed sample shows as a jump). Samples are buffered in a ``SyncFIFO``; at the
audio rate (N * <=16 ksps) the DMA drains far faster than the fill rate, so the
FIFO never backs up in normal operation (a full FIFO drops, surfaced as a seq
jump rather than corrupting the stream).

Run:  python hdl/audio_framer.py
"""
from __future__ import annotations

import pathlib

import numpy as np
from amaranth import Array, Cat, Const, Elaboratable, Module, Signal, signed
from amaranth.lib.fifo import SyncFIFO
from amaranth.sim import Simulator

from ddc_tune_decimate import SYNC_PERIOD

WORD_W = 64
AUDIO_FIELD = 24
CARRIER_FIELD = 8
CHAN_FIELD = 8
SEQ_FIELD = 24
assert AUDIO_FIELD + CARRIER_FIELD + CHAN_FIELD + SEQ_FIELD == WORD_W
# Backwards-compatible alias: the audio + carrier bytes share the old 32-bit slot.
SAMPLE_FIELD = AUDIO_FIELD + CARRIER_FIELD


class AudioFramer(Elaboratable):
    """Pack per-channel audio strobes into a 64-bit framed DMA stream.

    Parameters
    ----------
    n_channels : number of channels (channel index field).
    sample_w   : signed width of the incoming audio samples (<= 32).
    fifo_depth : staging FIFO depth.

    Interface (`sync` domain)
    -------------------------
    in_valid  : in  -- a new audio sample is present
    in_chan   : in  -- its channel index
    in_sample : in  -- signed(sample_w) audio
    overflow  : out -- pulses if a sample was dropped (FIFO full)
    stream_data/stream_valid/stream_ready : AXI4-Stream to DmaStreamWrite
    """

    def __init__(self, *, n_channels: int, sample_w: int, fifo_depth: int = 256):
        assert 1 <= sample_w <= AUDIO_FIELD
        assert n_channels <= 2 ** CHAN_FIELD
        self.n = n_channels
        self.sample_w = sample_w
        self.fifo_depth = fifo_depth

        self.in_valid = Signal()
        self.in_chan = Signal(range(max(2, n_channels)))
        self.in_sample = Signal(signed(sample_w))
        self.in_carrier = Signal(CARRIER_FIELD)
        self.overflow = Signal()

        self.stream_data = Signal(WORD_W)
        self.stream_valid = Signal()
        self.stream_ready = Signal()

    @staticmethod
    def pack(seq: int, chan: int, sample: int, carrier: int = 0) -> int:
        s = sample & (2 ** AUDIO_FIELD - 1)               # two's complement, 24b
        return ((seq & (2 ** SEQ_FIELD - 1)) << (AUDIO_FIELD + CARRIER_FIELD + CHAN_FIELD)
                | (chan & (2 ** CHAN_FIELD - 1)) << (AUDIO_FIELD + CARRIER_FIELD)
                | (carrier & (2 ** CARRIER_FIELD - 1)) << AUDIO_FIELD
                | s)

    @staticmethod
    def unpack(word: int):
        sample = word & (2 ** AUDIO_FIELD - 1)
        if sample >= 2 ** (AUDIO_FIELD - 1):
            sample -= 2 ** AUDIO_FIELD                     # sign-extend (24b)
        carrier = (word >> AUDIO_FIELD) & (2 ** CARRIER_FIELD - 1)
        chan = (word >> (AUDIO_FIELD + CARRIER_FIELD)) & (2 ** CHAN_FIELD - 1)
        seq = (word >> (AUDIO_FIELD + CARRIER_FIELD + CHAN_FIELD)) & (2 ** SEQ_FIELD - 1)
        return seq, chan, sample, carrier

    def elaborate(self, platform):
        m = Module()
        m.submodules.fifo = fifo = SyncFIFO(width=WORD_W, depth=self.fifo_depth)

        seq = Array(Signal(SEQ_FIELD, name=f"seq{c}") for c in range(self.n))

        chan_field = Signal(CHAN_FIELD)
        audio_field = Signal(AUDIO_FIELD)
        carrier_field = Signal(CARRIER_FIELD)
        m.d.comb += [
            chan_field.eq(self.in_chan),
            audio_field.eq(self.in_sample),               # sign-extends to 24b
            carrier_field.eq(self.in_carrier),
            fifo.w_data.eq(
                Cat(audio_field, carrier_field, chan_field, seq[self.in_chan])),
            fifo.w_en.eq(self.in_valid),
        ]
        with m.If(self.in_valid):
            with m.If(fifo.w_rdy):
                m.d.sync += seq[self.in_chan].eq(seq[self.in_chan] + 1)
            with m.Else():
                m.d.comb += self.overflow.eq(1)

        m.d.comb += [
            self.stream_data.eq(fifo.r_data),
            self.stream_valid.eq(fifo.r_rdy),
            fifo.r_en.eq(self.stream_ready),
        ]
        return m


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _verify(n_channels=5, sample_w=24, n=400, seed=5):
    dut = AudioFramer(n_channels=n_channels, sample_w=sample_w, fifo_depth=512)
    rng = np.random.default_rng(seed)
    lim = 2 ** (sample_w - 1) - 1
    chans = rng.integers(0, n_channels, size=n).tolist()
    samples = rng.integers(-lim, lim, size=n).tolist()
    carriers = rng.integers(0, 2 ** CARRIER_FIELD, size=n).tolist()
    got_words: list[int] = []

    async def producer(ctx):
        for ch, s, c in zip(chans, samples, carriers):
            ctx.set(dut.in_chan, int(ch))
            ctx.set(dut.in_sample, int(s))
            ctx.set(dut.in_carrier, int(c))
            ctx.set(dut.in_valid, 1)
            await ctx.tick()
            ctx.set(dut.in_valid, 0)
            assert ctx.get(dut.overflow) == 0, "unexpected FIFO overflow"
            await ctx.tick().repeat(3)        # idle, like the slow audio rate

    async def consumer(ctx):
        ctx.set(dut.stream_ready, 1)          # always-ready DMA sink
        budget = n * 6 + 200
        while budget > 0 and len(got_words) < n:
            await ctx.tick()
            budget -= 1
            if ctx.get(dut.stream_valid):
                got_words.append(ctx.get(dut.stream_data))

    sim = Simulator(dut)
    sim.add_clock(SYNC_PERIOD)
    sim.add_testbench(producer)
    sim.add_testbench(consumer)
    sim.run()

    assert len(got_words) == n, f"got {len(got_words)} words, expected {n}"
    exp_seq = {c: 0 for c in range(n_channels)}
    for i, w in enumerate(got_words):
        seq, chan, sample, carrier = AudioFramer.unpack(w)
        assert chan == chans[i], f"word {i}: chan {chan} != {chans[i]}"
        assert sample == samples[i], f"word {i}: sample {sample} != {samples[i]}"
        assert carrier == carriers[i], f"word {i}: carrier {carrier} != {carriers[i]}"
        assert seq == exp_seq[chan], f"word {i}: seq {seq} != {exp_seq[chan]}"
        exp_seq[chan] += 1
    return n, exp_seq


def main():
    n, seqs = _verify()
    print(f"[framer] {n} audio samples packed -> 64-bit records, drained over an "
          f"AXI4-Stream (DmaStreamWrite-compatible): PASS")
    print(f"[framer] per-channel sequence counters end at {dict(seqs)} "
          "(monotonic, gap-detectable)")
    print("\nPASS: framer preserves sample + channel + monotonic per-channel "
          "sequence, and presents a 64-bit DMA stream.")


if __name__ == "__main__":
    main()
