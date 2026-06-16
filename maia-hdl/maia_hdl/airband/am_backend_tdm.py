#!/usr/bin/env python3
"""Time-multiplexed AM back-end (handoff §7 step 5 + 7, folded over channels).

`am_audio.AMChannel` demodulates ONE channel (envelope magnitude -> DC block ->
audio CIC decimate). The full receiver has dozens of channels at a low audio rate,
so -- exactly like the channelizer lane and the cleanup FIR -- the AM back-end is
**folded**: one physical datapath iterates over the channels, with each channel's
DC-block and CIC state held in `amaranth.lib.memory.Memory` (block/distributed RAM)
indexed by the channel number. The envelope magnitude is memoryless so it is just
shared combinational logic.

Datapath per presented (chan, I, Q) sample:

    |I+jQ|  (CORDIC vectoring magnitude, multiplier-free, no angle ripple)
       -> DC block      (per-channel leaky-integrator high-pass; strips carrier)
       -> CIC decimate  (per-channel N-stage integrator/comb to the audio rate)
       -> audio sample for `chan` (emitted on that channel's decimation strobe)

The magnitude was originally an alpha-max-beta-min approximation
(``max + 3/8*min``). That estimator's gain ripples ~10% with the I/Q phase
angle, and because each received carrier sits slightly off the (un-disciplined)
Pluto LO, its baseband phasor rotates at the residual offset df -- so the ripple
amplitude-modulates every channel, injecting spurious audio tones at 4*df and
harmonics at ~-30 dBc (an audible buzz, channel-dependent because df differs per
carrier). A CORDIC vectoring magnitude is exact to the iteration count (M=12 ->
< -140 dBc, independent of datapath width), multiplier-free (adds/shifts only),
and has a constant gain K~1.6468 that a 5/8 shift-add corrects back to ~|z| so
audio levels are unchanged. See hdl/am_demod.py for the spur analysis.

It consumes the per-channel complex baseband that `ChannelizerCore` emits (one
strobe per channel at the channel rate) and produces per-channel audio strobes.
Because the upstream channel rate is far below the PL clock, a simple sequential
FSM (one channel in flight, ~4 cycles/sample) is plenty; `busy` lets the caller
hold off (a small FIFO upstream absorbs the per-CIC-boundary burst, as in the
core). Bit-exact to the per-channel `cordic_magnitude -> DCBlock -> CICDecimator`
reference models.

Run:  python hdl/am_backend_tdm.py
"""
from __future__ import annotations

import math
import pathlib

import numpy as np
from amaranth import Cat, Elaboratable, Module, Mux, Signal, signed
from amaranth.lib.memory import Memory
from amaranth.sim import Simulator

from am_audio import CICDecimator, DCBlock
from ddc_tune_decimate import SYNC_PERIOD

OUT_DIR = pathlib.Path(__file__).parent / "out"


class TdmAmBackend(Elaboratable):
    """Folded AM back-end: |.| -> DC block -> CIC decimate, shared over N channels.

    Parameters
    ----------
    n_channels : channels multiplexed onto the one datapath.
    in_width   : signed width of the complex baseband I/Q inputs.
    audio_decim: CIC decimation R from the channel rate down to the audio rate.
    cic_stages : CIC order N.
    dcblock_k  : DC-block pole parameter (corner ~ fs/(2*pi*2**k)).

    Interface (all in the `sync` domain)
    ------------------------------------
    in_valid   : in   -- present a (chan, re, im) sample
    in_chan    : in   -- channel index
    re_in,im_in: in   -- signed(in_width) complex baseband
    busy       : out  -- high while processing (ignore in_valid)
    audio_valid: out  -- pulses when audio_out is a new sample for audio_chan
    audio_chan : out  -- channel index of the emitted audio sample
    audio_out  : out  -- signed full-CIC-gain audio (caller scales)
    """

    # CORDIC vectoring iterations for the magnitude. The relative error falls
    # ~6 dB per iteration regardless of datapath width; 12 puts the residual
    # magnitude ripple (and thus the AM spur floor) below ~-140 dBc.
    CORDIC_ITERS = 12

    def __init__(self, *, n_channels: int, in_width: int, audio_decim: int,
                 cic_stages: int = 3, dcblock_k: int = 8):
        assert n_channels >= 1 and audio_decim >= 2 and cic_stages >= 1
        self.n = n_channels
        self.iw = in_width
        self.decim = audio_decim
        self.stages = cic_stages
        self.k = dcblock_k

        # CORDIC magnitude internal width: |z|*K (K~1.6468) needs ~2 guard bits
        # over the in_width; after the 5/8 scale the result is ~|z| (<= in_width+1).
        self.cw = in_width + 3
        self.mag_w = in_width + 1                       # scaled |I+jQ| width
        self.dc_w = self.mag_w + 1                      # signed audio width
        self.s_w = self.dc_w + self.k + 2               # DC accumulator width
        self.growth = math.ceil(cic_stages * math.log2(audio_decim))
        self.acc_w = self.dc_w + self.growth

        self.in_valid = Signal()
        self.in_chan = Signal(range(max(2, n_channels)))
        self.re_in = Signal(signed(in_width))
        self.im_in = Signal(signed(in_width))
        self.busy = Signal()
        self.audio_valid = Signal()
        self.audio_chan = Signal(range(max(2, n_channels)))
        self.audio_out = Signal(signed(self.acc_w), reset_less=True)

    # -- bit-exact reference model -----------------------------------------
    @classmethod
    def cordic_magnitude(cls, re, im):
        """Exact integer model of the CORDIC vectoring magnitude (matches HW).

        Computes |re + j*im| via ``CORDIC_ITERS`` vectoring rotations on the
        first-quadrant vector (|re|, |im|), then the 5/8 gain correction. All
        arithmetic is integer with arithmetic right shifts, bit-identical to the
        elaborated datapath."""
        re = np.asarray(re, dtype=np.int64)
        im = np.asarray(im, dtype=np.int64)
        x = np.abs(re).astype(np.int64).copy()
        y = np.abs(im).astype(np.int64).copy()
        for i in range(cls.CORDIC_ITERS):
            dx = x >> i
            dy = y >> i               # arithmetic shift (y may be negative)
            pos = y >= 0
            xn = np.where(pos, x + dy, x - dy)
            yn = np.where(pos, y - dx, y + dx)
            x, y = xn, yn
        return (x * 5) >> 3            # ~ (1/K) -> recover |z| scale

    def model(self, samples):
        """samples: list of (chan, re, im). Returns {chan: audio_array} using the
        exact per-channel CORDIC |.| -> DCBlock -> CICDecimator arithmetic."""
        per = {c: ([], []) for c in range(self.n)}
        for ch, re, im in samples:
            per[ch][0].append(int(re))
            per[ch][1].append(int(im))
        out = {}
        for c, (re, im) in per.items():
            if not re:
                out[c] = np.zeros(0, np.int64)
                continue
            mag = self.cordic_magnitude(re, im)
            y = DCBlock.model(mag, k=self.k)
            out[c] = CICDecimator.model(y, self.decim, stages=self.stages)
        return out

    # -- hardware ----------------------------------------------------------
    def elaborate(self, platform):
        m = Module()
        n, S = self.n, self.stages

        def make(shape):
            return Memory(shape=shape, depth=n, init=[0] * n)

        s_mem = make(signed(self.s_w))
        integ_m = [make(signed(self.acc_w)) for _ in range(S)]
        comb_m = [make(signed(self.acc_w)) for _ in range(S)]
        dec_m = make(range(self.decim))
        for idx, mem in enumerate([s_mem, dec_m, *integ_m, *comb_m]):
            m.submodules[f"mem{idx}"] = mem

        s_rp, dec_rp = s_mem.read_port(), dec_m.read_port()
        integ_rp = [mem.read_port() for mem in integ_m]
        comb_rp = [mem.read_port() for mem in comb_m]
        s_wp, dec_wp = s_mem.write_port(), dec_m.write_port()
        integ_wp = [mem.write_port() for mem in integ_m]
        comb_wp = [mem.write_port() for mem in comb_m]

        chan = Signal(range(max(2, n)))

        # all state read at the in-flight channel index (addr is constant while
        # busy, so every read port presents mem[chan] -- the pre-write state --
        # throughout the multi-cycle pipeline below)
        for rp in [s_rp, dec_rp, *integ_rp, *comb_rp]:
            m.d.comb += [rp.addr.eq(chan), rp.en.eq(1)]

        # --- envelope magnitude: CORDIC vectoring on (|re|, |im|) ---
        # Multiplier-free and (unlike alpha-max-beta-min) with no angle-dependent
        # gain ripple, so it does not modulate rotating carriers into audio spurs
        # (the buzz). One rotation per clock over CORDIC_ITERS cycles; the
        # converged cx = K*|z| (K~1.6468) is corrected to ~|z| by a 5/8 shift-add.
        cw = self.cw
        cx = Signal(signed(cw))
        cy = Signal(signed(cw))
        ci = Signal(range(self.CORDIC_ITERS))
        dx = Signal(signed(cw))
        dy = Signal(signed(cw))
        m.d.comb += [dx.eq(cx >> ci), dy.eq(cy >> ci)]   # arithmetic shifts
        cx_n = Signal(signed(cw))
        cy_n = Signal(signed(cw))
        with m.If(cy >= 0):
            m.d.comb += [cx_n.eq(cx + dy), cy_n.eq(cy - dx)]
        with m.Else():
            m.d.comb += [cx_n.eq(cx - dy), cy_n.eq(cy + dx)]
        mag = Signal(self.mag_w)
        m.d.comb += mag.eq((cx + (cx << 2)) >> 3)        # cx * 5/8 ~ |z|

        # Pipelined datapath: one arithmetic level per clock so the 62.5 MHz
        # path closes. Only the latency grows -- the AM duty cycle has ample
        # headroom (CORDIC adds ~CORDIC_ITERS cycles; duty stays well under 1).
        # Sequence per (chan, I, Q):
        #   CORDIC: CORDIC_ITERS cycles, one vectoring rotation each (reads settle)
        #   SCALE : latch magnitude = converged cx * 5/8
        #   DC    : y = mag - (s>>k), s' = s+y, advance decimation counter
        #   INTEG : S cycles, one integrator stage each (integ[st] += chain)
        #   COMB  : on the decimation strobe, S cycles, one comb stage each
        mag_r = Signal(self.mag_w)
        chain = Signal(signed(self.acc_w))     # accumulates through integ/comb
        dec_stb_reg = Signal()
        stage = Signal(range(max(2, S)))

        # stage-selected read data for the integrator / comb memories
        integ_sel = Signal(signed(self.acc_w))
        comb_sel = Signal(signed(self.acc_w))
        with m.Switch(stage):
            for st in range(S):
                with m.Case(st):
                    m.d.comb += [integ_sel.eq(integ_rp[st].data),
                                 comb_sel.eq(comb_rp[st].data)]

        ni = Signal(signed(self.acc_w))        # integrator stage result
        di = Signal(signed(self.acc_w))        # comb stage result
        m.d.comb += [ni.eq(integ_sel + chain), di.eq(chain - comb_sel)]

        with m.FSM():
            with m.State("IDLE"):
                m.d.sync += self.audio_valid.eq(0)
                with m.If(self.in_valid & ~self.busy):
                    m.d.sync += [
                        chan.eq(self.in_chan), self.busy.eq(1), ci.eq(0),
                        cx.eq(Mux(self.re_in[-1], -self.re_in, self.re_in)),
                        cy.eq(Mux(self.im_in[-1], -self.im_in, self.im_in)),
                    ]
                    m.next = "CORDIC"
            with m.State("CORDIC"):
                # one vectoring rotation per cycle; memory reads settle here too
                m.d.sync += [cx.eq(cx_n), cy.eq(cy_n)]
                with m.If(ci == self.CORDIC_ITERS - 1):
                    m.next = "SCALE"
                with m.Else():
                    m.d.sync += ci.eq(ci + 1)
            with m.State("SCALE"):
                # converged cx -> scaled magnitude (cx * 5/8 ~ |z|)
                m.d.sync += mag_r.eq(mag)
                m.next = "DC"
            with m.State("DC"):
                # y = mag - (s >> k); s' = s + y  (two adds, short path)
                dc = Signal(signed(self.dc_w))
                y = Signal(signed(self.dc_w))
                m.d.comb += [dc.eq(s_rp.data >> self.k),
                             y.eq(mag_r.as_signed() - dc),
                             s_wp.addr.eq(chan), s_wp.data.eq(s_rp.data + y),
                             s_wp.en.eq(1)]
                dec_stb = Signal()
                m.d.comb += [dec_stb.eq(dec_rp.data == self.decim - 1),
                             dec_wp.addr.eq(chan), dec_wp.en.eq(1),
                             dec_wp.data.eq(Mux(dec_stb, 0, dec_rp.data + 1))]
                m.d.sync += [chain.eq(y), stage.eq(0), dec_stb_reg.eq(dec_stb)]
                m.next = "INTEG"
            with m.State("INTEG"):
                # one integrator stage per cycle: integ[st] += chain
                with m.Switch(stage):
                    for st in range(S):
                        with m.Case(st):
                            m.d.comb += [integ_wp[st].addr.eq(chan),
                                         integ_wp[st].data.eq(ni),
                                         integ_wp[st].en.eq(1)]
                m.d.sync += chain.eq(ni)       # carries into next stage / COMB
                with m.If(stage == S - 1):
                    m.d.sync += stage.eq(0)
                    m.next = "COMB"
                with m.Else():
                    m.d.sync += stage.eq(stage + 1)
            with m.State("COMB"):
                with m.If(~dec_stb_reg):
                    m.d.sync += self.busy.eq(0)
                    m.next = "IDLE"
                with m.Else():
                    # one comb stage per cycle: out = chain - comb[st]; the new
                    # comb delay is the stage input (chain), which enters as the
                    # newest integrator output
                    with m.Switch(stage):
                        for st in range(S):
                            with m.Case(st):
                                m.d.comb += [comb_wp[st].addr.eq(chan),
                                             comb_wp[st].data.eq(chain),
                                             comb_wp[st].en.eq(1)]
                    m.d.sync += chain.eq(di)
                    with m.If(stage == S - 1):
                        m.d.sync += [self.audio_out.eq(di),
                                     self.audio_chan.eq(chan),
                                     self.audio_valid.eq(1),
                                     self.busy.eq(0), stage.eq(0)]
                        m.next = "IDLE"
                    with m.Else():
                        m.d.sync += stage.eq(stage + 1)
        return m


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _run(dut: TdmAmBackend, samples):
    out = {c: [] for c in range(dut.n)}

    async def bench(ctx):
        for ch, re, im in samples:
            ctx.set(dut.in_chan, int(ch))
            ctx.set(dut.re_in, int(re))
            ctx.set(dut.im_in, int(im))
            ctx.set(dut.in_valid, 1)
            await ctx.tick()
            ctx.set(dut.in_valid, 0)
            while True:                       # one channel in flight: wait for done
                if ctx.get(dut.audio_valid):
                    out[ctx.get(dut.audio_chan)].append(ctx.get(dut.audio_out))
                if not ctx.get(dut.busy):
                    break
                await ctx.tick()

    sim = Simulator(dut)
    sim.add_clock(SYNC_PERIOD)
    sim.add_testbench(bench)
    sim.run()
    return {c: np.array(v, np.int64) for c, v in out.items()}


def _verify_bitexact():
    dut = TdmAmBackend(n_channels=5, in_width=16, audio_decim=8, cic_stages=3,
                       dcblock_k=6)
    rng = np.random.default_rng(17)
    n = 700
    lim = 2 ** (dut.iw - 1) - 1
    # an AM-ish baseband per sample: carrier amplitude + a little wander, random phase
    chans = rng.integers(0, dut.n, size=n)
    amp = rng.integers(lim // 4, lim // 2, size=n)
    ph = rng.uniform(0, 2 * np.pi, size=n)
    re = np.round(amp * np.cos(ph)).astype(np.int64)
    im = np.round(amp * np.sin(ph)).astype(np.int64)
    samples = list(zip(chans.tolist(), re.tolist(), im.tolist()))

    hw = _run(dut, samples)
    ref = dut.model(samples)
    nout = 0
    for c in range(dut.n):
        g, e = hw[c], ref[c]
        kk = min(len(g), len(e))
        np.testing.assert_array_equal(g[:kk], e[:kk])
        nout = max(nout, kk)
    return dut.n, dut.acc_w, nout


def _demo_am():
    """End-to-end-ish: a real AM tone on one channel -> recovered audio tone."""
    dut = TdmAmBackend(n_channels=3, in_width=16, audio_decim=8, cic_stages=3,
                       dcblock_k=8)
    nper = 4096
    fa = 0.02                          # audio tone at the channel rate (cyc/sample)
    depth = 0.6
    A = 0.5 * (2 ** (dut.iw - 1) - 1)
    nn = np.arange(nper)
    env = A * (1 + depth * np.cos(2 * np.pi * fa * nn))
    phase = 0.8
    re = np.round(env * np.cos(phase)).astype(np.int64)
    im = np.round(env * np.sin(phase)).astype(np.int64)
    samples = [(0, int(re[i]), int(im[i])) for i in range(nper)]   # only ch0 active

    hw = _run(dut, samples)
    a = hw[0].astype(float)
    a = a[len(a) // 4:]
    a = a - np.mean(a)
    spec = np.abs(np.fft.rfft(a * np.hanning(len(a))))
    fb = np.fft.rfftfreq(len(a))
    peak = fb[np.argmax(spec)]
    expected = fa * dut.decim          # audio tone after decimation
    return peak, expected, len(hw[0])


def main():
    n, acc_w, nout = _verify_bitexact()
    print(f"[bit-exact] TDM AM back-end HW == per-channel (|.| -> DC block -> CIC) "
          f"model: PASS  ({n} channels share one datapath, {nout} audio samples/ch, "
          f"acc width {acc_w} b)")

    peak, expected, na = _demo_am()
    print(f"\n[AM recovery] ch0 AM tone -> audio peak {peak:.4f} cyc/sample "
          f"(expected {expected:.4f}; {na} audio samples)")
    assert abs(peak - expected) < 0.01, "did not recover the AM audio tone"
    print("\nPASS: one folded datapath AM-demodulates N channels to audio "
          "(bit-exact), and recovers a real AM tone.")


if __name__ == "__main__":
    main()
