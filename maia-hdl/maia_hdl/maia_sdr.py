#
# Copyright (C) 2022-2025 Daniel Estevez <daniel@destevez.net>
#
# This file is part of maia-sdr
#
# SPDX-License-Identifier: MIT
#

import argparse
import os
import sys

from amaranth import *
from amaranth.lib.cdc import FFSynchronizer, PulseSynchronizer
import amaranth.back.verilog

from .axi4_lite import Axi4LiteRegisterBridge
from .cdc import RegisterCDC, RxIQCDC
from .clknx import ClkNxCommonEdge
from .config import MaiaSDRConfig
from . import configs
from .ddc import DDC
from .dma import DmaStreamWrite
from .pulse import PulseStretcher
from .pluto_platform import PlutoPlatform
from .register import Access, Field, Registers, Register, RegisterMap
from .recorder import Recorder16IQ, RecorderMode
from .spectrometer import Spectrometer

# The airband multichannel receiver DSP is vendored under maia_hdl/airband/.
# It is imported by flat module name (the directory is placed on sys.path) so
# the verified hdl/ sources can be dropped in unchanged.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'airband'))
from receiver_top import ReceiverTop  # noqa: E402

# IP core version
_version = '0.6.2'

# Airband receiver deployment configuration (see hdl/realtime_budget.py:
# Fs=16 MHz, chans_per_lane=3, lane_decim=160, 63-tap cleanup FIR -> 6 lanes;
# duties lane=0.77 / fir=0.33 / am=0.77 at the fixed 62.5 MHz sync clock). The
# cleanup-FIR coefficients are precomputed from
# design_cic_compensation(160, 3, 63, 0.10, 0.16) and embedded so the bitstream
# build needs no scipy.
#
# 18 channels over 6 lanes of 3: at Fs=16 MHz a lane can carry at most 3 channels
# (62.5/16 ~= 3.9 PL cycles/input sample), and 7 lanes (21 ch) overflow the
# XC7Z010 LUTs (18234 > 17600 in Vivado), so the plan is capped at 18 ch / 6 lanes
# (the same lane count as the proven 14 MHz build).
#
# A 16 MHz capture (re-centered ~126.4 MHz) widens the window to ~+/-8 MHz so the
# 133.65 MHz channel fits alongside the existing plan. The channel rate is
# Fs / lane_decim = 16e6 / 160 = 100000 Hz.
#
# The cleanup FIR doubles as the channel-select filter, narrowed to the AM voice
# bandwidth (~+/-5 kHz at the 100000 Hz channel rate: flat through ~4 kHz, -0.9 dB
# @ 4 kHz, -5.8 dB @ 6 kHz, -19 dB @ 8 kHz, ~-109 dB at the 25 kHz adjacent
# channel). This trims the wideband IQ before envelope detection so out-of-voice
# noise is not folded down into the audio (the +-8 kHz original passed a broadband
# HF shelf that the demod reproduced as harshness). Tap count is unchanged (63), so
# the folded FIR's duty/BRAM/DSP cost - and the proven place-and-route/timing - are
# identical regardless of the passband width; only out_shift changes (the narrower
# taps sum higher, so renormalize by 2**18 instead of 2**17).
#
# audio_decim=5 -> 20000 sps audio (100000/5), Nyquist 10 kHz, so the order-4
# audio CIC passes the widened voice with little droop (-1.6 dB @3.4 kHz, -5.1 dB
# @6 kHz; a host de-droop biquad flattens the residual). Lowering audio_decim is
# throughput-free (the AM back-end runs at the channel rate regardless). The host
# audio rate MUST match (20000 sps).
_AIRBAND_N_CHANNELS = 18
_AIRBAND_CHANS_PER_LANE = 3
_AIRBAND_LANE_DECIM = 160
_AIRBAND_AUDIO_DECIM = 5
_AIRBAND_CIC_STAGES = 4
_AIRBAND_DCBLOCK_K = 10
_AIRBAND_NCO_WIDTH = 24
_AIRBAND_STAGES = 3
_AIRBAND_SAMPLE_W = 24
_AIRBAND_FIR_OUT_SHIFT = 18
_AIRBAND_FIR_COEFFS = [
    -1, -4, -10, -18, -24, -23, -5, 39, 117, 228, 356, 470, 519, 442, 180, -306,
    -1013, -1877, -2760, -3452, -3694, -3211, -1763, 799, 4487, 9145, 14445, 19915,
    24998, 29131, 31829, 32767, 31829, 29131, 24998, 19915, 14445, 9145, 4487, 799,
    -1763, -3211, -3694, -3452, -2760, -1877, -1013, -306, 180, 442, 519, 470, 356,
    228, 117, 39, -5, -23, -24, -18, -10, -4, -1]


class MaiaSDR(Elaboratable):
    """Maia SDR top level

    This elaboratable is the top-level Maia SDR IP core.
    """
    def __init__(self, config=MaiaSDRConfig()):
        config.validate()
        self.config = config
        # 5-bit register address space: control (0x0), recorder (0x10),
        # sdr (0x20), airband (0x40).
        self.axi4_awidth = 5
        self.s_axi_lite = ClockDomain()
        self.sampling = ClockDomain()
        # A clock domain called 'sync' is added to override the default
        # behaviour, since we drive the reset internally.
        #
        # See https://github.com/amaranth-lang/amaranth/issues/1506
        self.sync = ClockDomain()
        self.clk2x = ClockDomain()
        self.clk3x = ClockDomain()

        self.axi4lite = Axi4LiteRegisterBridge(
            self.axi4_awidth, name='s_axi_lite')
        self.control_registers = Registers(
            'control',
            {
                0b00: Register(
                    'product_id', [
                        Field('product_id', Access.R, 32, 0x6169616d)
                    ]),
                0b01: Register('version', [
                    Field('bugfix', Access.R, 8,
                          int(_version.split('.')[2])),
                    Field('minor', Access.R, 8,
                          int(_version.split('.')[1])),
                    Field('major', Access.R, 8,
                          int(_version.split('.')[0])),
                    Field('platform', Access.R, 8, config.platform),
                ]),
                0b10: Register('control', [
                    Field('sdr_reset', Access.RW, 1, 1),
                ]),
                0b11: Register('interrupts', [
                    Field('spectrometer', Access.Rsticky, 1, 0),
                    Field('recorder', Access.Rsticky, 1, 0),
                ], interrupt=True),
            },
            2)
        self.recorder_registers = Registers(
            'recorder',
            {
                0b0: Register('recorder_control', [
                    Field('start', Access.Wpulse, 1, 0),
                    Field('stop', Access.Wpulse, 1, 0),
                    Field('mode', Access.RW,
                          Shape.cast(RecorderMode).width, 0),
                    Field('dropped_samples', Access.R, 1, 0),
                ]),
                0b1: Register('recorder_next_address', [
                    Field('next_address', Access.R, 32, 0),
                ]),
            },
            1)
        self.spectrometer = Spectrometer(
            config.spectrometer_address,
            config.spectrometer_buffers.bit_length() - 1,
            dma_name='m_axi_spectrometer')
        self.recorder = Recorder16IQ(
            config.recorder_address_range[0],
            config.recorder_address_range[1],
            dma_name='m_axi_recorder', domain_in='sync',
            domain_dma='s_axi_lite')
        self.ddc = DDC('clk3x')
        self.sdr_registers = Registers(
            'sdr', {
                0b000: Register(
                    'spectrometer',
                    [
                        Field('use_ddc_out',
                              Access.RW,
                              1,
                              0),
                        Field('num_integrations',
                              Access.RW,
                              self.spectrometer.nint_width,
                              -1),
                        Field('abort', Access.Wpulse, 1, 0),
                        Field('last_buffer',
                              Access.R,
                              len(self.spectrometer.last_buffer),
                              0),
                        Field('peak_detect',
                              Access.RW,
                              1,
                              0),
                    ]),
                0b001: Register(
                    'ddc_coeff_addr',
                    [
                        Field('coeff_waddr',
                              Access.RW,
                              10,
                              0),
                    ]),
                0b010: Register(
                    'ddc_coeff',
                    [
                        Field('coeff_wren',
                              Access.Wpulse,
                              1,
                              0),
                        Field('coeff_wdata',
                              Access.RW,
                              18,
                              0),
                    ]),
                0b011: Register(
                    'ddc_decimation',
                    [
                        Field('decimation1',
                              Access.RW,
                              7,
                              0),
                        Field('decimation2',
                              Access.RW,
                              6,
                              0),
                        Field('decimation3',
                              Access.RW,
                              7,
                              0),
                    ]),
                0b100: Register(
                    'ddc_frequency',
                    [
                        Field('frequency',
                              Access.RW,
                              28,
                              0),
                    ]),
                0b101: Register(
                    'ddc_control',
                    [
                        Field('operations_minus_one1',
                              Access.RW,
                              7,
                              0),
                        Field('operations_minus_one2',
                              Access.RW,
                              6,
                              0),
                        Field('operations_minus_one3',
                              Access.RW,
                              7,
                              0),
                        Field('odd_operations1',
                              Access.RW,
                              1,
                              0),
                        Field('odd_operations3',
                              Access.RW,
                              1,
                              0),
                        Field('bypass2',
                              Access.RW,
                              1,
                              0),
                        Field('bypass3',
                              Access.RW,
                              1,
                              0),
                        Field('enable_input',
                              Access.RW,
                              1,
                              0),
                    ]),
            }, 3)
        # Airband multichannel receiver and its framed-audio DMA.
        self.receiver = ReceiverTop(
            n_channels=_AIRBAND_N_CHANNELS,
            chans_per_lane=_AIRBAND_CHANS_PER_LANE,
            decimation=_AIRBAND_LANE_DECIM,
            coeffs=_AIRBAND_FIR_COEFFS,
            out_shift=_AIRBAND_FIR_OUT_SHIFT,
            audio_decim=_AIRBAND_AUDIO_DECIM,
            cic_stages=_AIRBAND_CIC_STAGES,
            dcblock_k=_AIRBAND_DCBLOCK_K,
            in_width=12,
            nco_width=_AIRBAND_NCO_WIDTH,
            stages=_AIRBAND_STAGES,
            audio_sample_w=_AIRBAND_SAMPLE_W)
        self.airband_dma = DmaStreamWrite(
            config.airband_address_range[0],
            config.airband_address_range[1],
            width=64, cyclic=True, name='m_axi_airband')
        self.airband_registers = Registers(
            'airband',
            {
                0b00: Register('airband_control', [
                    Field('dma_start', Access.Wpulse, 1, 0),
                    Field('dma_stop', Access.Wpulse, 1, 0),
                    Field('enable', Access.RW, 1, 0),
                    Field('overflow', Access.R, 1, 0),
                ]),
                0b01: Register('airband_freq_addr', [
                    Field('freq_waddr', Access.RW,
                          len(self.receiver.freq_waddr), 0),
                ]),
                0b10: Register('airband_freq', [
                    Field('freq_wren', Access.Wpulse, 1, 0),
                    Field('freq_wdata', Access.RW, _AIRBAND_NCO_WIDTH, 0),
                ]),
                0b11: Register('airband_dma_next_address', [
                    Field('next_address', Access.R, 32, 0),
                ]),
            },
            2)
        metadata = {
            'vendor': 'Daniel Estevez',
            'vendorID': 'destevez.net',
            'name': 'Maia SDR',
            'series': 'Maia SDR',
            'version': _version,
            'description': f'Maia SDR IP core (platform {config.platform})',
            'licenseText': ('SPDX-License-Identifier: MIT '
                            'Copyright (C) Daniel Estevez 2022-2024'),
        }
        self.register_map = RegisterMap({
            0x0: self.control_registers,
            0x10: self.recorder_registers,
            0x20: self.sdr_registers,
            0x40: self.airband_registers,
        }, metadata)

        self.iq_in_width = 12
        self.re_in = Signal(self.iq_in_width)
        self.im_in = Signal(self.iq_in_width)
        self.interrupt_out = Signal()

    def ports(self):
        return (
            self.axi4lite.axi.ports()
            + self.spectrometer.dma.axi.ports()
            + self.recorder.dma.axi.ports()
            + self.airband_dma.axi.ports()
            + [
                self.re_in,
                self.im_in,
                self.interrupt_out,
                self.s_axi_lite.clk,
                self.s_axi_lite.rst,
                self.sampling.clk,
                self.sync.clk,
                self.sync.rst,
                self.clk2x.clk,
                self.clk3x.clk,
            ]
        )

    def svd(self):
        return self.register_map.svd()

    def elaborate(self, platform):
        m = Module()
        m.domains += [
            self.s_axi_lite,
            self.sampling,
            self.sync,
            self.clk2x,
            self.clk3x,
        ]
        s_axi_lite_renamer = DomainRenamer({'sync': 's_axi_lite'})
        m.submodules.axi4lite = s_axi_lite_renamer(self.axi4lite)
        m.submodules.control_registers = s_axi_lite_renamer(
            self.control_registers)
        m.submodules.recorder_registers = s_axi_lite_renamer(
            self.recorder_registers)
        m.submodules.spectrometer = self.spectrometer
        m.submodules.sync_spectrometer_interrupt = \
            sync_spectrometer_interrupt = PulseSynchronizer(
                i_domain='sync', o_domain='s_axi_lite')
        m.submodules.recorder = self.recorder
        m.submodules.ddc = self.ddc
        m.submodules.sdr_registers = self.sdr_registers
        m.submodules.sdr_registers_cdc = sdr_registers_cdc = RegisterCDC(
            's_axi_lite', 'sync', self.sdr_registers.aw)
        m.submodules.airband_registers = self.airband_registers
        m.submodules.airband_registers_cdc = airband_registers_cdc = \
            RegisterCDC('s_axi_lite', 'sync', self.airband_registers.aw)
        m.submodules.receiver = self.receiver
        m.submodules.airband_dma = self.airband_dma

        m.submodules.common_edge_2x = common_edge_2x = ClkNxCommonEdge(
            'sync', 'clk2x', 2)
        m.submodules.common_edge_3x = common_edge_3x = ClkNxCommonEdge(
            'sync', 'clk3x', 3)

        # RX IQ CDC
        m.submodules.rxiq_cdc = rxiq_cdc = RxIQCDC(
            'sampling', 'sync', self.iq_in_width)
        m.d.comb += [rxiq_cdc.re_in.eq(self.re_in),
                     rxiq_cdc.im_in.eq(self.im_in)]

        # Spectrometer (sync domain)
        spectrometer_re_in = Signal(
            self.spectrometer.width_in, reset_less=True)
        spectrometer_im_in = Signal(
            self.spectrometer.width_in, reset_less=True)
        assert len(spectrometer_re_in) == len(self.ddc.re_out)
        assert len(spectrometer_im_in) == len(self.ddc.im_out)
        spectrometer_strobe_in = Signal()
        with m.If(self.sdr_registers['spectrometer']['use_ddc_out']):
            m.d.sync += [
                spectrometer_re_in.eq(self.ddc.re_out),
                spectrometer_im_in.eq(self.ddc.im_out),
                spectrometer_strobe_in.eq(self.ddc.strobe_out),
            ]
        with m.Else():
            shift = self.spectrometer.width_in - self.iq_in_width
            m.d.sync += [
                # The RX IQ samples have 12 bits, but the spectrometer input
                # has 16 bits. Push the 12 bits to the MSBs.
                spectrometer_re_in.eq(rxiq_cdc.re_out << shift),
                spectrometer_im_in.eq(rxiq_cdc.im_out << shift),
                spectrometer_strobe_in.eq(rxiq_cdc.strobe_out),
            ]
        m.d.comb += [
            self.spectrometer.strobe_in.eq(spectrometer_strobe_in),
            self.spectrometer.common_edge_2x.eq(common_edge_2x.common_edge),
            self.spectrometer.common_edge_3x.eq(common_edge_3x.common_edge),
            self.spectrometer.re_in.eq(spectrometer_re_in),
            self.spectrometer.im_in.eq(spectrometer_im_in),
            sync_spectrometer_interrupt.i.eq(self.spectrometer.interrupt_out),
            self.spectrometer.number_integrations.eq(
                self.sdr_registers['spectrometer']['num_integrations']),
            self.spectrometer.abort.eq(
                self.sdr_registers['spectrometer']['abort']),
            self.spectrometer.peak_detect.eq(
                self.sdr_registers['spectrometer']['peak_detect']),
            self.sdr_registers['spectrometer']['last_buffer'].eq(
                self.spectrometer.last_buffer),
        ]

        # Recorder
        m.d.comb += [
            # sync domain
            self.recorder.strobe_in.eq(spectrometer_strobe_in),
            self.recorder.re_in.eq(spectrometer_re_in),
            self.recorder.im_in.eq(spectrometer_im_in),
            # s_axi_lite domain
            self.recorder.mode.eq(
                self.recorder_registers['recorder_control']['mode']),
            self.recorder.start.eq(
                self.recorder_registers['recorder_control']['start']),
            self.recorder.stop.eq(
                self.recorder_registers['recorder_control']['stop']),
            self.recorder_registers['recorder_control']['dropped_samples'].eq(
                self.recorder.dropped_samples),
            (self.recorder_registers['recorder_next_address']
             ['next_address'].eq(self.recorder.next_address)),
        ]

        # DDC
        m.d.comb += [
            self.ddc.common_edge.eq(common_edge_3x.common_edge),
            self.ddc.enable_input.eq(
                self.sdr_registers['ddc_control']['enable_input']),
            self.ddc.frequency.eq(
                self.sdr_registers['ddc_frequency']['frequency']),
            self.ddc.coeff_waddr.eq(
                self.sdr_registers['ddc_coeff_addr']['coeff_waddr']),
            self.ddc.coeff_wren.eq(
                self.sdr_registers['ddc_coeff']['coeff_wren']),
            self.ddc.coeff_wdata.eq(
                self.sdr_registers['ddc_coeff']['coeff_wdata']),
            self.ddc.decimation1.eq(
                self.sdr_registers['ddc_decimation']['decimation1']),
            self.ddc.decimation2.eq(
                self.sdr_registers['ddc_decimation']['decimation2']),
            self.ddc.decimation3.eq(
                self.sdr_registers['ddc_decimation']['decimation3']),
            self.ddc.bypass2.eq(
                self.sdr_registers['ddc_control']['bypass2']),
            self.ddc.bypass3.eq(
                self.sdr_registers['ddc_control']['bypass3']),
            self.ddc.operations_minus_one1.eq(
                self.sdr_registers['ddc_control']['operations_minus_one1']),
            self.ddc.operations_minus_one2.eq(
                self.sdr_registers['ddc_control']['operations_minus_one2']),
            self.ddc.operations_minus_one3.eq(
                self.sdr_registers['ddc_control']['operations_minus_one3']),
            self.ddc.odd_operations1.eq(
                self.sdr_registers['ddc_control']['odd_operations1']),
            self.ddc.odd_operations3.eq(
                self.sdr_registers['ddc_control']['odd_operations3']),
            self.ddc.strobe_in.eq(rxiq_cdc.strobe_out),
            self.ddc.re_in.eq(rxiq_cdc.re_out),
            self.ddc.im_in.eq(rxiq_cdc.im_out),
        ]

        # Airband multichannel receiver (sync domain). The wideband RX IQ
        # (post-CDC, 12-bit) is fed directly; per-channel NCO words and DMA
        # start/stop come from the airband register bank (also sync domain).
        airband_overflow = Signal()
        m.d.comb += [
            self.receiver.in_valid.eq(
                rxiq_cdc.strobe_out
                & self.airband_registers['airband_control']['enable']),
            self.receiver.re_in.eq(rxiq_cdc.re_out),
            self.receiver.im_in.eq(rxiq_cdc.im_out),
            self.receiver.freq_wren.eq(
                self.airband_registers['airband_freq']['freq_wren']),
            self.receiver.freq_waddr.eq(
                self.airband_registers['airband_freq_addr']['freq_waddr']),
            self.receiver.freq_wdata.eq(
                self.airband_registers['airband_freq']['freq_wdata']),
            # framed audio stream -> DMA
            self.airband_dma.stream_data.eq(self.receiver.stream_data),
            self.airband_dma.stream_valid.eq(self.receiver.stream_valid),
            self.receiver.stream_ready.eq(self.airband_dma.stream_ready),
            self.airband_dma.start.eq(
                self.airband_registers['airband_control']['dma_start']),
            self.airband_dma.stop.eq(
                self.airband_registers['airband_control']['dma_stop']),
            self.airband_registers['airband_control']['overflow'].eq(
                airband_overflow),
            (self.airband_registers['airband_dma_next_address']
             ['next_address'].eq(self.airband_dma.next_address)),
        ]
        # Sticky overflow: latch any real-time overrun until the DMA is
        # (re)started.
        with m.If(self.receiver.overflow):
            m.d.sync += airband_overflow.eq(1)
        with m.If(self.airband_registers['airband_control']['dma_start']):
            m.d.sync += airband_overflow.eq(0)

        # Registers s_axi_lite domain
        # TODO: convert all of this into a RegisterCrossbar module
        address = Signal(self.axi4_awidth, reset_less=True)
        wdata = Signal(32, reset_less=True)
        airband_regs_select = self.axi4lite.address[4] == 1
        sdr_regs_select = (
            ~airband_regs_select & (self.axi4lite.address[3] == 1))
        recorder_regs_select = (
            ~airband_regs_select & ~sdr_regs_select
            & (self.axi4lite.address[2] == 1))
        control_regs_select = (
            ~airband_regs_select & ~sdr_regs_select
            & (self.axi4lite.address[2] == 0))
        m.d.s_axi_lite += [
            self.axi4lite.rdata.eq(self.control_registers.rdata
                                   | self.recorder_registers.rdata
                                   | sdr_registers_cdc.i_rdata
                                   | airband_registers_cdc.i_rdata),
            self.axi4lite.rdone.eq(self.control_registers.rdone
                                   | self.recorder_registers.rdone
                                   | sdr_registers_cdc.i_rdone
                                   | airband_registers_cdc.i_rdone),
            self.axi4lite.wdone.eq(self.control_registers.wdone
                                   | self.recorder_registers.wdone
                                   | sdr_registers_cdc.i_wdone
                                   | airband_registers_cdc.i_wdone),
            self.control_registers.ren.eq(
                self.axi4lite.ren & control_regs_select),
            self.control_registers.wstrobe.eq(
                Mux(control_regs_select, self.axi4lite.wstrobe, 0)),
            self.recorder_registers.ren.eq(
                self.axi4lite.ren & recorder_regs_select),
            self.recorder_registers.wstrobe.eq(
                Mux(recorder_regs_select, self.axi4lite.wstrobe, 0)),
            sdr_registers_cdc.i_ren.eq(
                self.axi4lite.ren & sdr_regs_select),
            sdr_registers_cdc.i_wstrobe.eq(
                Mux(sdr_regs_select, self.axi4lite.wstrobe, 0)),
            airband_registers_cdc.i_ren.eq(
                self.axi4lite.ren & airband_regs_select),
            airband_registers_cdc.i_wstrobe.eq(
                Mux(airband_regs_select, self.axi4lite.wstrobe, 0)),
            address.eq(self.axi4lite.address),
            wdata.eq(self.axi4lite.wdata),
        ]
        m.d.comb += [
            self.control_registers.address.eq(address),
            self.control_registers.wdata.eq(wdata),
            self.recorder_registers.address.eq(address),
            self.recorder_registers.wdata.eq(wdata),
            sdr_registers_cdc.i_address.eq(address),
            sdr_registers_cdc.i_wdata.eq(wdata),
            airband_registers_cdc.i_address.eq(address),
            airband_registers_cdc.i_wdata.eq(wdata),
        ]

        # Registers sync domain
        m.d.comb += [
            self.sdr_registers.ren.eq(sdr_registers_cdc.o_ren),
            self.sdr_registers.wstrobe.eq(sdr_registers_cdc.o_wstrobe),
            self.sdr_registers.address.eq(sdr_registers_cdc.o_address),
            self.sdr_registers.wdata.eq(sdr_registers_cdc.o_wdata),
            sdr_registers_cdc.o_rdone.eq(self.sdr_registers.rdone),
            sdr_registers_cdc.o_wdone.eq(self.sdr_registers.wdone),
            sdr_registers_cdc.o_rdata.eq(self.sdr_registers.rdata),
            self.airband_registers.ren.eq(airband_registers_cdc.o_ren),
            self.airband_registers.wstrobe.eq(airband_registers_cdc.o_wstrobe),
            self.airband_registers.address.eq(airband_registers_cdc.o_address),
            self.airband_registers.wdata.eq(airband_registers_cdc.o_wdata),
            airband_registers_cdc.o_rdone.eq(self.airband_registers.rdone),
            airband_registers_cdc.o_wdone.eq(self.airband_registers.wdone),
            airband_registers_cdc.o_rdata.eq(self.airband_registers.rdata),
        ]
        # internal resets
        # We use FFSynchronizer rather than ResetSynchronizer because of
        # https://github.com/amaranth-lang/amaranth/issues/721
        for internal in ['sync', 'clk2x', 'clk3x', 'sampling']:
            setattr(m.submodules, f'{internal}_rst', FFSynchronizer(
                self.control_registers['control']['sdr_reset'],
                ResetSignal(internal), o_domain=internal,
                init=1))
        m.d.comb += rxiq_cdc.reset.eq(
            self.control_registers['control']['sdr_reset'])

        # Interrupts (s_axi_lite domain)
        interrupts_reg = self.control_registers['interrupts']
        m.d.comb += [
            self.interrupt_out.eq(interrupts_reg.interrupt),
            interrupts_reg['spectrometer'].eq(sync_spectrometer_interrupt.o),
            interrupts_reg['recorder'].eq(self.recorder.finished),
        ]

        return m


def write_svd(path):
    top = MaiaSDR()
    with open(path, 'wb') as f:
        f.write(top.svd())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', default='default',
        help='Maia SDR configuration name [default=%(default)r]')
    parser.add_argument(
        'output_file', help='Output verilog file')
    return parser.parse_args()


def main():
    args = parse_args()
    config = getattr(configs, args.config)()
    top = MaiaSDR(config)
    platform = PlutoPlatform()
    with open(args.output_file, 'w') as f:
        f.write(amaranth.back.verilog.convert(
            top, platform=platform, ports=top.ports()))


if __name__ == '__main__':
    main()
