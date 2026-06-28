#
# Copyright (C) 2024 Daniel Estevez <daniel@destevez.net>
#
# This file is part of maia-sdr
#
# SPDX-License-Identifier: MIT
#

class MaiaSDRConfig:
    """Maia SDR configuration

    This class defines configuration parameters for the Maia SDR top-level.
    """
    def __init__(self):
        # create default configuration

        # general
        self.platform = 0

        # spectrometer
        self.spectrometer_address = 0x1a00_0000
        self.spectrometer_buffers = 8

        # IQ recorder
        #
        # The stock recorder reserves a huge ~384 MiB DDR region (0x0100_0000..
        # 0x1900_0000) as `no-map`, which (with the airband + spectrometer
        # reserves) leaves Linux only ~96 MiB of usable RAM -- the root cause of
        # the maia-httpd OOM race on the airband build, which never uses the IQ
        # recorder. We therefore SHRINK the recorder reserve to 16 MiB
        # (0x0100_0000..0x0200_0000), returning ~368 MiB to userspace
        # (~96 MiB -> ~464 MiB) while keeping a small, still-functional recorder.
        # The recorder DMA only ever writes inside this range (and only when a
        # recording is started, which the airband deployment never does), so the
        # freed region is safe for Linux. This MUST be kept in lockstep with the
        # devicetree reserved-memory `maia_sdr_recording` reg
        # (firmware/apply_airband_devicetree.py RECORDING_SIZE) and shipped as a
        # set (bitstream + DT together) per BUILD.md.
        self.recorder_address_range = (0x0100_0000, 0x0200_0000)

        # Airband multichannel audio DMA ring (framed 64-bit audio records).
        #
        # This MUST sit inside the maia-sdr reserved-memory block and MUST NOT
        # overlap the kernel's default CMA region, which the Zynq FPGA manager
        # places at the very top of usable DDR (0x1f00_0000..0x2000_0000 on the
        # 512 MiB Pluto). Reserving that top range as `no-map` collides with CMA
        # and hangs the kernel before USB comes up, so the ring is placed well
        # below CMA at 0x1900_0000. (It used to be carved from the top of the
        # full 384 MiB recorder region; now that the recorder reserve is shrunk
        # to 16 MiB the ring is a standalone reserved node, with the reclaimed
        # 0x0200_0000..0x1900_0000 returned to Linux in between. CMA placement is
        # unchanged, so this only ever GIVES RAM back.) The reserved-memory
        # devicetree node (apply_airband_devicetree.py) adds this node at
        # 0x1900_0000 and shrinks the recording region to match config above.
        self.airband_address_range = (0x1900_0000, 0x1a00_0000)

    def validate(self):
        assert self.platform >= 0 and self.platform < 256
        assert self.spectrometer_buffers > 0
        assert self.spectrometer_buffers.bit_count() == 1
        assert self.recorder_address_range[0] < self.recorder_address_range[1]
        assert self.airband_address_range[0] < self.airband_address_range[1]
        # The recorder and airband DMA regions must not overlap (for the
        # `default` config the airband ring is carved from the top of the
        # recorder region, so they are adjacent but disjoint).
        assert (self.airband_address_range[1] <= self.recorder_address_range[0]
                or self.airband_address_range[0]
                >= self.recorder_address_range[1])
