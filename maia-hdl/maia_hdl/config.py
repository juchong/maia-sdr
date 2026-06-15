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
        # The top 16 MiB of the original recording region is carved off for the
        # airband audio ring (see below), so the recorder ends at 0x1900_0000
        # instead of 0x1a00_0000.
        self.recorder_address_range = (0x0100_0000, 0x1900_0000)

        # Airband multichannel audio DMA ring (framed 64-bit audio records).
        #
        # This MUST sit inside the maia-sdr reserved-memory block and MUST NOT
        # overlap the kernel's default CMA region, which the Zynq FPGA manager
        # places at the very top of usable DDR (0x1f00_0000..0x2000_0000 on the
        # 512 MiB Pluto). Reserving that top range as `no-map` collides with CMA
        # and hangs the kernel before USB comes up. Instead the ring is carved
        # from the top of the (already reserved) recorder region, so usable RAM
        # and CMA placement stay byte-for-byte identical to stock maia-sdr. The
        # reserved-memory devicetree node (apply_airband_devicetree.py) shrinks
        # the recording region to match and adds this node at 0x1900_0000.
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
