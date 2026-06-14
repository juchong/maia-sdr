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
        self.recorder_address_range = (0x0100_0000, 0x1a00_0000)

        # Airband multichannel audio DMA ring (framed 64-bit audio records).
        # NOTE: this physical DDR region must be reconciled with the
        # reserved-memory devicetree node and maia-kmod buffer allocation
        # before use on hardware; here it is carved from the top of DDR
        # (above the spectrometer buffers) for the bitstream build.
        self.airband_address_range = (0x1f00_0000, 0x2000_0000)

    def validate(self):
        assert self.platform >= 0 and self.platform < 256
        assert self.spectrometer_buffers > 0
        assert self.spectrometer_buffers.bit_count() == 1
        assert self.recorder_address_range[0] < self.recorder_address_range[1]
        assert self.airband_address_range[0] < self.airband_address_range[1]
        # TODO: check that spectrometer, recorder, and airband buffers do not
        # overlap
