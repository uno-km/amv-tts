# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""termux_tts.control — AMEVA Component Protocol v1."""
from .component import TTSControl
from .status import TTSStatusWriter
__all__ = ["TTSControl", "TTSStatusWriter"]
