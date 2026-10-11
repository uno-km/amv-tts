# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
engine_onnx.py — Backward-compatibility redirect to SherpaNeuralEngine.
Python onnxruntime is permanently deprecated in favor of C++ isolated SherpaNeuralEngine.
"""
from .engine_sherpa import SherpaNeuralEngine as ONNXNeuralEngine
from .engine_sherpa import SherpaResult as ONNXResult

__all__ = ["ONNXNeuralEngine", "ONNXResult"]
