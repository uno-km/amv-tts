# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
import onnxruntime as ort
import numpy as np
from pathlib import Path

d = Path.home() / ".cache/termux-tts/models/sota/chattts"
sess_vocos = ort.InferenceSession(str(d / "vocos.int8.onnx"), providers=["CPUExecutionProvider"])
mel = np.zeros((1, 100, 50), dtype=np.float32)
out = sess_vocos.run(None, {"mel": mel})
print("mag shape:", out[0].shape, "min:", out[0].min(), "max:", out[0].max())
print("x shape:", out[1].shape, "min:", out[1].min(), "max:", out[1].max())
print("y shape:", out[2].shape, "min:", out[2].min(), "max:", out[2].max())
