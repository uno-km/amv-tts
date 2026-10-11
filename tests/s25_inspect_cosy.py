# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
import onnxruntime as ort
from pathlib import Path

p = Path.home() / ".cache/termux-tts/models/sota/cosyvoice2/flow_hift_combined_fp16.onnx"
sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
print("=== CosyVoice 2 flow_hift_combined_fp16.onnx ===")
print("Inputs:")
for i in sess.get_inputs():
    print(f"  {i.name}: shape={i.shape}, type={i.type}")
print("Outputs:")
for o in sess.get_outputs():
    print(f"  {o.name}: shape={o.shape}, type={o.type}")
