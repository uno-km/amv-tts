# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
import onnxruntime as ort
from pathlib import Path

d = Path.home() / ".cache/termux-tts/models/sota/f5tts"
for name in ["F5_Preprocess.onnx", "F5_Transformer.onnx", "F5_Decode.onnx"]:
    p = d / name
    if not p.exists():
        continue
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    print(f"\n=== {name} ===")
    print("Inputs:")
    for inp in sess.get_inputs():
        print(f"  {inp.name}: shape={inp.shape}, type={inp.type}")
    print("Outputs:")
    for out in sess.get_outputs():
        print(f"  {out.name}: shape={out.shape}, type={out.type}")
