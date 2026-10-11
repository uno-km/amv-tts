# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
import onnxruntime as ort
from pathlib import Path

d = Path.home() / ".cache/termux-tts/models/sota/chattts"
for name in ["gpt_prefill.int8.onnx", "gpt_decode.int8.onnx", "decoder.int8.onnx", "vocos.int8.onnx"]:
    p = d / name
    if not p.exists():
        print(f"Missing: {p}")
        continue
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    print(f"\n=== {name} ===")
    print("Inputs:")
    for inp in sess.get_inputs():
        print(f"  {inp.name}: shape={inp.shape}, type={inp.type}")
    print("Outputs:")
    for out in sess.get_outputs():
        print(f"  {out.name}: shape={out.shape}, type={out.type}")
