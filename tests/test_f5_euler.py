# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
import onnxruntime as ort
import numpy as np
import wave
from pathlib import Path
import time

f5_dir = Path.home() / '.cache/termux-tts/models/sota/f5tts'
ref_audio_path = Path.home() / 'tts_test_s25/real_korean_shorts_10s.wav'

sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 4
providers = ['CPUExecutionProvider']

print("Loading F5-TTS ONNX models...")
sess_pre = ort.InferenceSession(str(f5_dir / 'F5_Preprocess.onnx'), sess_opts, providers=providers)
sess_trans = ort.InferenceSession(str(f5_dir / 'F5_Transformer.onnx'), sess_opts, providers=providers)
sess_dec = ort.InferenceSession(str(f5_dir / 'F5_Decode.onnx'), sess_opts, providers=providers)
print("F5 models loaded.")

# Load clean reference audio
with wave.open(str(ref_audio_path), 'rb') as wf:
    n = wf.getnframes()
    pcm = np.frombuffer(wf.readframes(min(n, 48000)), dtype=np.int16).astype(np.float32) / 32768.0
ref_pcm = pcm.reshape(1, 1, -1).astype(np.float16)

# Test single English sentence
text = "The royal architects constructed underground culverts."
text_ids = np.array([[ord(c) for c in text]], dtype=np.int32)
max_duration = np.array(450, dtype=np.int64)

print(f"Running Preprocess for: '{text}'...")
pre_out = sess_pre.run(None, {
    'audio': ref_pcm,
    'text_ids': text_ids,
    'max_duration': max_duration
})
noise, rope_cos, rope_sin, cat_mel, cat_mel_drop, qk_empty, ref_len = pre_out

# Euler ODE Sampling Loop (8 steps)
n_steps = 8
x = noise.copy()
dt = 1.0 / n_steps
print(f"Running Flow Matching ODE integration ({n_steps} steps)...")

t0 = time.time()
for step in range(n_steps):
    step_idx = np.array(step, dtype=np.int32)
    vt = sess_trans.run(None, {
        'noise': x,
        'rope_cos': rope_cos,
        'rope_sin': rope_sin,
        'cat_mel_text': cat_mel,
        'cat_mel_text_drop': cat_mel_drop,
        'qk_rotated_empty': qk_empty,
        'time_step': step_idx
    })[0]
    # Flow matching update: dx = v_t * dt
    x = (x.astype(np.float32) + vt.astype(np.float32) * dt).astype(np.float16)
    print(f"  Step {step+1}/{n_steps} done ({time.time()-t0:.2f}s)")

print("Decoding denoised mel to audio...")
dec_out = sess_dec.run(None, {
    'denoised': x,
    'ref_signal_len': ref_len
})
audio = dec_out[0].squeeze().astype(np.float32)

out_file = Path.home() / 'tts_test_s25/test_f5_euler_sample.wav'
pcm_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
with wave.open(str(out_file), 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(24000)
    wf.writeframes(pcm_int16.tobytes())

print(f"Saved: {out_file}, duration={len(pcm_int16)/24000:.2f}s, rms={np.sqrt(np.mean(pcm_int16.astype(float)**2)):.1f}")
