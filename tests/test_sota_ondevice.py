# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
import onnxruntime as ort
import numpy as np
import wave
from pathlib import Path
import sys
import time

f5_dir = Path.home() / '.cache/termux-tts/models/sota/f5tts'
ref_audio_path = Path.home() / 'tts_test_s25/real_s25_chattts_e2e.wav'

# 1. Load ref audio
if ref_audio_path.exists():
    with wave.open(str(ref_audio_path), 'rb') as wf:
        n = wf.getnframes()
        pcm = np.frombuffer(wf.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
    ref_pcm = pcm[:min(n, 48000)].reshape(1, 1, -1).astype(np.float16)
else:
    ref_pcm = np.zeros((1, 1, 48000), dtype=np.float16)

# 2. Text tokens
text = "The secret was culvert engineering."
text_ids = np.array([[ord(c) for c in text]], dtype=np.int32)
max_duration = np.array(250, dtype=np.int64)

print(f"F5-TTS Preprocess starting: ref_pcm={ref_pcm.shape}, text_ids={text_ids.shape}")
sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 4
providers = ['CPUExecutionProvider']

t0 = time.time()
sess_pre = ort.InferenceSession(str(f5_dir / 'F5_Preprocess.onnx'), sess_opts, providers=providers)
pre_out = sess_pre.run(None, {
    'audio': ref_pcm,
    'text_ids': text_ids,
    'max_duration': max_duration
})

noise = pre_out[0]
rope_cos = pre_out[1]
rope_sin = pre_out[2]
cat_mel_text = pre_out[3]
cat_mel_text_drop = pre_out[4]
qk_rotated_empty = pre_out[5]
ref_signal_len = pre_out[6]

print(f"Preprocess OK ({time.time()-t0:.2f}s): noise={noise.shape}, cat_mel={cat_mel_text.shape}")

print("F5-TTS Transformer starting...")
t1 = time.time()
sess_trans = ort.InferenceSession(str(f5_dir / 'F5_Transformer.onnx'), sess_opts, providers=providers)
trans_out = sess_trans.run(None, {
    'noise': noise,
    'rope_cos': rope_cos,
    'rope_sin': rope_sin,
    'cat_mel_text': cat_mel_text,
    'cat_mel_text_drop': cat_mel_text_drop,
    'qk_rotated_empty': qk_rotated_empty,
    'time_step': np.array(0, dtype=np.int32)
})
denoised = trans_out[0]
print(f"Transformer OK ({time.time()-t1:.2f}s): denoised={denoised.shape}")

print("F5-TTS Decode starting...")
t2 = time.time()
sess_dec = ort.InferenceSession(str(f5_dir / 'F5_Decode.onnx'), sess_opts, providers=providers)
dec_out = sess_dec.run(None, {
    'denoised': denoised,
    'ref_signal_len': ref_signal_len
})
audio_out = dec_out[0]
print(f"Decode OK ({time.time()-t2:.2f}s): audio_out={audio_out.shape}, dtype={audio_out.dtype}")

# Save WAV
out_wav = Path.home() / 'tts_test_s25/real_s25_f5_step.wav'
out_pcm = (np.clip(audio_out.squeeze().astype(np.float32), -1.0, 1.0) * 32767).astype(np.int16)
with wave.open(str(out_wav), 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(24000)
    wf.writeframes(out_pcm.tobytes())
print(f"Saved to {out_wav}, size={out_wav.stat().st_size} bytes")
