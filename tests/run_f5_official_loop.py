# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
import onnxruntime as ort
import numpy as np
import wave
from pathlib import Path
import time
import sys

f5_dir = Path.home() / '.cache/termux-tts/models/sota/f5tts'
vocab_file = f5_dir / 'vocab.txt'
ref_audio_file = Path.home() / 'tts_test_s25/real_korean_shorts_10s.wav'

# 1. Load Vocab
with open(vocab_file, 'r', encoding='utf-8') as f:
    vocab_char_map = {line.strip('\r\n'): i for i, line in enumerate(f)}

print(f"Vocab loaded: {len(vocab_char_map)} tokens.")

# 2. Text Tokenization
text = "The secret was culvert engineering."
char_ids = [vocab_char_map.get(c, 0) for c in text]
text_ids = np.array([char_ids], dtype=np.int32)
print(f"Text '{text}' -> token ids: {char_ids[:10]}... len={len(char_ids)}")

# 3. Load Reference Audio
with wave.open(str(ref_audio_file), 'rb') as wf:
    n = wf.getnframes()
    pcm = np.frombuffer(wf.readframes(min(n, 48000)), dtype=np.int16).astype(np.float32) / 32768.0
ref_pcm = pcm.reshape(1, 1, -1).astype(np.float16)

# 4. Safe Duration (under 2048)
max_duration = np.array(min(900, max(200, len(text) * 10)), dtype=np.int64)

sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 4
providers = ['CPUExecutionProvider']

print("Loading ONNX sessions...")
sess_pre = ort.InferenceSession(str(f5_dir / 'F5_Preprocess.onnx'), sess_opts, providers=providers)
sess_trans = ort.InferenceSession(str(f5_dir / 'F5_Transformer.onnx'), sess_opts, providers=providers)
sess_dec = ort.InferenceSession(str(f5_dir / 'F5_Decode.onnx'), sess_opts, providers=providers)

# Check model metadata for NFE_STEP
meta = sess_trans.get_modelmeta().custom_metadata_map
nfe_step = int(meta.get("nfe_step", 16))
# For CPU speed on phone, clamp to 8~16 steps
test_steps = min(16, max(8, nfe_step))
print(f"Exported NFE: {nfe_step}, Testing with {test_steps} steps")

t0 = time.time()
pre_out = sess_pre.run(None, {
    'audio': ref_pcm,
    'text_ids': text_ids,
    'max_duration': max_duration
})
noise, rope_cos, rope_sin, cat_mel, cat_mel_drop, qk_empty, ref_len = pre_out
print(f"Preprocess OK ({time.time()-t0:.2f}s): noise shape={noise.shape}")

# 5. Denoising Loop (Feedback loop: noise_t+1 = denoised_t)
cur_noise = noise
t_dit = time.time()
print(f"Starting DiT feedback loop for {test_steps} steps...")
for step in range(test_steps):
    t_s = time.time()
    out = sess_trans.run(None, {
        'noise': cur_noise,
        'rope_cos': rope_cos,
        'rope_sin': rope_sin,
        'cat_mel_text': cat_mel,
        'cat_mel_text_drop': cat_mel_drop,
        'qk_rotated_empty': qk_empty,
        'time_step': np.array(step, dtype=np.int32)
    })
    cur_noise = out[0]
    print(f"  Step {step+1}/{test_steps} completed in {time.time()-t_s:.2f}s")

print(f"DiT completed in {time.time()-t_dit:.2f}s. Decoding waveform...")

# 6. Vocoder Decode
dec_out = sess_dec.run(None, {
    'denoised': cur_noise,
    'ref_signal_len': ref_len
})
raw_audio = dec_out[0].squeeze().astype(np.float32)

peak = np.max(np.abs(raw_audio))
if peak > 0:
    raw_audio = raw_audio / max(0.01, peak) * 0.90

out_wav = Path.home() / 'tts_test_s25/f5_official_verified.wav'
pcm_int16 = (raw_audio * 32767).astype(np.int16)
with wave.open(str(out_wav), 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(24000)
    wf.writeframes(pcm_int16.tobytes())

print(f"Saved: {out_wav}, size={out_wav.stat().st_size} bytes, duration={len(pcm_int16)/24000:.2f}s")
