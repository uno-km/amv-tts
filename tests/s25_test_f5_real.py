# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
import time
import numpy as np
import onnxruntime as ort
import wave
from pathlib import Path

print("=" * 65)
print("  GALAXY S25 F5-TTS ON-DEVICE PROPER INFERENCE (NO FAKE STUBS)")
print("=" * 65)

f5_dir = Path.home() / ".cache/termux-tts/models/sota/f5tts"
vocab_file = f5_dir / "vocab.txt"

# 1. Load Vocab mapping
with open(vocab_file, "r", encoding="utf-8") as f:
    vocab = {line.strip("\r\n"): i for i, line in enumerate(f)}
print(f"[1/5] Loaded vocab with {len(vocab)} tokens.")

# 2. Text and reference audio
text = "The secret of how Joseon built royal palaces on swamp land."
print(f"[2/5] Script: '{text}'")

char_ids = []
for c in text:
    if c in vocab:
        char_ids.append(vocab[c])
    else:
        print(f"Warning: Char '{c}' missing from vocab, using space (0)")
        char_ids.append(0)

text_ids = np.array([char_ids], dtype=np.int32)
print(f"      Tokenized length: {text_ids.shape[1]}")

ref_audio_path = Path.home() / "tts_test_s25/real_korean_shorts_10s.wav"
if not ref_audio_path.exists():
    ref_audio_path = Path.home() / "tts_kokoro_result.wav"

with wave.open(str(ref_audio_path), "rb") as wf:
    n = wf.getnframes()
    pcm = np.frombuffer(wf.readframes(min(n, 48000)), dtype=np.int16).astype(np.float32) / 32768.0
ref_pcm = pcm.reshape(1, 1, -1).astype(np.float16)
print(f"      Loaded reference audio: {ref_audio_path} ({ref_pcm.shape[-1]} samples)")

# 3. ONNX Sessions
sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 6

print("[3/5] Loading F5-TTS ONNX graphs on Snapdragon 8 Elite...")
t_load = time.time()
sess_pre = ort.InferenceSession(str(f5_dir / "F5_Preprocess.onnx"), sess_opts, providers=["CPUExecutionProvider"])
sess_trans = ort.InferenceSession(str(f5_dir / "F5_Transformer.onnx"), sess_opts, providers=["CPUExecutionProvider"])
sess_dec = ort.InferenceSession(str(f5_dir / "F5_Decode.onnx"), sess_opts, providers=["CPUExecutionProvider"])
print(f"      Loaded in {time.time()-t_load:.2f}s")

# 4. Preprocess
max_duration = np.array(min(600, max(200, len(text) * 9)), dtype=np.int64)
t0 = time.time()
pre_out = sess_pre.run(None, {
    "audio": ref_pcm,
    "text_ids": text_ids,
    "max_duration": max_duration
})
noise, rope_cos, rope_sin = pre_out[0], pre_out[1], pre_out[2]
cat_mel_text, cat_mel_text_drop, qk_rotated_empty = pre_out[3], pre_out[4], pre_out[5]
ref_signal_len = pre_out[6]
print(f"[4/5] Preprocess OK in {time.time()-t0:.2f}s. Initial noise shape: {noise.shape}")

# 5. Denoising Loop (8 steps for on-device benchmark)
n_steps = 8
cur_noise = noise
print(f"[5/5] Running Denoising Loop ({n_steps} steps)...")
t_loop = time.time()

for step in range(n_steps):
    step_t0 = time.time()
    out = sess_trans.run(None, {
        "noise": cur_noise,
        "rope_cos": rope_cos,
        "rope_sin": rope_sin,
        "cat_mel_text": cat_mel_text,
        "cat_mel_text_drop": cat_mel_text_drop,
        "qk_rotated_empty": qk_rotated_empty,
        "time_step": np.array(step, dtype=np.int32)
    })
    cur_noise = out[0]
    print(f"      Step {step+1:02d}/{n_steps} completed in {time.time()-step_t0:.3f}s")

total_loop_time = time.time() - t_loop
print(f"      Loop finished in {total_loop_time:.2f}s ({total_loop_time/n_steps:.3f}s/step)")

# 6. Vocoder Decode
print("Decoding denoised spectrogram with Vocos...")
t_dec = time.time()
dec_out = sess_dec.run(None, {
    "denoised": cur_noise,
    "ref_signal_len": ref_signal_len
})
audio_out = dec_out[0].squeeze().astype(np.float32)
print(f"Decoded in {time.time()-t_dec:.2f}s! Raw audio shape: {audio_out.shape}")

peak = np.max(np.abs(audio_out))
rms = np.sqrt(np.mean(audio_out**2))
print(f"Raw Peak: {peak:.4f}, RMS: {rms:.4f}")

out_wav = Path.home() / "tts_test_s25/s25_real_f5_step.wav"
scaled_audio = (audio_out / max(0.01, peak) * 26000.0) if peak > 0 else audio_out
pcm16 = np.clip(scaled_audio, -32768, 32767).astype(np.int16)

with wave.open(str(out_wav), "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(24000)
    wf.writeframes(pcm16.tobytes())

print(f"Saved: {out_wav} ({out_wav.stat().st_size} bytes, Duration: {len(pcm16)/24000:.2f}s)")
print("=" * 65)
