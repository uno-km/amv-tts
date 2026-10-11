# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
import onnxruntime as ort
import numpy as np
import wave
from pathlib import Path
import time
import sys

f5_dir = Path.home() / ".cache/termux-tts/models/sota/f5tts"
vocab_file = f5_dir / "vocab.txt"
ref_audio_file = Path("tests/real_english_kokoro_10s.wav")

print("=" * 65)
print("  F5-TTS NVIDIA GTX 1660 SUPER (DirectML GPU) INFERENCE")
print("=" * 65)

# 1. Load Vocab
with open(vocab_file, "r", encoding="utf-8") as f:
    vocab_char_map = {line.strip("\r\n"): i for i, line in enumerate(f)}

print(f"[1/5] Vocab loaded: {len(vocab_char_map)} tokens.")

# 2. Text Tokenization
text = "The secret of how Joseon built royal palaces on swamp land."
char_ids = [vocab_char_map.get(c, 0) for c in text]
text_ids = np.array([char_ids], dtype=np.int32)
print(f"[2/5] Script: '{text}' ({len(char_ids)} tokens)")

# 3. Load Reference Audio (24kHz Mono float16)
with wave.open(str(ref_audio_file), "rb") as wf:
    n = wf.getnframes()
    pcm = np.frombuffer(wf.readframes(min(n, 48000)), dtype=np.int16).astype(np.float32) / 32768.0
ref_pcm = pcm.reshape(1, 1, -1).astype(np.float16)

# 4. Safe Sequence Duration
max_duration = np.array(min(900, max(200, len(text) * 11)), dtype=np.int64)

# GPU Session Options (DirectML)
sess_opts = ort.SessionOptions()
providers = ["DmlExecutionProvider", "CPUExecutionProvider"]

print("[3/5] Loading ONNX graphs onto NVIDIA GPU (DirectML)...")
t_load = time.time()
sess_pre = ort.InferenceSession(str(f5_dir / "F5_Preprocess.onnx"), sess_opts, providers=providers)
sess_trans = ort.InferenceSession(str(f5_dir / "F5_Transformer.onnx"), sess_opts, providers=providers)
sess_dec = ort.InferenceSession(str(f5_dir / "F5_Decode.onnx"), sess_opts, providers=providers)
print(f"      Models loaded onto GPU in {time.time()-t_load:.2f}s")

# 5. Preprocess
t0 = time.time()
pre_out = sess_pre.run(None, {
    "audio": ref_pcm,
    "text_ids": text_ids,
    "max_duration": max_duration
})
noise, rope_cos, rope_sin, cat_mel, cat_mel_drop, qk_empty, ref_len = pre_out
print(f"[4/5] Preprocess OK ({time.time()-t0:.2f}s): noise={noise.shape}")

# 6. GPU Accelerated DiT Feedback Loop (16 steps)
n_steps = 16
cur_noise = noise
t_dit = time.time()
print(f"[5/5] Running GPU DiT Flow Matching ({n_steps} steps)...")
for step in range(n_steps):
    step_t0 = time.time()
    out = sess_trans.run(None, {
        "noise": cur_noise,
        "rope_cos": rope_cos,
        "rope_sin": rope_sin,
        "cat_mel_text": cat_mel,
        "cat_mel_text_drop": cat_mel_drop,
        "qk_rotated_empty": qk_empty,
        "time_step": np.array(step, dtype=np.int32)
    })
    cur_noise = out[0]
    print(f"      Step {step+1:02d}/{n_steps} done in {time.time()-step_t0:.3f}s")

elapsed_dit = time.time() - t_dit
print(f"      GPU DiT completed in {elapsed_dit:.2f}s (Average {elapsed_dit/n_steps:.3f}s/step)!")

# 7. Decode
dec_out = sess_dec.run(None, {
    "denoised": cur_noise,
    "ref_signal_len": ref_len
})
raw_audio = dec_out[0].squeeze().astype(np.float32)

peak = np.max(np.abs(raw_audio))
if peak > 0:
    raw_audio = raw_audio / max(0.01, peak) * 0.90

out_wav = Path("tests/output_f5_gpu.wav")
pcm_int16 = (raw_audio * 32767).astype(np.int16)
with wave.open(str(out_wav), "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(24000)
    wf.writeframes(pcm_int16.tobytes())

dur = len(pcm_int16) / 24000.0
print(f"\n[DONE] Audio saved: {out_wav} (Duration: {dur:.2f}s, Peak: {np.max(np.abs(pcm_int16))})")

# 8. Whisper STT Self-Audit
print("\n[STT AUDIT] Self-auditing generated audio with Whisper...")
from faster_whisper import WhisperModel
stt_model = WhisperModel("tiny", device="cpu", compute_type="int8")
segs, info = stt_model.transcribe(str(out_wav), language="en")
stt_text = " ".join([s.text for s in segs]).strip()
print(f"  Detected Language: {info.language} (confidence: {info.language_probability:.2f})")
print(f"  STT Transcribed Text: \"{stt_text}\"")
verdict = "PASS (HUMAN SPEECH)" if len(stt_text) > 5 else "FAIL (NOISE)"
print(f"  Verdict: {verdict}")
print("=" * 65)
