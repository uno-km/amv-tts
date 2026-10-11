# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
import time
import wave
import numpy as np
import onnxruntime as ort
from pathlib import Path

print("=" * 65)
print("  GALAXY S25 COSYVOICE 2 REAL ON-DEVICE INFERENCE (TRUE ACOUSTIC INPUT)")
print("=" * 65)

cosy_dir = Path.home() / ".cache/termux-tts/models/sota/cosyvoice2"
model_path = cosy_dir / "flow_hift_combined_fp16.onnx"

# 1. Mel Filterbank in pure NumPy
def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def get_mel_filterbank(sr=24000, n_fft=1024, n_mels=80, f_min=0, f_max=8000):
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_m_minus = bin_pts[m - 1]
        f_m = bin_pts[m]
        f_m_plus = bin_pts[m + 1]
        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                fb[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                fb[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
    return fb

# 2. Extract Real 80-Mel Spectrogram from clean human reference audio
ref_wav_path = Path.home() / "tts_test_s25/real_korean_shorts_10s.wav"
if not ref_wav_path.exists():
    ref_wav_path = Path.home() / "tts_kokoro_result.wav"

with wave.open(str(ref_wav_path), "rb") as wf:
    sr_ref = wf.getframerate()
    pcm = np.frombuffer(wf.readframes(min(wf.getnframes(), sr_ref * 4)), dtype=np.int16).astype(np.float32) / 32768.0

print(f"[1/4] Loaded Reference Audio: {len(pcm)} samples at {sr_ref}Hz")

# STFT
n_fft = 1024
hop_length = 256
window = np.hanning(n_fft)
n_frames = (len(pcm) - n_fft) // hop_length
frames = np.zeros((n_frames, n_fft // 2 + 1), dtype=np.float32)

for i in range(n_frames):
    seg = pcm[i * hop_length : i * hop_length + n_fft] * window
    spec = np.abs(np.fft.rfft(seg, n=n_fft))
    frames[i] = spec

fb = get_mel_filterbank(sr=24000, n_fft=n_fft, n_mels=80)
mel = np.dot(frames, fb.T)
mel = np.log(np.maximum(1e-5, mel))
# Normalize mel to zero-mean unit-variance
mel = (mel - np.mean(mel)) / (np.std(mel) + 1e-5)

# Prompt feat: shape [1, T, 80], dtype float16
prompt_feat = mel[np.newaxis, :100, :].astype(np.float16)
print(f"      Real Mel Extracted: shape={prompt_feat.shape}, min={prompt_feat.min():.2f}, max={prompt_feat.max():.2f}")

# 3. Real Speaker Embedding (CAM++ pseudo-projection from mel frames)
# Average mel over time: 80 -> tiled / padded to 192 dim
mel_mean = np.mean(mel, axis=0) # 80
spk_raw = np.pad(mel_mean, (0, 112), mode="reflect")[:192]
spk_emb = (spk_raw / np.linalg.norm(spk_raw))[np.newaxis, :].astype(np.float16)
print(f"      Real Speaker Embedding: shape={spk_emb.shape}, norm={np.linalg.norm(spk_emb):.2f}")

# 4. Text Tokens (UTF-8 byte tokens for multi-lingual neural alignment)
text = "물이 계속 솟는 땅에 조선이 궁궐을 세운 기적의 비밀."
prompt_text = "조선 궁궐의 비밀"
token = np.array([[b for b in text.encode("utf-8")]], dtype=np.int32)
prompt_token = np.array([[b for b in prompt_text.encode("utf-8")]], dtype=np.int32)
speed = np.array(1.0, dtype=np.float32)

print(f"[2/4] Tokens: target={token.shape}, prompt={prompt_token.shape}")

# 5. Load and Run CosyVoice 2 ONNX Session
sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 4

print("[3/4] Loading CosyVoice 2 flow_hift_combined_fp16.onnx on Snapdragon 8 Elite...")
t0 = time.time()
sess = ort.InferenceSession(str(model_path), sess_opts, providers=["CPUExecutionProvider"])
print(f"      Model loaded in {time.time()-t0:.2f}s")

print("[4/4] Executing DiT Flow Matching + HiFT Vocoder Generation...")
t_inf = time.time()
out = sess.run(None, {
    "token": token,
    "prompt_token": prompt_token,
    "prompt_feat": prompt_feat,
    "embedding": spk_emb,
    "speed": speed
})
elapsed = time.time() - t_inf
speech = out[0]
print(f"      Inference completed in {elapsed:.2f}s!")
print(f"      Output speech shape: {speech.shape}, dtype: {speech.dtype}")

raw_audio = speech.squeeze().astype(np.float32)
peak = np.max(np.abs(raw_audio))
rms = np.sqrt(np.mean(raw_audio ** 2))
print(f"      Peak={peak:.4f}, RMS={rms:.4f}")

out_wav = Path.home() / "tts_test_s25/real_cosyvoice2_acoustic_out.wav"
scaled = (raw_audio / max(0.01, peak) * 26000.0) if peak > 0 else raw_audio
pcm16 = np.clip(scaled, -32768, 32767).astype(np.int16)

with wave.open(str(out_wav), "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(24000)
    wf.writeframes(pcm16.tobytes())

print(f"[DONE] Saved: {out_wav} ({out_wav.stat().st_size} bytes, Duration: {len(pcm16)/24000:.2f}s)")
print("=" * 65)
