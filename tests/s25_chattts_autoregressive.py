# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
import time
import numpy as np
import onnxruntime as ort
import wave
from pathlib import Path

print("=" * 65)
print("  GALAXY S25 CHATTTS TRUE AUTOREGRESSIVE ON-DEVICE SYNTHESIS")
print("=" * 65)

d = Path.home() / ".cache/termux-tts/models/sota/chattts"
spk_bin = d / "default_speaker.bin"
with open(spk_bin, "rb") as f:
    spk_emb = np.frombuffer(f.read(), dtype=np.float32).reshape(1, 768)

print(f"[1/5] Loaded default speaker embedding. Norm: {np.linalg.norm(spk_emb):.3f}")

# 1. Text to tokens
text = "물이 계속 솟는 땅에 조선이 궁궐을 세운 기적의 비밀. 경복궁 지하 배수로의 비밀."
vocab_file = d / "vocab.txt"
with open(vocab_file, "r", encoding="utf-8") as f:
    vocab = {line.strip(): i for i, line in enumerate(f)}

tokens = [vocab.get("[CLS]", 101)]
for c in text:
    tokens.append(vocab.get(c, vocab.get("[UNK]", 100)))
tokens.append(vocab.get("[SEP]", 102))

input_ids = np.array([tokens], dtype=np.int64)
N = input_ids.shape[1]
position_ids = np.arange(N, dtype=np.int64).reshape(1, N)
attention_mask = np.ones((1, 1, N, N), dtype=np.float32)
spk_pos = np.array([0], dtype=np.int64)
last_pos = np.array([N - 1], dtype=np.int64)

print(f"[2/5] Sequence length N={N}. Loading prefill session...")
sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 4

sess_prefill = ort.InferenceSession(str(d / "gpt_prefill.int8.onnx"), sess_opts, providers=["CPUExecutionProvider"])
sess_decode = ort.InferenceSession(str(d / "gpt_decode.int8.onnx"), sess_opts, providers=["CPUExecutionProvider"])

t0 = time.time()
prefill_inputs = {
    "input_ids": input_ids,
    "position_ids": position_ids,
    "attention_mask": attention_mask,
    "spk_emb": spk_emb,
    "spk_pos": spk_pos,
    "last_pos": last_pos,
}
prefill_outs = sess_prefill.run(None, prefill_inputs)
prefill_sec = time.time() - t0

logits = prefill_outs[0]  # [1, 626, 4]
hidden = prefill_outs[1]  # [1, 768]
past_k = [prefill_outs[2 + i] for i in range(20)]  # [1, N, 12, 64]
past_v = [prefill_outs[22 + i] for i in range(20)] # [1, N, 12, 64]

print(f"      Prefill OK in {prefill_sec:.3f}s. Initial logits: {logits.shape}")

# 2. Autoregressive Decoding Loop
max_decode_steps = 150
hidden_list = []
cur_logits = logits
cur_pos = N

print(f"[3/5] Starting Autoregressive Decode Loop (Target: {max_decode_steps} steps)...")
t_dec_start = time.time()

for step in range(max_decode_steps):
    # Sample vq_ids from cur_logits: [1, 626, 4] -> argmax over dim 1 -> [1, 1, 4]
    vq_ids = np.argmax(cur_logits, axis=1, keepdims=True) # shape [1, 1, 4]
    
    # Check if EOS (token 0 or 625 in codebook 0)
    top_tok = vq_ids[0, 0, 0]
    if step > 20 and top_tok in [0, 625]:
        print(f"      EOS encountered at step {step} (token {top_tok})")
        break
        
    pos_id = np.array([[cur_pos]], dtype=np.int64)
    attn_mask = np.ones((1, 1, 1, cur_pos + 1), dtype=np.float32)
    
    decode_inputs = {
        "vq_ids": vq_ids,
        "position_ids": pos_id,
        "attention_mask": attn_mask,
    }
    for i in range(20):
        decode_inputs[f"history_k_{i}"] = past_k[i]
        decode_inputs[f"history_v_{i}"] = past_v[i]
        
    dec_outs = sess_decode.run(None, decode_inputs)
    cur_logits = dec_outs[0]   # [1, 626, 4]
    step_hidden = dec_outs[1]  # [1, 768]
    hidden_list.append(step_hidden)
    
    # Update KV cache with new_k and new_v
    for i in range(20):
        new_k = dec_outs[2 + i]   # [1, 1, 12, 64]
        new_v = dec_outs[22 + i]  # [1, 1, 12, 64]
        past_k[i] = np.concatenate([past_k[i], new_k], axis=1)
        past_v[i] = np.concatenate([past_v[i], new_v], axis=1)
        
    cur_pos += 1
    if (step + 1) % 25 == 0 or step == max_decode_steps - 1:
        print(f"      Decoded {step + 1}/{max_decode_steps} tokens ({(time.time()-t_dec_start)/(step+1):.3f}s/token)...")

total_dec_time = time.time() - t_dec_start
num_frames = len(hidden_list)
print(f"      Autoregressive loop completed: {num_frames} frames in {total_dec_time:.2f}s ({total_dec_time/max(1, num_frames):.3f}s/frame)!")

# 3. DVAE Decoder & Vocos
print("[4/5] Running DVAE Decoder -> Mel Spectrogram...")
sess_decoder = ort.InferenceSession(str(d / "decoder.int8.onnx"), sess_opts, providers=["CPUExecutionProvider"])
sess_vocos = ort.InferenceSession(str(d / "vocos.int8.onnx"), sess_opts, providers=["CPUExecutionProvider"])

# dvae_input: shape [1, 768, T]
# Transpose each [1, 768] hidden into [1, 768, 1], then concatenate along axis 2
dvae_input = np.concatenate([h.reshape(1, 768, 1).astype(np.float32) for h in hidden_list], axis=2)
print(f"      dvae_input shape: {dvae_input.shape}")

mel = sess_decoder.run(["mel"], {"dvae_input": dvae_input})[0]
print(f"      Mel shape: {mel.shape} (min={mel.min():.2f}, max={mel.max():.2f})")

print("[5/5] Running Vocos -> Audio Waveform...")
t_vocos = time.time()
vocos_out = sess_vocos.run(None, {"mel": mel})
print(f"      Vocos completed in {time.time()-t_vocos:.2f}s")

mag = vocos_out[0][0]  # shape (513, T_mel)
cos_x = vocos_out[1][0]
sin_y = vocos_out[2][0]

# Overlap-Add iSTFT
n_fft = 1024
hop_length = 256
window = np.hanning(n_fft).astype(np.float32)
complex_spec = (mag * (cos_x + 1j * sin_y)).astype(np.complex64)
n_frames = complex_spec.shape[1]
total_len = (n_frames - 1) * hop_length + n_fft

audio = np.zeros(total_len, dtype=np.float32)
window_sum = np.zeros(total_len, dtype=np.float32)

for i in range(n_frames):
    frame_spec = complex_spec[:, i]
    time_frame = np.fft.irfft(frame_spec, n=n_fft)
    pos = i * hop_length
    audio[pos:pos+n_fft] += time_frame * window
    window_sum[pos:pos+n_fft] += window ** 2

mask = window_sum > 1e-4
audio[mask] /= window_sum[mask]

sr = 24000
duration = len(audio) / sr
peak = np.max(np.abs(audio))
rms = np.sqrt(np.mean(audio**2))
print(f"\n[DONE] Audio Synthesized: {len(audio)} samples ({duration:.2f}s), Peak={peak:.4f}, RMS={rms:.4f}")

out_wav = Path.home() / "tts_test_s25/s25_real_chattts_autoregressive.wav"
scaled_audio = (audio / max(0.01, peak) * 26000.0) if peak > 0 else audio
pcm16 = np.clip(scaled_audio, -32768, 32767).astype(np.int16)

with wave.open(str(out_wav), "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes(pcm16.tobytes())

print(f"Saved: {out_wav} ({out_wav.stat().st_size} bytes)")
print("=" * 65)
