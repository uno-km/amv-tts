# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
import time
import numpy as np
import onnxruntime as ort
import wave
from pathlib import Path

print("=" * 65)
print("  GALAXY S25 CHATTTS EXACT SPECIFICATION ON-DEVICE SYNTHESIS")
print("=" * 65)

d = Path.home() / ".cache/termux-tts/models/sota/chattts"

# 1. Load Speaker Embedding
spk_bin = d / "default_speaker.bin"
with open(spk_bin, "rb") as f:
    spk_emb = np.frombuffer(f.read(), dtype=np.float32).reshape(1, 768)

# 2. Vocabulary & Special Tokens
vocab_file = d / "vocab.txt"
with open(vocab_file, "r", encoding="utf-8") as f:
    vocab = {line.strip(): i for i, line in enumerate(f)}

SPK_EMB_ID = 21143
STTS_ID = 21131
ETTS_ID = 21133
LAUGH_ID = 21142
ORAL_1_ID = 21159 # Inhale / throat paralinguistic token
BREAK_4_ID = 21151

# Chinese Shorts Script (Natural Native Domain for ChatTTS with paralinguistic oral tokens)
# "在不断涌出地下水的土地上建造宏伟宫殿的秘密。历代工匠巧妙设计了地下暗渠系统。"
text_parts = [
    "在不断涌出地下水的土地上建造宏伟宫殿的秘密",
    "[oral_1]",
    "历代工匠巧妙设计了地下暗渠系统",
    "[laugh]"
]

input_tokens = [SPK_EMB_ID, STTS_ID]
for part in text_parts:
    if part == "[oral_1]":
        input_tokens.append(ORAL_1_ID)
    elif part == "[laugh]":
        input_tokens.append(LAUGH_ID)
    else:
        for c in part:
            input_tokens.append(vocab.get(c, vocab.get("[UNK]", 100)))
input_tokens.append(ETTS_ID)

input_ids = np.array([input_tokens], dtype=np.int64)
N = input_ids.shape[1]
position_ids = np.arange(N, dtype=np.int64).reshape(1, N)
attention_mask = np.ones((1, 1, N, N), dtype=np.float32)
spk_pos = np.array([0], dtype=np.int64) # Points exactly to SPK_EMB_ID at index 0
last_pos = np.array([N - 1], dtype=np.int64)

print(f"[1/5] Tokens prepared: length={N}, spk_pos=0, last_pos={N-1}")

# 3. ONNX Sessions
sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 4

print("[2/5] Loading ChatTTS INT8 ONNX models...")
sess_prefill = ort.InferenceSession(str(d / "gpt_prefill.int8.onnx"), sess_opts, providers=["CPUExecutionProvider"])
sess_decode = ort.InferenceSession(str(d / "gpt_decode.int8.onnx"), sess_opts, providers=["CPUExecutionProvider"])
sess_decoder = ort.InferenceSession(str(d / "decoder.int8.onnx"), sess_opts, providers=["CPUExecutionProvider"])
sess_vocos = ort.InferenceSession(str(d / "vocos.int8.onnx"), sess_opts, providers=["CPUExecutionProvider"])
print("      All 4 ONNX models loaded successfully.")

# 4. Run Prefill
print("[3/5] Running Prefill...")
t0 = time.time()
prefill_outs = sess_prefill.run(None, {
    "input_ids": input_ids,
    "position_ids": position_ids,
    "attention_mask": attention_mask,
    "spk_emb": spk_emb,
    "spk_pos": spk_pos,
    "last_pos": last_pos,
})
print(f"      Prefill completed in {time.time()-t0:.2f}s")

logits = prefill_outs[0]  # [1, 626, 4]
hidden = prefill_outs[1]  # [1, 768]
past_k = [prefill_outs[2 + i] for i in range(20)]
past_v = [prefill_outs[22 + i] for i in range(20)]

# 5. Autoregressive DVAE Token Decoding
max_decode_steps = 160
hidden_list = []
cur_logits = logits
cur_pos = N

print(f"[4/5] Running Autoregressive Token Decoding ({max_decode_steps} steps)...")
t_dec_start = time.time()

for step in range(max_decode_steps):
    # Temperature sampling / top-k for natural human prosody
    # logits shape: [1, 626, 4]
    sampled_vqs = []
    for c_idx in range(4):
        lg = cur_logits[0, :, c_idx]
        # Softmax with temperature 0.7
        lg_scaled = lg / 0.7
        exp_lg = np.exp(lg_scaled - np.max(lg_scaled))
        probs = exp_lg / np.sum(exp_lg)
        # Top-20 sample
        top_indices = np.argsort(probs)[-20:]
        top_probs = probs[top_indices] / np.sum(probs[top_indices])
        choice = np.random.choice(top_indices, p=top_probs)
        sampled_vqs.append(choice)
        
    vq_ids = np.array([[sampled_vqs]], dtype=np.int64) # shape: [1, 1, 4]
    
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
    cur_logits = dec_outs[0]
    step_hidden = dec_outs[1]
    hidden_list.append(step_hidden)
    
    for i in range(20):
        past_k[i] = np.concatenate([past_k[i], dec_outs[2 + i]], axis=1)
        past_v[i] = np.concatenate([past_v[i], dec_outs[22 + i]], axis=1)
        
    cur_pos += 1
    if (step + 1) % 40 == 0:
        print(f"      Decoded {step+1}/{max_decode_steps} frames...")

print(f"      Autoregressive loop completed in {time.time()-t_dec_start:.2f}s ({len(hidden_list)} frames)")

# 6. DVAE Decoder -> Mel Spectrogram
print("[5/5] Generating Waveform with Decoder & Vocos...")
dvae_input = np.concatenate([h.reshape(1, 768, 1).astype(np.float32) for h in hidden_list], axis=2)
mel = sess_decoder.run(["mel"], {"dvae_input": dvae_input})[0]

vocos_out = sess_vocos.run(None, {"mel": mel})
mag = np.exp(vocos_out[0][0])
mag = np.clip(mag, 0, 100.0)
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

scaled = (audio / max(0.01, peak) * 26000.0) if peak > 0 else audio
pcm16 = np.clip(scaled, -32768, 32767).astype(np.int16)

out_file = Path.home() / "tts_test_s25/chattts_sota_ondevice.wav"
with wave.open(str(out_file), "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes(pcm16.tobytes())

print(f"\n[SUCCESS] Synthesized {duration:.2f}s of pure SOTA ChatTTS audio!")
print(f"Saved: {out_file} ({out_file.stat().st_size} bytes)")
print("=" * 65)
