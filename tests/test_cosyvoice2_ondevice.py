# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
import onnxruntime as ort
import numpy as np
import wave
from pathlib import Path
import time

cosy_dir = Path.home() / '.cache/termux-tts/models/sota/cosyvoice2'
model_path = cosy_dir / 'flow_hift_combined_fp16.onnx'

sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 4
providers = ['CPUExecutionProvider']

print("Loading CosyVoice 2 flow_hift_combined_fp16.onnx...")
t0 = time.time()
sess = ort.InferenceSession(str(model_path), sess_opts, providers=providers)
print(f"Loaded in {time.time()-t0:.2f}s")

# Tokens: Korean/Multilingual character or BPE tokens
text = "물이 계속 솟는 땅에 조선이 궁궐을 세운 기적의 비밀"
token = np.array([[ord(c) % 4096 for c in text]], dtype=np.int32)
prompt_token = np.array([[ord(c) % 4096 for c in "조선 궁궐 비밀"]], dtype=np.int32)

# Prompt feat: 80-dim log-mel spectrogram [1, prompt_feat_len, 80]
prompt_feat_len = 50
prompt_feat = np.random.randn(1, prompt_feat_len, 80).astype(np.float16) * 0.1

# Speaker embedding: 192-dim CAM++ or ERes2Net embedding [1, 192]
spk_emb = np.random.randn(1, 192).astype(np.float16)
spk_emb = spk_emb / np.linalg.norm(spk_emb)

speed = np.array(1.0, dtype=np.float32)

print(f"Running CosyVoice 2 inference: token={token.shape}, feat={prompt_feat.shape}, emb={spk_emb.shape}...")
t1 = time.time()
out = sess.run(None, {
    'token': token,
    'prompt_token': prompt_token,
    'prompt_feat': prompt_feat,
    'embedding': spk_emb,
    'speed': speed
})
print(f"CosyVoice 2 inference OK in {time.time()-t1:.2f}s!")
generated_speech = out[0]
print(f"Output shape: {generated_speech.shape}, dtype: {generated_speech.dtype}")

# Save WAV
out_wav = Path.home() / 'tts_test_s25/real_s25_cosy_step.wav'
speech_data = np.clip(generated_speech.squeeze(), -1.0, 1.0)
out_pcm = (speech_data * 32767).astype(np.int16)
with wave.open(str(out_wav), 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(24000)
    wf.writeframes(out_pcm.tobytes())
print(f"Saved to {out_wav}, size={out_wav.stat().st_size} bytes")
