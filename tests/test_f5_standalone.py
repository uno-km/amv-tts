# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
import onnxruntime as ort
import numpy as np
import wave
from pathlib import Path
import time

f5_dir = Path.home() / '.cache/termux-tts/models/sota/f5tts'
ref_audio_path = Path.home() / 'tts_test_s25/s25_real_chattts_10s.wav'

sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 4
providers = ['CPUExecutionProvider']

print("Loading F5-TTS models...")
sess_pre = ort.InferenceSession(str(f5_dir / 'F5_Preprocess.onnx'), sess_opts, providers=providers)
sess_trans = ort.InferenceSession(str(f5_dir / 'F5_Transformer.onnx'), sess_opts, providers=providers)
sess_dec = ort.InferenceSession(str(f5_dir / 'F5_Decode.onnx'), sess_opts, providers=providers)
print("Models loaded successfully.")

# Load reference audio
with wave.open(str(ref_audio_path), 'rb') as wf:
    n = wf.getnframes()
    pcm = np.frombuffer(wf.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
ref_pcm = pcm[:min(n, 48000)].reshape(1, 1, -1).astype(np.float16)

sentences = [
    "The secret of how Joseon built royal palaces on swamp land.",
    "Royal architects constructed sophisticated underground culverts to drain the water."
]

audio_pieces = []

for idx, sent in enumerate(sentences):
    print(f"\nProcessing sentence {idx+1}: '{sent}'")
    text_ids = np.array([[ord(c) for c in sent]], dtype=np.int32)
    # Strictly safe length: 400 ~ 600 frames (under 2048!)
    max_duration = np.array(min(900, max(200, len(sent) * 9)), dtype=np.int64)

    t0 = time.time()
    pre_out = sess_pre.run(None, {
        'audio': ref_pcm,
        'text_ids': text_ids,
        'max_duration': max_duration
    })
    noise, rope_cos, rope_sin, cat_mel, cat_mel_drop, qk_empty, ref_len = pre_out
    print(f"  Preprocess OK in {time.time()-t0:.2f}s")

    t1 = time.time()
    trans_out = sess_trans.run(None, {
        'noise': noise,
        'rope_cos': rope_cos,
        'rope_sin': rope_sin,
        'cat_mel_text': cat_mel,
        'cat_mel_text_drop': cat_mel_drop,
        'qk_rotated_empty': qk_empty,
        'time_step': np.array(0, dtype=np.int32)
    })
    denoised = trans_out[0]
    print(f"  Transformer OK in {time.time()-t1:.2f}s")

    t2 = time.time()
    dec_out = sess_dec.run(None, {
        'denoised': denoised,
        'ref_signal_len': ref_len
    })
    pcm_out = dec_out[0].squeeze().astype(np.float32)
    print(f"  Vocoder Decode OK in {time.time()-t2:.2f}s, samples={len(pcm_out)}")
    audio_pieces.append(pcm_out)
    # 0.2s pause between sentences
    audio_pieces.append(np.zeros(int(24000 * 0.2), dtype=np.float32))

full_audio = np.concatenate(audio_pieces)
peak = np.max(np.abs(full_audio))
if peak > 0:
    full_audio = full_audio / max(0.01, peak) * 0.88

out_wav = Path.home() / 'tts_test_s25/final_f5_perfect_10s.wav'
pcm_int16 = (full_audio * 32767).astype(np.int16)
with wave.open(str(out_wav), 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(24000)
    wf.writeframes(pcm_int16.tobytes())

dur = len(pcm_int16) / 24000.0
rms = float(np.sqrt(np.mean(pcm_int16.astype(np.float64)**2)))
print(f"\n[F5-TTS SUCCESS] Saved to {out_wav}")
print(f"  Duration: {dur:.2f}s | RMS: {rms:.1f} | Peak: {np.max(np.abs(pcm_int16))}")
