# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
Vulkan GPU Neural Speech Synthesis Engine for termux-tts.
Runs 100% on-device neural inference using Vulkan Compute Shaders.
Supports:
1. Piper VITS NCNN (via sherpa-ncnn-offline-tts CLI compute shaders)
2. MeloTTS Plan 1 (HiFi-GAN Decoder NCNN Vulkan Slicing)
3. MeloTTS Plan 2 (MNN Vulkan End-to-End Compute Pipeline)

Strict Zero-Silent-Fallback [AMEVA-TTS-E001]:
- If Vulkan GPU or required model assets are not available, fails fast with VulkanInitializationError or TTSModelLoadError.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np

from .audio import AudioBuffer
from .exceptions import TTSModelLoadError, TTSInferenceError, VulkanInitializationError
from .hardware import get_clean_execution_env
from .tokenizer_melo import MeloTokenizer

logger = logging.getLogger("termux_tts.engine_vulkan")


@dataclass
class VulkanResult:
    text: str
    audio_buffer: AudioBuffer
    sample_rate: int
    duration_sec: float
    elapsed_ms: float
    rtf: float
    model_name: str
    backend: str
    gpu_device: str
    strategy: str = "sherpa_vits"

    def save(self, filepath: str) -> str:
        return self.audio_buffer.save(filepath)

    @property
    def wav_bytes(self) -> bytes:
        return self.audio_buffer.to_wav_bytes()


class VulkanNeuralEngine:
    """Production Vulkan GPU Neural Speech Synthesis Engine."""

    @staticmethod
    def _get_search_prefixes() -> List[Path]:
        prefixes = [Path.home()]
        prefix_env = os.environ.get("PREFIX")
        if prefix_env:
            prefixes.append(Path(prefix_env))
        return prefixes

    @classmethod
    def get_candidate_binaries(cls, binary_name: str) -> List[str]:
        candidates = [binary_name]
        for p in cls._get_search_prefixes():
            candidates.extend([
                str(p / "bin" / binary_name),
                str(p / ".local" / "bin" / binary_name),
                str(p / binary_name),
            ])
        return candidates

    @classmethod
    def get_model_search_dirs(cls) -> List[Path]:
        dirs: List[Path] = []
        env_models = os.environ.get("AMEVA_MODELS_DIR")
        if env_models:
            dirs.append(Path(env_models))

        home = Path.home()
        dirs.extend([
            home / "models" / "tts",
            home / "models",
            home / ".cache" / "termux-tts" / "models",
            home / "vits-melo-tts-zh_en",
            home / "melo-mnn",
            home / "ncnn-vits-piper-en_US-lessac-high-fp16",
            home / "ncnn-vits-piper-en_US-amy-medium",
        ])
        return [d for d in dirs if d.exists()]

    def __init__(
        self,
        model_path: Optional[str] = None,
        language: str = "en",
        device: str = "vulkan",
        threads: int = 1,
        sample_rate: Optional[int] = None,
        model_tier: Optional[str] = None,
        model_type: str = "vits",
    ):
        self.language = language.lower()
        self.requested_device = device.lower()
        self.threads = threads
        self.model_tier = (model_tier or "high").lower()
        self.model_type = (model_type or "vits").lower()
        self._is_closed = False
        self.gpu_device = "Vulkan GPU"

        if self.model_type in ("melo", "melo_vulkan", "melo_mnn"):
            self._init_melo(model_path, sample_rate)
        else:
            self._init_piper_vits(model_path, sample_rate)

    def _find_binary(self, binary_name: str) -> str:
        candidates = self.get_candidate_binaries(binary_name)
        for cand in candidates:
            found = shutil.which(cand) if not os.path.isabs(cand) else cand
            if found and os.path.isfile(found) and (os.access(found, os.X_OK) or os.name == "nt"):
                return str(Path(found).resolve())

        raise VulkanInitializationError(
            f"[FAIL-FAST] [ERROR: AMEVA-TTS-E001] Native Vulkan executable '{binary_name}' not found.\n"
            f"Cause: Hardware acceleration native binary is not installed in standard binary paths.\n"
            f"Candidate paths checked: {candidates}\n"
            f"Action Required: Provision native executable via AMEVA toolchain."
        )

    def _init_piper_vits(self, model_path: Optional[str], sample_rate: Optional[int]) -> None:
        self.binary = self._find_binary("sherpa-ncnn-offline-tts")
        self.model_dir = self._resolve_piper_model_dir(model_path)
        self.model_name = Path(self.model_dir).name
        self.backend = "VULKAN_GPU_NCNN_VITS"
        self.strategy = "sherpa_vits"
        self.sample_rate = sample_rate or 22050

    def _resolve_piper_model_dir(self, model_path: Optional[str]) -> str:
        if model_path:
            p = Path(model_path).expanduser().resolve()
            if not p.exists():
                raise TTSModelLoadError(f"[FAIL-FAST] [ERROR: AMEVA-TTS-E001] Explicit model path not found: '{model_path}'")
            return str(p)

        all_candidates: List[Path] = []
        for d in self.get_model_search_dirs():
            all_candidates.append(d)
            if d.is_dir():
                for sub in d.iterdir():
                    if sub.is_dir():
                        all_candidates.append(sub)

        preferred = "amy-medium" if self.model_tier == "medium" else "lessac-high"
        for d in all_candidates:
            if (d / "config.json").exists() and (d / "decoder.ncnn.bin").exists():
                if preferred in d.name.lower():
                    return str(d)

        for d in all_candidates:
            if (d / "config.json").exists() and (d / "decoder.ncnn.bin").exists():
                return str(d)

        raise TTSModelLoadError(
            f"[FAIL-FAST] [ERROR: AMEVA-TTS-E001] No valid Piper VITS NCNN model found for tier '{self.model_tier}'.\n"
            f"Search directories: {[str(d) for d in all_candidates]}"
        )

    def _init_melo(self, model_path: Optional[str], sample_rate: Optional[int]) -> None:
        self.sample_rate = sample_rate or 44100
        self.strategy = "melo_mnn"
        self.backend = "VULKAN_GPU_MNN_MELO"

        search_dirs: List[Path] = []
        if model_path:
            p = Path(model_path).expanduser().resolve()
            search_dirs.append(p if p.is_dir() else p.parent)
        search_dirs.extend(self.get_model_search_dirs())

        # Find melo.mnn first
        mnn_file = None
        for d in search_dirs:
            candidate = d / "melo.mnn"
            if candidate.exists():
                mnn_file = candidate
                break

        if not mnn_file:
            raise TTSModelLoadError(
                f"[FAIL-FAST] [ERROR: AMEVA-TTS-E001] MeloTTS Vulkan 'melo.mnn' model weights not found.\n"
                f"Searched paths: {[str(d) for d in search_dirs]}"
            )

        self.model_file = str(mnn_file)
        self.model_name = mnn_file.parent.name
        self.binary = self._find_binary("melo-mnn-cli")

        # Find lexicon & tokens
        tokens_file = None
        lexicon_file = None
        for d in search_dirs:
            if (d / "tokens.txt").exists() and not tokens_file:
                tokens_file = str(d / "tokens.txt")
            if (d / "lexicon.txt").exists() and not lexicon_file:
                lexicon_file = str(d / "lexicon.txt")

        self._tokenizer = MeloTokenizer(tokens_path=tokens_file, lexicon_path=lexicon_file)

    def synthesize(
        self,
        text: str,
        output: Optional[str] = None,
        speed: float = 1.0,
        preset: Optional[str] = None,
    ) -> VulkanResult:
        if self._is_closed:
            raise TTSInferenceError("Cannot synthesize: Engine session is closed.")

        clean_text = text.strip() if text else ""
        if not clean_text:
            raise TTSInferenceError("Cannot synthesize empty text.")

        t0 = time.perf_counter()

        if self.strategy == "sherpa_vits":
            return self._synthesize_piper_vits(clean_text, output, speed, t0)
        elif self.strategy == "melo_mnn":
            return self._synthesize_melo_mnn(clean_text, output, speed, t0)

        raise TTSInferenceError(f"[FAIL-FAST] Unknown engine strategy: '{self.strategy}'")

    def _synthesize_piper_vits(
        self,
        clean_text: str,
        output: Optional[str],
        speed: float,
        t0: float,
    ) -> VulkanResult:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            temp_wav = tmp_file.name

        try:
            cmd = [
                self.binary,
                f"--vits-model-dir={self.model_dir}",
                "--use-vulkan-compute=1",
                f"--num-threads={self.threads}",
                f"--output-filename={temp_wav}",
                clean_text,
            ]
            env = get_clean_execution_env()

            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                timeout=120,
            )

            if proc.returncode != 0:
                raise TTSInferenceError(
                    f"[FAIL-FAST] [ERROR: AMEVA-TTS-E002] Vulkan GPU synthesis failed with exit code {proc.returncode}.\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"Stderr: {proc.stderr}"
                )

            audio_buf = AudioBuffer.from_wav_file(temp_wav)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            dur_sec = audio_buf.duration_seconds
            rtf = (elapsed_ms / 1000.0) / dur_sec if dur_sec > 0 else 0.0

            if output:
                out_path = Path(output).expanduser().resolve()
                out_path.parent.mkdir(parents=True, exist_ok=True)
                audio_buf.save(str(out_path))

            return VulkanResult(
                text=clean_text,
                audio_buffer=audio_buf,
                sample_rate=audio_buf.sample_rate,
                duration_sec=dur_sec,
                elapsed_ms=elapsed_ms,
                rtf=rtf,
                model_name=self.model_name,
                backend=self.backend,
                gpu_device=self.gpu_device,
                strategy=self.strategy,
            )
        finally:
            if os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except Exception:
                    pass

    def _synthesize_melo_mnn(
        self,
        clean_text: str,
        output: Optional[str],
        speed: float,
        t0: float,
    ) -> VulkanResult:
        inputs = self._tokenizer.build_inputs(clean_text, speed=speed)
        tokens = inputs["x"][0].tolist()
        tones = inputs["tones"][0].tolist()
        sid = int(inputs["sid"][0])
        length_scale = float(inputs["length_scale"][0])
        noise_scale = float(inputs["noise_scale"][0])
        noise_scale_w = float(inputs["noise_scale_w"][0])

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as f_in:
            f_in.write(",".join(str(x) for x in tokens) + "\n")
            f_in.write(",".join(str(x) for x in tones) + "\n")
            f_in.write(f"{sid},{length_scale},{noise_scale},{noise_scale_w}\n")
            temp_params = f_in.name

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f_out:
            temp_bin = f_out.name

        try:
            cmd = [self.binary, self.model_file, temp_params, temp_bin]
            env = get_clean_execution_env()

            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                timeout=120,
            )

            if proc.returncode != 0:
                raise TTSInferenceError(
                    f"[FAIL-FAST] [ERROR: AMEVA-TTS-E002] MeloTTS Vulkan native execution failed (code: {proc.returncode}).\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"Stdout: {proc.stdout}\n"
                    f"Stderr: {proc.stderr}"
                )

            # Read raw float32 samples
            with open(temp_bin, "rb") as f:
                raw_bytes = f.read()

            if not raw_bytes:
                raise TTSInferenceError(
                    "[FAIL-FAST] [ERROR: AMEVA-TTS-E003] MeloTTS Vulkan output binary is empty."
                )

            samples = np.frombuffer(raw_bytes, dtype=np.float32)
            audio_buf = AudioBuffer(samples, sample_rate=self.sample_rate)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            dur_sec = audio_buf.duration_seconds
            rtf = (elapsed_ms / 1000.0) / dur_sec if dur_sec > 0 else 0.0

            if output:
                out_path = Path(output).expanduser().resolve()
                out_path.parent.mkdir(parents=True, exist_ok=True)
                audio_buf.save(str(out_path))

            return VulkanResult(
                text=clean_text,
                audio_buffer=audio_buf,
                sample_rate=self.sample_rate,
                duration_sec=dur_sec,
                elapsed_ms=elapsed_ms,
                rtf=rtf,
                model_name=self.model_name,
                backend=self.backend,
                gpu_device=self.gpu_device,
                strategy=self.strategy,
            )
        finally:
            for p in (temp_params, temp_bin):
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

    def close(self):
        self._is_closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
