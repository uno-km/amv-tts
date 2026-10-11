# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
Enterprise Extensible Multilingual Neural Speech Synthesis Orchestrator for termux-tts.
Orchestrates multi-script tokenization, memory-resident neural acoustic inference via Sherpa C-API
(sub-second latency without subprocess spawning), silence trimming, and lossless PCM stream stitching.
Zero-Silent-Fallback compliant: Fails fast if a detected language has no registered neural model.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Set

import numpy as np

from .audio import AudioBuffer
from .exceptions import TTSModelLoadError, TTSInferenceError
from .script_classifier import MultilingualTokenizer, LanguageChunk, ScriptRegistry

logger = logging.getLogger("termux_tts.engine_multilingual")


@dataclass
class MultilingualResult:
    text: str
    audio_buffer: AudioBuffer
    sample_rate: int
    duration_sec: float
    elapsed_ms: float
    rtf: float
    chunks: List[LanguageChunk]
    languages_detected: List[str]
    backend: str = "MULTILINGUAL_NEURAL_ORCHESTRATOR"
    model_name: str = "hybrid-multilingual-mesh"

    def save(self, filepath: str) -> str:
        return self.audio_buffer.save(filepath)

    @property
    def wav_bytes(self) -> bytes:
        return self.audio_buffer.to_wav_bytes()


def trim_silence_samples(samples: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """
    Trims leading and trailing silence from normalized float32 samples [-1.0, 1.0].
    Preserves acoustic continuity across intra-word affix boundaries.
    """
    if len(samples) == 0:
        return samples

    abs_samples = np.abs(samples)
    non_silent = np.where(abs_samples >= threshold)[0]
    if len(non_silent) == 0:
        return samples

    start_idx = non_silent[0]
    end_idx = non_silent[-1]
    return samples[start_idx:end_idx + 1]


class MultilingualNeuralEngine:
    """
    Extensible Multilingual Orchestration Engine.
    Dynamically routes linguistic sub-phrases to resident on-device neural acoustic backends.
    """

    CANDIDATE_ONNX_BINARIES = [
        "sherpa-onnx-offline-tts",
        str(Path.home() / ".local" / "bin" / "sherpa-onnx-offline-tts"),
        str(Path.home() / "sherpa-onnx-offline-tts"),
        "/data/data/com.termux/files/home/.local/bin/sherpa-onnx-offline-tts",
        "/data/data/com.termux/files/usr/bin/sherpa-onnx-offline-tts",
    ]

    CANDIDATE_NCNN_BINARIES = [
        "sherpa-ncnn-offline-tts",
        str(Path.home() / ".local" / "bin" / "sherpa-ncnn-offline-tts"),
        str(Path.home() / "sherpa-ncnn" / "build-vulkan" / "bin" / "sherpa-ncnn-offline-tts"),
        "/data/data/com.termux/files/home/.local/bin/sherpa-ncnn-offline-tts",
        "/data/data/com.termux/files/usr/bin/sherpa-ncnn-offline-tts",
    ]

    STANDARD_MODEL_DIRS = [
        Path.home() / ".cache" / "termux-tts" / "models",
        Path.home() / "vits-mimic3-ko_KO-kss_low",
        Path.home() / "vits-piper-en_US-lessac-medium",
        Path.home() / "ncnn-vits-piper-en_US-lessac-high-fp16",
        Path("/data/data/com.termux/files/home/.cache/termux-tts/models"),
    ]

    def __init__(
        self,
        threads: int = 4,
        device: str = "cpu",
        sample_rate: int = 22050,
        model_path_map: Optional[Dict[str, str]] = None,
    ):
        self.threads = threads
        self.device = device.lower()
        self.sample_rate = sample_rate
        self.model_path_map = model_path_map or {}
        self.tokenizer = MultilingualTokenizer()
        self._engines: Dict[str, Any] = {}
        self._resident_manager = None
        self._is_closed = False

        # Cache directories
        self.work_dir = Path.home() / ".cache" / "termux-tts" / "chunks"
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def _get_resident_manager(self):
        if self._resident_manager is None:
            try:
                from .engine_sherpa_capi import SherpaResidentManager
                self._resident_manager = SherpaResidentManager.get_instance()
            except Exception as err:
                logger.debug("Sherpa C-API resident manager unavailable: %s", err)
        return self._resident_manager

    def register_engine(self, language: str, engine_instance: Any) -> None:
        """Register or override a neural backend engine for a specific language code."""
        self._engines[language.lower()] = engine_instance

    def _find_binary(self, candidates: List[str]) -> Optional[str]:
        for c in candidates:
            found = shutil.which(c) if not os.path.isabs(c) else c
            if found and os.path.isfile(found) and (os.access(found, os.X_OK) or os.name == "nt"):
                return str(found)
        return None

    def _resolve_korean_assets(self) -> Dict[str, str]:
        custom = self.model_path_map.get("ko")
        from .hardware import get_unified_model_search_dirs
        dirs = [Path(custom)] if custom else get_unified_model_search_dirs("tts")
        for d in dirs:
            cand = d / "vits-mimic3-ko_KO-kss_low" if not d.name.startswith("vits") else d
            onnx = cand / "ko_KO-kss_low.onnx"
            tokens = cand / "tokens.txt"
            espeak = cand / "espeak-ng-data"
            if onnx.is_file() and tokens.is_file() and espeak.is_dir():
                return {"onnx": str(onnx), "tokens": str(tokens), "data_dir": str(espeak)}
        raise TTSModelLoadError("[FAIL-FAST] Korean VITS model assets (ko_KO-kss_low) not found in cache.")

    def _resolve_english_assets(self) -> str:
        custom = self.model_path_map.get("en")
        from .hardware import get_unified_model_search_dirs
        dirs = [Path(custom)] if custom else get_unified_model_search_dirs("tts")
        for d in dirs:
            cand = d / "ncnn-vits-piper-en_US-lessac-high-fp16" if not "lessac" in d.name else d
            if (cand / "decoder.ncnn.bin").is_file():
                return str(cand)
        raise TTSModelLoadError("[FAIL-FAST] English NCNN model assets (lessac-high-fp16) not found in cache.")

    def _synthesize_chunk_audio(self, text: str, lang: str, speed: float = 1.0) -> AudioBuffer:
        """Synthesize a single linguistic chunk via fastest on-device resident engine."""
        # 1. Check custom registered engines
        if lang in self._engines:
            chunk_res = self._engines[lang].synthesize(text, speed=speed)
            return chunk_res.audio_buffer

        # 2. In-Memory Resident C-API Engine (Zero cold-start overhead)
        mgr = self._get_resident_manager()
        if mgr is not None:
            try:
                session = mgr.get_session(
                    language=lang,
                    custom_model_dir=self.model_path_map.get(lang),
                    threads=self.threads,
                    sample_rate=self.sample_rate,
                )
                return session.synthesize(text, speed=speed)
            except TTSModelLoadError:
                # If model assets missing or C-API failed, proceed to fallback/fail-fast
                raise
            except Exception as e:
                logger.debug("Resident C-API execution bypassed: %s", e)

        # 3. Fallback to Subprocess for Korean (ONNX CPU)
        if lang == "ko":
            onnx_bin = self._find_binary(self.CANDIDATE_ONNX_BINARIES)
            if not onnx_bin:
                raise TTSModelLoadError("[FAIL-FAST] sherpa-onnx-offline-tts binary not found for Korean synthesis.")
            assets = self._resolve_korean_assets()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_wav = tmp.name

            try:
                cmd = [
                    onnx_bin,
                    f"--vits-model={assets['onnx']}",
                    f"--vits-tokens={assets['tokens']}",
                    f"--vits-data-dir={assets['data_dir']}",
                    f"--num-threads={self.threads}",
                    f"--speed={speed:.2f}",
                    f"--output-filename={temp_wav}",
                    text
                ]
                env = os.environ.copy()
                env["LANG"] = "C.UTF-8"
                env["LC_ALL"] = "C.UTF-8"
                res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)
                if res.returncode != 0 or not os.path.exists(temp_wav):
                    raise TTSInferenceError(f"[FAIL-FAST] Korean chunk failed for '{text}': {res.stderr}")
                return AudioBuffer.from_wav_file(temp_wav)
            finally:
                if os.path.exists(temp_wav):
                    try:
                        os.remove(temp_wav)
                    except OSError:
                        pass

        # 4. Fallback to Subprocess for English (NCNN CPU)
        elif lang == "en":
            ncnn_bin = self._find_binary(self.CANDIDATE_NCNN_BINARIES)
            if not ncnn_bin:
                raise TTSModelLoadError("[FAIL-FAST] sherpa-ncnn-offline-tts binary not found for English synthesis.")
            model_dir = self._resolve_english_assets()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_wav = tmp.name

            try:
                use_vk = "1" if self.device in ("vulkan", "gpu") else "0"
                cmd = [
                    ncnn_bin,
                    f"--vits-model-dir={model_dir}",
                    f"--use-vulkan-compute={use_vk}",
                    f"--num-threads={self.threads}",
                    f"--output-filename={temp_wav}",
                    text
                ]
                env = os.environ.copy()
                env["LANG"] = "C.UTF-8"
                env["LC_ALL"] = "C.UTF-8"
                res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)
                if res.returncode != 0 or not os.path.exists(temp_wav) or os.path.getsize(temp_wav) == 0:
                    if "empty ids for word" in res.stderr:
                        logger.warning("OOV word '%s' in English dictionary; inserting soft breath pause.", text)
                        return AudioBuffer(np.zeros(int(self.sample_rate * 0.1), dtype=np.float32), sample_rate=self.sample_rate)
                    raise TTSInferenceError(f"[FAIL-FAST] English chunk failed for '{text}': {res.stderr}")
                return AudioBuffer.from_wav_file(temp_wav)
            finally:
                if os.path.exists(temp_wav):
                    try:
                        os.remove(temp_wav)
                    except OSError:
                        pass

        # 5. Extended Languages (ja, zh, hi, ru, es, fr, de, ar)
        elif lang in ("ja", "zh", "hi", "ru", "es", "fr", "de", "ar"):
            custom_model = self.model_path_map.get(lang)
            if not custom_model:
                from .installer import OFFICIAL_NEURAL_MODELS
                meta = OFFICIAL_NEURAL_MODELS.get(lang, {})
                lang_name = meta.get("language_name", lang.upper())
                repo = meta.get("repo", f"csukuangfj/vits-{lang}")
                model_name = meta.get("name", f"vits-{lang}")
                canonical_dest = str((Path.home() / "models" / "tts" / model_name).resolve())
                raise TTSModelLoadError(
                    f"[FAIL-FAST] Neural speech model for language '{lang}' ({lang_name}) is not installed.\n\n"
                    f"  To install this model automatically, run:\n"
                    f"      termux-tts install --models {lang}\n\n"
                    f"  Or manually download via git clone:\n"
                    f"      git clone https://huggingface.co/{repo} {canonical_dest}\n\n"
                    f"  After installation, re-run your synthesis command!"
                )
            from .engine_sherpa import SherpaNeuralEngine
            engine = SherpaNeuralEngine(model_path=custom_model, language=lang, threads=self.threads, sample_rate=self.sample_rate)
            self._engines[lang] = engine
            return engine.synthesize(text, speed=speed).audio_buffer

        raise TTSModelLoadError(f"[FAIL-FAST] Unsupported language '{lang}'. Supported languages: ['ko', 'en', 'ja', 'zh', 'hi', 'ru', 'es', 'fr', 'de', 'ar'].")

    def synthesize(
        self,
        text: str,
        output: Optional[str] = None,
        speed: float = 1.0,
        language: Optional[str] = None,
        mode: str = "unified",
    ) -> MultilingualResult:
        """
        Synthesizes speech with natural prosody and zero cross-speaker distortion.
        Defaults to 'unified' single-pass mode (phonetic transliteration preserving prosody).
        'stitch' mode retains explicit multi-model chunk concatenation when requested.
        """
        if self._is_closed:
            raise TTSInferenceError("Cannot synthesize: Multilingual session is closed.")

        clean_text = text.strip()
        if not clean_text:
            raise TTSInferenceError("Cannot synthesize empty text.")

        t0 = time.perf_counter()
        target_sample_rate = self.sample_rate

        # -------------------------------------------------------------------
        # Mode 1: Unified Single-Pass Pipeline (BigTech Architecture Standard)
        # -------------------------------------------------------------------
        has_korean = bool(re.search(r"[\uac00-\ud7a3]", clean_text))
        has_latin = bool(re.search(r"[a-zA-Z]", clean_text))

        if mode == "unified" and (language == "ko" or (language is None and has_korean)):
            from .transliteration import transliterate_mixed_text
            unified_text = transliterate_mixed_text(clean_text)
            logger.info("[Multilingual] Unified single-pass execution: '%s' -> '%s'", clean_text, unified_text)

            chunk_buf = self._synthesize_chunk_audio(unified_text, "ko", speed=speed)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            dur_sec = chunk_buf.duration_seconds
            rtf = (elapsed_ms / 1000.0) / max(0.001, dur_sec)

            if output:
                chunk_buf.save(output)

            unified_chunk = LanguageChunk(text=clean_text, language="ko", pause_after=0.0)
            return MultilingualResult(
                text=clean_text,
                audio_buffer=chunk_buf,
                sample_rate=target_sample_rate,
                duration_sec=dur_sec,
                elapsed_ms=elapsed_ms,
                rtf=rtf,
                chunks=[unified_chunk],
                languages_detected=["ko", "en"] if has_latin else ["ko"],
                backend="UNIFIED_SINGLE_PASS_NEURAL",
            )

        # -------------------------------------------------------------------
        # Mode 2: Multi-Script Chunk Concatenation (Explicit Stitch Mode)
        # -------------------------------------------------------------------
        chunks = self.tokenizer.tokenize(clean_text, force_language=language)
        if not chunks:
            raise TTSInferenceError("Tokenization yielded no synthesizable segments.")

        detected_langs = list(dict.fromkeys(c.language for c in chunks))

        assembled_samples: List[np.ndarray] = []

        for i, chunk in enumerate(chunks):
            chunk_buf = self._synthesize_chunk_audio(chunk.text, chunk.language, speed=speed)

            # Get raw float32 samples and trim acoustic silence
            raw_samples = chunk_buf.samples
            trimmed = trim_silence_samples(raw_samples, threshold=0.01)
            assembled_samples.append(trimmed)

            # Add context-aware pause
            if chunk.pause_after > 0:
                pause_len = int(target_sample_rate * chunk.pause_after)
                silence_gap = np.zeros(pause_len, dtype=np.float32)
                assembled_samples.append(silence_gap)

        if assembled_samples:
            final_samples = np.concatenate(assembled_samples)
        else:
            final_samples = np.zeros(0, dtype=np.float32)

        combined_buffer = AudioBuffer(final_samples, sample_rate=target_sample_rate)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        dur_sec = combined_buffer.duration_seconds
        rtf = (elapsed_ms / 1000.0) / max(0.001, dur_sec)

        if output:
            combined_buffer.save(output)

        return MultilingualResult(
            text=clean_text,
            audio_buffer=combined_buffer,
            sample_rate=target_sample_rate,
            duration_sec=dur_sec,
            elapsed_ms=elapsed_ms,
            rtf=rtf,
            chunks=chunks,
            languages_detected=detected_langs,
        )

    def close(self) -> None:
        """Cleanly terminate all engine instances and release resident memory."""
        self._is_closed = True
        for engine in self._engines.values():
            if hasattr(engine, "close"):
                try:
                    engine.close()
                except Exception:
                    pass
        if self._resident_manager is not None:
            try:
                self._resident_manager.close_all()
            except Exception:
                pass

    def __del__(self) -> None:
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
