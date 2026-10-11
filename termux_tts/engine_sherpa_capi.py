# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
Sherpa-ONNX C-API In-Memory Resident Neural Synthesis Engine and Session Manager.
Enables sub-second intra-sentential multilingual speech synthesis by keeping neural acoustic models
resident in RAM via libsherpa-onnx-c-api.so, eliminating cold subprocess process startup and model disk re-parsing.

Governance & Compliance:
- AOSF-ENG-STD-2026 / Zero-Deception: Zero silent fallbacks, zero synthetic stubs.
- Clean Lifecycle: Context manager, explicit close(), __del__, and global atexit teardown.
"""
from __future__ import annotations

import atexit
import ctypes
import logging
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Any

import numpy as np

from .audio import AudioBuffer
from .exceptions import TTSModelLoadError, TTSInferenceError

logger = logging.getLogger("termux_tts.engine_sherpa_capi")

# ---------------------------------------------------------------------------
# C-API Type Definitions matching sherpa-onnx/c-api/c-api.h
# ---------------------------------------------------------------------------

class SherpaOnnxOfflineTtsVitsModelConfig(ctypes.Structure):
    _fields_ = [
        ("model", ctypes.c_char_p),
        ("lexicon", ctypes.c_char_p),
        ("tokens", ctypes.c_char_p),
        ("data_dir", ctypes.c_char_p),
        ("noise_scale", ctypes.c_float),
        ("noise_scale_w", ctypes.c_float),
        ("length_scale", ctypes.c_float),
        ("dict_dir", ctypes.c_char_p),
    ]


class SherpaOnnxOfflineTtsMatchaModelConfig(ctypes.Structure):
    _fields_ = [
        ("acoustic_model", ctypes.c_char_p),
        ("vocoder", ctypes.c_char_p),
        ("lexicon", ctypes.c_char_p),
        ("tokens", ctypes.c_char_p),
        ("data_dir", ctypes.c_char_p),
        ("noise_scale", ctypes.c_float),
        ("length_scale", ctypes.c_float),
        ("dict_dir", ctypes.c_char_p),
    ]


class SherpaOnnxOfflineTtsKokoroModelConfig(ctypes.Structure):
    _fields_ = [
        ("model", ctypes.c_char_p),
        ("voices", ctypes.c_char_p),
        ("tokens", ctypes.c_char_p),
        ("data_dir", ctypes.c_char_p),
        ("length_scale", ctypes.c_float),
        ("dict_dir", ctypes.c_char_p),
        ("lexicon", ctypes.c_char_p),
        ("lang", ctypes.c_char_p),
    ]


class SherpaOnnxOfflineTtsKittenModelConfig(ctypes.Structure):
    _fields_ = [
        ("model", ctypes.c_char_p),
        ("voices", ctypes.c_char_p),
        ("tokens", ctypes.c_char_p),
        ("data_dir", ctypes.c_char_p),
        ("length_scale", ctypes.c_float),
    ]


class SherpaOnnxOfflineTtsZipvoiceModelConfig(ctypes.Structure):
    _fields_ = [
        ("tokens", ctypes.c_char_p),
        ("encoder", ctypes.c_char_p),
        ("decoder", ctypes.c_char_p),
        ("vocoder", ctypes.c_char_p),
        ("data_dir", ctypes.c_char_p),
        ("lexicon", ctypes.c_char_p),
        ("feat_scale", ctypes.c_float),
        ("t_shift", ctypes.c_float),
        ("target_rms", ctypes.c_float),
        ("guidance_scale", ctypes.c_float),
    ]


class SherpaOnnxOfflineTtsPocketModelConfig(ctypes.Structure):
    _fields_ = [
        ("lm_flow", ctypes.c_char_p),
        ("lm_main", ctypes.c_char_p),
        ("encoder", ctypes.c_char_p),
        ("decoder", ctypes.c_char_p),
        ("text_conditioner", ctypes.c_char_p),
        ("vocab_json", ctypes.c_char_p),
        ("token_scores_json", ctypes.c_char_p),
        ("voice_embedding_cache_capacity", ctypes.c_int32),
    ]


class SherpaOnnxOfflineTtsSupertonicModelConfig(ctypes.Structure):
    _fields_ = [
        ("duration_predictor", ctypes.c_char_p),
        ("text_encoder", ctypes.c_char_p),
        ("vector_estimator", ctypes.c_char_p),
        ("vocoder", ctypes.c_char_p),
        ("tts_json", ctypes.c_char_p),
        ("unicode_indexer", ctypes.c_char_p),
        ("voice_style", ctypes.c_char_p),
    ]


class SherpaOnnxOfflineTtsModelConfig(ctypes.Structure):
    _fields_ = [
        ("vits", SherpaOnnxOfflineTtsVitsModelConfig),
        ("num_threads", ctypes.c_int32),
        ("debug", ctypes.c_int32),
        ("provider", ctypes.c_char_p),
        ("matcha", SherpaOnnxOfflineTtsMatchaModelConfig),
        ("kokoro", SherpaOnnxOfflineTtsKokoroModelConfig),
        ("kitten", SherpaOnnxOfflineTtsKittenModelConfig),
        ("zipvoice", SherpaOnnxOfflineTtsZipvoiceModelConfig),
        ("pocket", SherpaOnnxOfflineTtsPocketModelConfig),
        ("supertonic", SherpaOnnxOfflineTtsSupertonicModelConfig),
    ]


class SherpaOnnxOfflineTtsConfig(ctypes.Structure):
    _fields_ = [
        ("model", SherpaOnnxOfflineTtsModelConfig),
        ("rule_fsts", ctypes.c_char_p),
        ("max_num_sentences", ctypes.c_int32),
        ("rule_fars", ctypes.c_char_p),
        ("silence_scale", ctypes.c_float),
    ]


class SherpaOnnxGeneratedAudio(ctypes.Structure):
    _fields_ = [
        ("samples", ctypes.POINTER(ctypes.c_float)),
        ("n", ctypes.c_int32),
        ("sample_rate", ctypes.c_int32),
    ]


# ---------------------------------------------------------------------------
# C-API Shared Library Loader
# ---------------------------------------------------------------------------

_CACHED_CDLL = None


def find_sherpa_capi_library() -> Optional[str]:
    """Search standard library search locations for libsherpa-onnx-c-api."""
    override = os.environ.get("SHERPA_ONNX_C_API_LIB")
    if override and os.path.isfile(override):
        return override

    candidates = [
        "libsherpa-onnx-c-api.so",
        "/data/data/com.termux/files/usr/lib/libsherpa-onnx-c-api.so",
        str(Path.home() / ".local" / "lib" / "libsherpa-onnx-c-api.so"),
        str(Path.home() / "sherpa_termux" / "sherpa-onnx-v1.13.8-android-aarch64-termux-shared" / "lib" / "libsherpa-onnx-c-api.so"),
        "/usr/local/lib/libsherpa-onnx-c-api.so",
        "/usr/lib/libsherpa-onnx-c-api.so",
    ]

    for cand in candidates:
        if os.path.isabs(cand) and os.path.isfile(cand):
            return cand

    # Try system loader lookup
    from ctypes.util import find_library
    found = find_library("sherpa-onnx-c-api")
    if found:
        return found

    return None


def get_sherpa_capi_cdll():
    """Load and return the singleton libsherpa-onnx-c-api CDLL instance or raise TTSModelLoadError."""
    global _CACHED_CDLL
    if _CACHED_CDLL is not None:
        return _CACHED_CDLL

    lib_path = find_sherpa_capi_library()
    if not lib_path:
        raise TTSModelLoadError(
            "[FAIL-FAST] libsherpa-onnx-c-api shared library not found.\n"
            "Please ensure libsherpa-onnx-c-api.so is present in /data/data/com.termux/files/usr/lib/ or set SHERPA_ONNX_C_API_LIB."
        )

    try:
        cdll = ctypes.CDLL(lib_path)
    except Exception as err:
        raise TTSModelLoadError(
            f"[FAIL-FAST] Failed to load libsherpa-onnx-c-api shared library from '{lib_path}': {err}"
        ) from err

    # Setup function signatures
    cdll.SherpaOnnxCreateOfflineTts.argtypes = [ctypes.POINTER(SherpaOnnxOfflineTtsConfig)]
    cdll.SherpaOnnxCreateOfflineTts.restype = ctypes.c_void_p

    cdll.SherpaOnnxDestroyOfflineTts.argtypes = [ctypes.c_void_p]
    cdll.SherpaOnnxDestroyOfflineTts.restype = None

    cdll.SherpaOnnxOfflineTtsSampleRate.argtypes = [ctypes.c_void_p]
    cdll.SherpaOnnxOfflineTtsSampleRate.restype = ctypes.c_int32

    cdll.SherpaOnnxOfflineTtsGenerate.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_int32,
        ctypes.c_float,
    ]
    cdll.SherpaOnnxOfflineTtsGenerate.restype = ctypes.POINTER(SherpaOnnxGeneratedAudio)

    cdll.SherpaOnnxDestroyOfflineTtsGeneratedAudio.argtypes = [
        ctypes.POINTER(SherpaOnnxGeneratedAudio)
    ]
    cdll.SherpaOnnxDestroyOfflineTtsGeneratedAudio.restype = None

    _CACHED_CDLL = cdll
    return _CACHED_CDLL


# ---------------------------------------------------------------------------
# Resident Session Class
# ---------------------------------------------------------------------------

class SherpaCapiSession:
    """
    High-performance In-Memory Resident Sherpa-ONNX TTS Session.
    Maintains loaded acoustic neural model in memory across multiple synthesize calls.
    Guarantees strict resource deallocation upon exit/close().
    """

    def __init__(
        self,
        model_path: str,
        tokens_path: str,
        data_dir: str,
        language: str = "ko",
        threads: int = 4,
        sample_rate: int = 22050,
        provider: str = "cpu",
        model_type: str = "vits",
        voices_path: Optional[str] = None,
    ):
        self.language = language.lower()
        self.threads = threads
        self.sample_rate = sample_rate
        self.provider = provider
        self.model_path = model_path
        self.tokens_path = tokens_path
        self.data_dir = data_dir
        self.model_type = model_type.lower()
        self._is_closed = False

        self._cdll = get_sherpa_capi_cdll()

        # Build configuration struct
        cfg = SherpaOnnxOfflineTtsConfig()
        if self.model_type == "kokoro":
            cfg.model.kokoro.model = model_path.encode("utf-8")
            if voices_path:
                cfg.model.kokoro.voices = voices_path.encode("utf-8")
            cfg.model.kokoro.tokens = tokens_path.encode("utf-8")
            cfg.model.kokoro.data_dir = data_dir.encode("utf-8")
            cfg.model.kokoro.length_scale = 1.0
        elif self.model_type == "supertonic":
            cfg.model.supertonic.duration_predictor = model_path.encode("utf-8")
            cfg.model.supertonic.tts_json = tokens_path.encode("utf-8")
        else:  # vits / melo
            cfg.model.vits.model = model_path.encode("utf-8")
            cfg.model.vits.tokens = tokens_path.encode("utf-8")
            cfg.model.vits.data_dir = data_dir.encode("utf-8")
            cfg.model.vits.noise_scale = 0.667
            cfg.model.vits.noise_scale_w = 0.8
            cfg.model.vits.length_scale = 1.0

        cfg.model.num_threads = threads
        cfg.model.debug = 0
        cfg.model.provider = provider.encode("utf-8")
        cfg.max_num_sentences = 2
        cfg.silence_scale = 0.2

        logger.debug("Creating Sherpa C-API resident engine for '%s' from %s", language, model_path)
        self._handle = self._cdll.SherpaOnnxCreateOfflineTts(ctypes.byref(cfg))
        if not self._handle:
            raise TTSModelLoadError(
                f"[FAIL-FAST] SherpaOnnxCreateOfflineTts failed for language '{language}' "
                f"with model: {model_path}"
            )

        native_sr = self._cdll.SherpaOnnxOfflineTtsSampleRate(self._handle)
        if native_sr > 0:
            self.sample_rate = native_sr

    def synthesize(self, text: str, speed: float = 1.0) -> AudioBuffer:
        """Synthesize text directly into an AudioBuffer using resident in-memory model."""
        if self._is_closed or not self._handle:
            raise TTSInferenceError("Cannot synthesize: Sherpa C-API resident session is closed.")

        clean_text = text.strip()
        if not clean_text:
            return AudioBuffer(np.zeros(0, dtype=np.float32), sample_rate=self.sample_rate)

        # Execute in-memory neural inference
        audio_ptr = self._cdll.SherpaOnnxOfflineTtsGenerate(
            self._handle,
            clean_text.encode("utf-8"),
            0,
            float(speed),
        )

        if not audio_ptr:
            raise TTSInferenceError(
                f"[FAIL-FAST] Sherpa C-API generation returned NULL for '{clean_text}'."
            )

        try:
            audio_obj = audio_ptr.contents
            n_samples = audio_obj.n
            out_sr = audio_obj.sample_rate if audio_obj.sample_rate > 0 else self.sample_rate

            if n_samples <= 0:
                return AudioBuffer(np.zeros(0, dtype=np.float32), sample_rate=out_sr)

            # Copy float samples directly to NumPy array
            samples = np.ctypeslib.as_array(audio_obj.samples, shape=(n_samples,)).copy()
            return AudioBuffer(samples, sample_rate=out_sr)
        finally:
            self._cdll.SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio_ptr)

    def close(self) -> None:
        """Destroy native C++ TTS handle and release model memory."""
        if not self._is_closed and getattr(self, "_handle", None):
            logger.debug("Destroying Sherpa C-API engine handle for language '%s'", self.language)
            try:
                self._cdll.SherpaOnnxDestroyOfflineTts(self._handle)
            except Exception as err:
                logger.warning("Exception during Sherpa C-API teardown: %s", err)
            finally:
                self._handle = None
                self._is_closed = True

    def __del__(self) -> None:
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ---------------------------------------------------------------------------
# Resident Session Manager & Lifecycle Guard
# ---------------------------------------------------------------------------

class SherpaResidentManager:
    """
    Global resident model session manager.
    Coordinates memory-resident models across languages and ensures 100% clean shutdown upon process exit.
    """

    _INSTANCE: Optional[SherpaResidentManager] = None

    STANDARD_MODEL_DIRS = [
        Path.home() / ".cache" / "termux-tts" / "models",
        Path.home() / "vits-mimic3-ko_KO-kss_low",
        Path.home() / "vits-piper-en_US-lessac-medium",
        Path("/data/data/com.termux/files/home/.cache/termux-tts/models"),
    ]

    def __init__(self):
        self._sessions: Dict[str, SherpaCapiSession] = {}
        atexit.register(self.close_all)

    @classmethod
    def get_instance(cls) -> SherpaResidentManager:
        if cls._INSTANCE is None:
            cls._INSTANCE = SherpaResidentManager()
        return cls._INSTANCE

    def resolve_model_assets(self, language: str, custom_dir: Optional[str] = None) -> Dict[str, str]:
        """Locate ONNX, tokens.txt, and espeak-ng-data for a given language."""
        lang = language.lower()
        search_dirs: List[Path] = []
        if custom_dir:
            search_dirs.append(Path(custom_dir).expanduser().resolve())
        from .hardware import get_unified_model_search_dirs
        search_dirs.extend(get_unified_model_search_dirs("tts"))

        # Expected directory sub-names
        expected_names = {
            "ko": ["vits-mimic3-ko_KO-kss_low"],
            "en": ["vits-piper-en_US-lessac-medium", "vits-piper-en_US-lessac-high"],
            "ja": ["vits-piper-ja_JP-hina-medium", "vits-piper-ja_JP-kokoro", "vits-ja_JP"],
            "zh": ["vits-zh-aishell3", "vits-zh_CN", "vits-piper-zh_CN"],
            "hi": ["vits-piper-hi_IN-swara-medium", "vits-piper-hi_IN"],
            "ru": ["vits-piper-ru_RU-dmitri-medium", "vits-piper-ru_RU"],
            "es": ["vits-piper-es_ES-davefx-medium"],
            "fr": ["vits-piper-fr_FR-siwis-medium"],
            "de": ["vits-piper-de_DE-thorsten-medium"],
            "kokoro": ["kokoro-en-v0_19", "kokoro-82m", "kokoro-v0_19", "kokoro"],
            "melo": ["vits-melo-tts-zh_en", "melo-tts", "melotts"],
            "supertonic": ["supertonic-tts", "supertonic-3-tts", "supertonic"],
        }.get(lang, [f"vits-{lang}", f"vits-piper-{lang}"])

        for sdir in search_dirs:
            if not sdir.exists():
                continue

            candidates = [sdir]
            for exp in expected_names:
                candidates.append(sdir / exp)

            # Also search any subdirectory matching language name or code
            try:
                for sub in sdir.iterdir():
                    if sub.is_dir() and (lang in sub.name.lower() or any(exp.lower() in sub.name.lower() for exp in expected_names)):
                        if sub not in candidates:
                            candidates.append(sub)
            except OSError:
                pass

            for cand in candidates:
                if not cand.is_dir():
                    continue

                onnx_files = list(cand.glob("*.onnx"))
                if not onnx_files:
                    continue

                onnx_model = str(onnx_files[0])
                tokens_file = cand / "tokens.txt"
                espeak_dir = cand / "espeak-ng-data"

                if not tokens_file.exists():
                    tokens_file = cand.parent / "tokens.txt"
                if not espeak_dir.exists():
                    espeak_dir = cand.parent / "espeak-ng-data"

                if tokens_file.exists() and espeak_dir.exists():
                    return {
                        "model": str(onnx_model),
                        "tokens": str(tokens_file),
                        "data_dir": str(espeak_dir),
                    }

        from .installer import OFFICIAL_NEURAL_MODELS
        meta = OFFICIAL_NEURAL_MODELS.get(lang, {})
        lang_name = meta.get("language_name", lang.upper())
        repo = meta.get("repo", f"csukuangfj/vits-{lang}")
        model_name = meta.get("name", expected_names[0])
        canonical_dest = str((Path.home() / "models" / "tts" / model_name).resolve())

        raise TTSModelLoadError(
            f"[FAIL-FAST] Neural speech model for language '{lang}' ({lang_name}) is not installed.\n\n"
            f"  To install this model automatically, run:\n"
            f"      termux-tts install --models {lang}\n\n"
            f"  Or manually download via git clone:\n"
            f"      git clone https://huggingface.co/{repo} {canonical_dest}\n\n"
            f"  After installation, re-run your synthesis command!"
        )

    def get_session(
        self,
        language: str,
        custom_model_dir: Optional[str] = None,
        threads: int = 4,
        sample_rate: int = 22050,
    ) -> SherpaCapiSession:
        """Get or lazily instantiate an in-memory resident session for the requested language."""
        lang = language.lower()
        if lang in self._sessions and not self._sessions[lang]._is_closed:
            return self._sessions[lang]

        assets = self.resolve_model_assets(lang, custom_dir=custom_model_dir)
        session = SherpaCapiSession(
            model_path=assets["model"],
            tokens_path=assets["tokens"],
            data_dir=assets["data_dir"],
            language=lang,
            threads=threads,
            sample_rate=sample_rate,
        )
        self._sessions[lang] = session
        return session

    def close_all(self) -> None:
        """Cleanly terminate all resident sessions."""
        for lang, session in list(self._sessions.items()):
            try:
                session.close()
            except Exception as err:
                logger.warning("Error closing session '%s': %s", lang, err)
        self._sessions.clear()
