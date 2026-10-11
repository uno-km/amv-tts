# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
Unified 4-Tier Multi-Backend TTS Gateway for termux-tts:
- Tier 1: Pure Parametric DSP Formant Synthesizer Engine (ParametricDSPEngine - 0MB Zero-Dependency)
- Tier 2: Android System Native Samsung/Google Voice Engine Bridge (NativeAndroidEngine - 0MB OS IPC)
- Tier 3: Authentic C++ Isolated Sherpa-ONNX VITS Neural Vocoder (SherpaNeuralEngine - Deep Learning)
- Tier 4: Pure On-Device Expressive Emotional Synthesizer (ExpressiveEngine - Conversational Tags)

Strict Zero-Silent-Fallback Protocol:
- If caller explicitly requests 'neural' and assets are missing -> Raise TTSModelLoadError (FAIL-FAST)
- If caller explicitly requests 'native' and termux-api is missing -> Raise TTSInferenceError (FAIL-FAST)
- Never swallow exceptions or mask failures with robotic buzzers without explicit instruction.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Union, Dict, Any

from .exceptions import TTSInferenceError, TTSModelLoadError, VulkanInitializationError
from .engine_native import NativeAndroidEngine, NativeResult
from .engine_sherpa import SherpaNeuralEngine, SherpaResult
from .engine_vulkan import VulkanNeuralEngine, VulkanResult
from .engine_expressive import ExpressiveEngine, ExpressiveResult
from .engine_multilingual import MultilingualNeuralEngine, MultilingualResult
from .engine_sota import SOTANeuralEngine, SOTABackend, SOTAResult
from .cluster import parse_cluster_rpc_spec, verify_rpc_cluster_nodes
from .script_classifier import MultilingualTokenizer
from .hardware import (
    resolve_device_backend,
    bind_tts_hardware,
    _resolve_ameva_runtime,
    ERROR_AMEVA_TTS_E001,
)

logger = logging.getLogger("termux_tts.engine")


QUALITY_PRESETS = {
    "fast": {"sample_rate": 16000, "description": "16kHz Low Latency"},
    "balanced": {"sample_rate": 22050, "description": "22.05kHz Standard Audio"},
    "expressive": {"sample_rate": 24000, "description": "24kHz Expressive High Fidelity"},
    "ultra": {"sample_rate": 44100, "description": "44.1kHz Studio Master"},
    "cinematic": {"sample_rate": 24000, "description": "24kHz SOTA Expressive Flow Matching (CosyVoice 2 / F5-TTS)"},
    "sota": {"sample_rate": 24000, "description": "24kHz SOTA Voice Cloning & Conversational Flow"},
}


class TTSEngine:
    """Production 4-Tier Multi-Backend Gateway supporting Synth, Native, Neural, and Expressive engines."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        language: str = "ko",
        preset: str = "balanced",
        device: str = "auto",
        threads: int = 4,
        sample_rate: Optional[int] = None,
        engine_type: str = "auto",
        model_tier: Optional[str] = None,
        rpc: Optional[str] = None,
        tensor_split: Optional[str] = None,
        cluster_rpc_servers: Optional[Any] = None,
        cluster_split_mode: Optional[str] = None,
        cluster_tensor_split: Optional[str] = None,
        cluster_vram_budget: Optional[dict] = None,
    ):
        self.language = language.lower()
        self.preset = preset.lower()
        self.requested_device = device.lower()
        self.requested_engine_type = engine_type.lower()
        self.model_path = model_path
        self.threads = threads
        self.sample_rate = sample_rate
        self.model_tier = model_tier
        self.cluster_rpc_servers = parse_cluster_rpc_spec(cluster_rpc_servers or rpc)
        self.rpc = ",".join(self.cluster_rpc_servers) if self.cluster_rpc_servers else None
        self.cluster_split_mode = cluster_split_mode
        self.cluster_tensor_split = cluster_tensor_split or tensor_split
        self.tensor_split = self.cluster_tensor_split
        self.cluster_vram_budget = cluster_vram_budget
        self._is_closed = False

        # 1. Resolve effective device and engine_type via safe hardware gateway
        self.device, effective_engine = resolve_device_backend(
            self.requested_device, self.requested_engine_type
        )
        if self.requested_engine_type == "auto" and effective_engine != "auto":
            self.requested_engine_type = effective_engine

        self.backend = "auto"

        # 2. Attempt ameva-runtime hardware binding if available
        self._binding_plan = self._bind_hardware()

        self.native_engine = NativeAndroidEngine(language=language)
        self._multilingual_engine: Optional[MultilingualNeuralEngine] = None
        self.synth_engine = self._resolve_synth_engine()

    def _bind_hardware(self):
        return bind_tts_hardware(self, self.requested_device)

    def _resolve_synth_engine(self):
        t = self.requested_engine_type
        from .script_classifier import normalize_language_code
        norm_lang = normalize_language_code(self.language)

        # Tier 5: SOTA Neural Voice Cloning & Expressive Flow (CosyVoice 2 / F5-TTS / ChatTTS)
        if t in ("cosyvoice", "cosyvoice2", "f5", "f5tts", "chattts", "sota"):
            return SOTANeuralEngine(
                backend=t,
                model_path=self.model_path,
                language=self.language,
                device=self.requested_device,
                threads=self.threads,
                sample_rate=self.sample_rate,
                rpc=self.rpc,
            )

        # Extended Languages (hi, ru, ja, zh, es, fr, de, ar) route to MultilingualNeuralEngine
        if norm_lang in ("hi", "ru", "ja", "zh", "es", "fr", "de", "ar"):
            return self._get_multilingual_engine()

        # Explicit Vulkan GPU MeloTTS Tier (Plan 1 NCNN Sliced / Plan 2 MNN Vulkan)
        if t in ("melo_vulkan", "melo_ncnn", "melo_mnn") or (t == "melo" and self.requested_device in ("vulkan", "gpu")):
            try:
                return VulkanNeuralEngine(
                    model_path=self.model_path,
                    language=self.language,
                    device=self.requested_device,
                    threads=self.threads,
                    sample_rate=self.sample_rate or 44100,
                    model_tier=self.model_tier,
                    model_type=t,
                )
            except (VulkanInitializationError, TTSModelLoadError) as err:
                if self.requested_device in ("vulkan", "gpu"):
                    raise
                logger.debug("Vulkan melo engine not ready, using sherpa melo: %s", err)

        # Explicit Vulkan GPU Tier (Vulkan NCNN engine targets English Lessac)
        if (t in ("vulkan", "gpu", "ncnn") or (self.requested_device in ("vulkan", "gpu") and t in ("neural", "vits", "auto"))) and norm_lang in ("en", "auto"):
            try:
                return VulkanNeuralEngine(
                    model_path=self.model_path,
                    language=self.language,
                    device=self.requested_device,
                    threads=self.threads,
                    sample_rate=self.sample_rate or 22050,
                    model_tier=self.model_tier,
                    model_type="vits",
                )
            except (VulkanInitializationError, TTSModelLoadError) as err:
                if t in ("vulkan", "gpu", "ncnn") or self.requested_device in ("vulkan", "gpu"):
                    raise
                logger.debug("Vulkan neural engine not ready, trying fallback: %s", err)

        # Explicit Tier 3: Neural (Sherpa-ONNX CPU)
        if t in ("neural", "onnx", "sherpa", "vits"):
            return SherpaNeuralEngine(
                model_path=self.model_path,
                language=self.language,
                device=self.requested_device,
                threads=self.threads,
                sample_rate=self.sample_rate or 22050,
                model_type="vits",
                rpc=self.rpc,
                tensor_split=self.tensor_split,
                cluster_rpc_servers=self.cluster_rpc_servers,
                cluster_split_mode=self.cluster_split_mode,
                cluster_tensor_split=self.cluster_tensor_split,
                cluster_vram_budget=self.cluster_vram_budget,
            )

        # BigTech 3rd-Party Neural Speech Engines (StyleTTS2/Kokoro, MeloTTS, Supertonic)
        if t in ("kokoro", "melo", "supertonic"):
            return SherpaNeuralEngine(
                model_path=self.model_path,
                language=self.language,
                device=self.requested_device,
                threads=self.threads,
                sample_rate=self.sample_rate or 22050,
                model_type=t,
                rpc=self.rpc,
                tensor_split=self.tensor_split,
                cluster_rpc_servers=self.cluster_rpc_servers,
                cluster_split_mode=self.cluster_split_mode,
                cluster_tensor_split=self.cluster_tensor_split,
                cluster_vram_budget=self.cluster_vram_budget,
            )

        # Explicit Tier 4: Expressive (Fail-Fast)
        elif t in ("expressive", "chat", "conversational"):
            return ExpressiveEngine(
                model_path=self.model_path,
                language=self.language,
                device=self.requested_device,
                threads=self.threads,
                sample_rate=self.sample_rate or 22050,
            )

        # Explicit Multilingual / Code-Switching Tier
        elif t in ("multilingual", "codeswitch", "hybrid"):
            return self._get_multilingual_engine()

        # Explicit Tier 2: Native
        elif t == "native":
            return self.native_engine

        # Auto Mode (Zero-Silent-Fallback)
        elif t == "auto":
            # 1. Check if SherpaNeuralEngine assets exist
            try:
                return SherpaNeuralEngine(
                    model_path=self.model_path,
                    language=self.language,
                    device=self.requested_device,
                    threads=self.threads,
                    sample_rate=self.sample_rate or 22050,
                    rpc=self.rpc,
                    tensor_split=self.tensor_split,
                )
            except TTSModelLoadError as err:
                logger.info("Auto tier: Neural model assets not ready (%s). Probing native system voice...", err)

            # 2. Check if NativeAndroidEngine is available
            if self.native_engine.binary:
                return self.native_engine

            # 3. Fail-Fast: No silent robotic fallback
            raise TTSModelLoadError(
                "[FAIL-FAST] No TTS voice engine available. Neural speech model is not installed, "
                "and native Android TTS engine is unavailable.\n"
                "  To install official neural models automatically, run:\n"
                "      termux-tts install\n"
            )
        else:
            raise TTSInferenceError(
                f"[FAIL-FAST] Unknown engine_type '{self.requested_engine_type}'. "
                f"Available tiers: ['auto', 'native', 'neural', 'expressive', 'multilingual', 'kokoro', 'melo', 'supertonic']"
            )

    @property
    def active_backend(self) -> str:
        return getattr(self.synth_engine, "backend", "NATIVE_SYSTEM")

    @property
    def model_name(self) -> str:
        return getattr(self.synth_engine, "model_name", "native-system-voice")

    @property
    def binary(self) -> Optional[str]:
        return getattr(self.synth_engine, "binary", getattr(self.native_engine, "binary", None))

    def _get_multilingual_engine(self) -> MultilingualNeuralEngine:
        if self._multilingual_engine is None:
            self._multilingual_engine = MultilingualNeuralEngine(
                threads=self.threads,
                device=self.device,
                sample_rate=self.sample_rate or 22050,
            )
        return self._multilingual_engine

    def speak(self, text: str, stream: Optional[str] = None) -> NativeResult:
        """Speak text directly through physical Android speaker (Native Engine)."""
        if self._is_closed:
            raise TTSInferenceError("Cannot speak: Engine session is closed.")
        return self.native_engine.speak(text, stream=stream)

    def synthesize(
        self,
        text: str,
        output: Optional[str] = None,
        speed: float = 1.0,
        preset: Optional[str] = None,
        language: Optional[str] = None,
        mode: str = "unified",
        output_path: Optional[str] = None,
        ref_audio: Optional[Union[str, os.PathLike]] = None,
        ref_text: Optional[str] = None,
        profile: Optional[Union[str, os.PathLike]] = None,
        speaker_id: Optional[int] = None,
        prompt: Optional[str] = None,
    ) -> Union[SherpaResult, ExpressiveResult, NativeResult, MultilingualResult, SOTAResult]:
        """Synthesize text into speech audio buffer / WAV file with Zero-Config intelligent routing."""
        if self._is_closed:
            raise TTSInferenceError("Cannot synthesize: Engine session is closed.")

        output = output or output_path
        clean_text = text.strip() if text else ""
        if not clean_text:
            raise TTSInferenceError("Cannot synthesize empty text.")

        from .script_classifier import normalize_language_code
        target_lang = normalize_language_code(language or self.language)

        # Resolve speaker profile if provided via --profile or --ref-audio
        resolved_profile = None
        if profile is not None or ref_audio is not None:
            try:
                from .speaker_profile import SpeakerProfileEngine
                prof_engine = SpeakerProfileEngine()
                if profile is not None:
                    resolved_profile = prof_engine.get_profile(str(profile))
                elif ref_audio is not None:
                    resolved_profile = prof_engine.extract_from_file(ref_audio)
            except Exception as e:
                logger.warning(f"Could not load/extract speaker profile: {e}")

        # 1. If user explicitly pinned to OS Native voice, speak directly
        if self.requested_engine_type == "native":
            return self.native_engine.speak(clean_text)

        # 2. Multilingual auto-detection:
        # Route to MultilingualNeuralEngine if multiple languages are present in text
        # or if caller specifically requested multilingual / hybrid engine.
        detected_langs = MultilingualTokenizer.detect_languages(clean_text)
        is_mixed_text = len(detected_langs) > 1

        is_sota_engine = self.requested_engine_type in ("cosyvoice", "cosyvoice2", "f5", "f5tts", "chattts", "sota")

        should_route_multilingual = (
            not is_sota_engine and (
                (self.requested_engine_type in ("multilingual", "codeswitch", "hybrid")) or
                (self.requested_engine_type == "auto" and (is_mixed_text or self._multilingual_engine is not None)) or
                (is_mixed_text and self.requested_engine_type != "native") or
                (target_lang in ("hi", "ru", "ja", "zh", "es", "fr", "de", "ar") and self.requested_engine_type != "native")
            )
        )

        if should_route_multilingual:
            multi_engine = self._get_multilingual_engine()
            kwargs = {}
            if target_lang != "auto":
                kwargs["language"] = target_lang
            if resolved_profile is not None:
                kwargs["speaker_profile"] = resolved_profile
            return multi_engine.synthesize(
                clean_text,
                output=output,
                speed=speed,
                mode=mode,
                **kwargs
            )

        # 4. Standard single-language engine / SOTA engine
        if hasattr(self.synth_engine, "synthesize"):
            import inspect
            sig = inspect.signature(self.synth_engine.synthesize)
            kwargs = {}
            if "preset" in sig.parameters and preset is not None:
                kwargs["preset"] = preset
            if "ref_audio" in sig.parameters and ref_audio is not None:
                kwargs["ref_audio"] = ref_audio
            if "ref_text" in sig.parameters and ref_text is not None:
                kwargs["ref_text"] = ref_text
            if "prompt" in sig.parameters and prompt is not None:
                kwargs["prompt"] = prompt
            if "speaker_id" in sig.parameters and speaker_id is not None:
                kwargs["speaker_id"] = speaker_id
            if "speaker_profile" in sig.parameters and resolved_profile is not None:
                kwargs["speaker_profile"] = resolved_profile
            elif "profile" in sig.parameters and resolved_profile is not None:
                kwargs["profile"] = resolved_profile
            return self.synth_engine.synthesize(clean_text, output=output, speed=speed, **kwargs)
        elif hasattr(self.synth_engine, "speak"):
            return self.synth_engine.speak(clean_text)
        raise TTSInferenceError(f"Selected engine '{type(self.synth_engine).__name__}' does not support synthesize.")

    def close(self) -> None:
        self._is_closed = True
        self.native_engine.close()
        if hasattr(self.synth_engine, "close"):
            self.synth_engine.close()
        if self._multilingual_engine is not None:
            self._multilingual_engine.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def load(
    model: Optional[str] = None,
    language: str = "auto",
    preset: str = "balanced",
    device: str = "auto",
    threads: int = 4,
    sample_rate: Optional[int] = None,
    engine: str = "auto",
    tier: Optional[str] = None,
    rpc: Optional[str] = None,
    tensor_split: Optional[str] = None,
    cluster_rpc_servers: Optional[Any] = None,
    cluster_split_mode: Optional[str] = None,
    cluster_tensor_split: Optional[str] = None,
    cluster_vram_budget: Optional[dict] = None,
) -> TTSEngine:
    return TTSEngine(
        model_path=model,
        language=language,
        preset=preset,
        device=device,
        threads=threads,
        sample_rate=sample_rate,
        engine_type=engine,
        model_tier=tier,
        rpc=rpc,
        tensor_split=tensor_split,
        cluster_rpc_servers=cluster_rpc_servers,
        cluster_split_mode=cluster_split_mode,
        cluster_tensor_split=cluster_tensor_split,
        cluster_vram_budget=cluster_vram_budget,
    )


def doctor() -> Dict[str, Any]:
    ameva_mod = _resolve_ameva_runtime()
    if ameva_mod is not None:
        try:
            from ameva_runtime.adapters import TtsAdapter
            adapter = TtsAdapter()
            rep = adapter.resolve_diagnostic_report()
            return {
                "doctor_report": rep,
                "overall_success": getattr(rep, "overall_success", False),
                "passed_stages": getattr(rep, "passed_stages", 0),
                "recommended_backend": getattr(rep, "recommended_backend", "cpu_neon"),
                "status": "DIAGNOSED_VIA_AMEVA",
            }
        except Exception as e:
            logger.debug("AMEVA Doctor invocation exception: %s", e)

    return {
        "doctor_report": None,
        "overall_success": False,
        "passed_stages": 0,
        "recommended_backend": "cpu_neon",
        "status": "NO_AMEVA_RUNTIME",
    }
