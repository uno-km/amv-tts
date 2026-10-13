# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
termux-tts: Production-Grade 4-Tier TTS Framework for Android Termux.
- Tier 1: Parametric DSP Formant Synthesizer Engine (0MB Zero-Dependency)
- Tier 2: Android System Native Voice Engine Bridge (Samsung / Google Voice)
- Tier 3: Authentic C++ Subprocess-Isolated Sherpa-ONNX Neural Vocoder (VITS Deep Learning)
- Tier 4: Pure On-Device Expressive Emotional Synthesizer (Conversational Tags)
"""

__version__ = "1.7.0"

from .engine import TTSEngine, load, doctor, QUALITY_PRESETS
from .engine_native import NativeAndroidEngine, NativeResult
from .engine_sherpa import SherpaNeuralEngine, SherpaResult
from .engine_vulkan import VulkanNeuralEngine, VulkanResult
from .engine_expressive import ExpressiveEngine, ExpressiveResult
from .engine_multilingual import MultilingualNeuralEngine, MultilingualResult
from .engine_sota import SOTANeuralEngine, SOTABackend, SOTAResult
from .engine_sherpa_capi import SherpaResidentManager, SherpaCapiSession
from .script_classifier import MultilingualTokenizer, ScriptRegistry, LanguageChunk
from .installer import run_installation
from .tokenizer import PhoneticTokenizer, EXPRESSIVE_TAGS
from .g2p_korean import KoreanG2PEngine, korean_text_to_phonemes
from .audio import AudioBuffer
from .exceptions import (
    TTSError,
    TTSModelLoadError,
    TTSInferenceError,
    VulkanInitializationError,
    TTSAudioEncodingError,
    TTSLanguageNotSupportedError,
    AmevaTermuxError,
    TermuxTTSError,
    ClusterConnectionError,
    ClusterConfigurationError,
)
from . import cluster
from .cluster import (
    parse_cluster_rpc_spec,
    verify_rpc_cluster_nodes,
    verify_rpc_cluster_health,
)
from .hardware import HardwareProfile, detect_hardware, resolve_device
from .downloader import list_models, download_model, resolve_model_path

# Backward-compatibility alias
ONNXNeuralEngine = SherpaNeuralEngine
ONNXResult = SherpaResult

__version__ = "1.7.0"
__all__ = [
    "TTSEngine",
    "load",
    "doctor",
    "NativeAndroidEngine",
    "NativeResult",
    "SherpaNeuralEngine",
    "SherpaResult",
    "VulkanNeuralEngine",
    "VulkanResult",
    "ExpressiveEngine",
    "ExpressiveResult",
    "MultilingualNeuralEngine",
    "MultilingualResult",
    "SOTANeuralEngine",
    "SOTABackend",
    "SOTAResult",
    "SherpaResidentManager",
    "SherpaCapiSession",
    "MultilingualTokenizer",
    "ScriptRegistry",
    "LanguageChunk",
    "run_installation",
    "ONNXNeuralEngine",
    "ONNXResult",
    "QUALITY_PRESETS",
    "PhoneticTokenizer",
    "KoreanG2PEngine",
    "korean_text_to_phonemes",
    "EXPRESSIVE_TAGS",
    "AudioBuffer",
    "HardwareProfile",
    "detect_hardware",
    "resolve_device",
    "list_models",
    "download_model",
    "resolve_model_path",
    "AmevaTermuxError",
    "TermuxTTSError",
    "ClusterConnectionError",
    "ClusterConfigurationError",
    "cluster",
    "parse_cluster_rpc_spec",
    "verify_rpc_cluster_nodes",
    "verify_rpc_cluster_health",
    "TTSError",
    "TTSModelLoadError",
    "TTSInferenceError",
    "VulkanInitializationError",
    "TTSAudioEncodingError",
    "TTSLanguageNotSupportedError"
]
