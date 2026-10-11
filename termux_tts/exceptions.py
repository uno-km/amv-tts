# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
AMEVA Unified Exception Hierarchy for Termux AI Engines.
Component: [TTS]
"""
from typing import Optional, Any


class AmevaTermuxError(Exception):
    """Root exception for all Termux On-Device AI Engines."""
    COMPONENT_TAG = "[TTS]"
    DEFAULT_CODE = "E000_UNKNOWN"

    def __init__(self, message: str, code: Optional[str] = None, details: Optional[Any] = None):
        self.code = code or self.DEFAULT_CODE
        self.details = details
        self.raw_message = message
        super().__init__(f"{self.COMPONENT_TAG} [{self.code}] {message}")


# Package-specific Root Alias
TermuxTTSError = AmevaTermuxError


# Standard Common Exceptions
class ModelNotFoundError(AmevaTermuxError):
    """Raised when the specified model checkpoint or weights cannot be located."""
    DEFAULT_CODE = "E001_MODEL_NOT_FOUND"


class PlatformNotSupportedError(AmevaTermuxError):
    """Raised when running on an incompatible platform, architecture, or OS."""
    DEFAULT_CODE = "E002_PLATFORM_NOT_SUPPORTED"

class RuntimeRequiredError(PlatformNotSupportedError):
    """A GPU request needs ameva-runtime (AMUDA), which is missing or too old.

    The message is the shared AMU_ERR_RUNTIME_REQUIRED explanation. The CLI exits with status 69;
    nothing is run and there is no CPU fallback.
    """

    runtime_required = True


class HardwareCompatibilityError(AmevaTermuxError):
    """Raised when device hardware (RAM, NEON, Vulkan, NPU) is insufficient."""
    DEFAULT_CODE = "E003_HARDWARE_INCOMPATIBLE"


class RuntimeNotFoundError(AmevaTermuxError):
    """Raised when the native binary executable or shared library is missing."""
    DEFAULT_CODE = "E004_RUNTIME_NOT_FOUND"


class ProvisioningError(AmevaTermuxError):
    """Raised when downloading, compiling, or provisioning binaries fails."""
    DEFAULT_CODE = "E005_PROVISIONING_FAILED"


class InferenceTimeoutError(AmevaTermuxError):
    """Raised when inference execution exceeds the safety deadline."""
    DEFAULT_CODE = "E006_INFERENCE_TIMEOUT"


class InferenceExecutionError(AmevaTermuxError):
    """Raised when engine process or native runtime crashes during inference."""
    DEFAULT_CODE = "E007_INFERENCE_FAILED"


class ModelCorruptedError(AmevaTermuxError):
    """Raised when model weights fail SHA-256 or GGUF/Safetensors integrity checks."""
    DEFAULT_CODE = "E008_MODEL_CORRUPTED"


class ModelDownloadError(ProvisioningError):
    """Raised when network download for model weights fails."""
    DEFAULT_CODE = "E009_MODEL_DOWNLOAD_FAILED"


class ClusterConnectionError(AmevaTermuxError):
    """Raised when one or more distributed RPC worker nodes cannot be reached or fail handshake."""
    DEFAULT_CODE = "E010_CLUSTER_CONNECTION"


class ClusterConfigurationError(AmevaTermuxError):
    """Raised when cluster parameters (tensor split, backend, vram budget) are invalid."""
    DEFAULT_CODE = "E011_CLUSTER_CONFIG"


class ClusterLicenseRequiredError(AmevaTermuxError):
    """Raised when distributed clustering is invoked without required AMEVA Cluster license."""
    DEFAULT_CODE = "E403_CLUSTER_LICENSE_REQUIRED"

    def __init__(self, message: str = ""):
        if not message:
            message = (
                "\n================================================================================\n"
                "[AMEVA-CLUSTER] CLUSTER LICENSE REQUIRED (E403)\n"
                "================================================================================\n"
                "Multi-device distributed clustering is an exclusive capability of 'ameva-cluster'.\n"
                "Standalone distributed execution without the official AMEVA-Cluster package is prohibited.\n\n"
                "Resolution:\n"
                "  1. Install official AMEVA-Cluster runtime:\n"
                "     pip install ameva-cluster  (or npm install @ameva/cluster)\n"
                "  2. Launch worker/master through 'ameva-cluster' control plane:\n"
                "     ameva-cluster worker\n"
                "     ameva-cluster master -m model.gguf\n"
                "================================================================================"
            )
        super().__init__(message)



# Legacy TTS Aliases
TTSError = AmevaTermuxError
TTSModelLoadError = ModelNotFoundError
TTSInferenceError = InferenceExecutionError

class VulkanInitializationError(InferenceExecutionError):
    DEFAULT_CODE = "E201_VULKAN_INIT_FAILED"

class TTSAudioEncodingError(AmevaTermuxError):
    DEFAULT_CODE = "E202_AUDIO_ENCODING_FAILED"

class TTSLanguageNotSupportedError(AmevaTermuxError):
    DEFAULT_CODE = "E203_LANGUAGE_NOT_SUPPORTED"

class TTSConfigurationError(AmevaTermuxError):
    DEFAULT_CODE = "E204_CONFIG_INVALID"

