# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
Hardware abstraction gateway and dynamic ameva-runtime soft-dependency resolver for termux-tts.
Provides strict Zero-Silent-Fallback [AMEVA-TTS-E001] compliance, hardware-agnostic routing,
and seamless fallback to Tier 1 DSP Formant / Tier 2 Native voice when running standalone.
"""
from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
import sys
from typing import Optional, Tuple, Any

from .exceptions import VulkanInitializationError

logger = logging.getLogger("termux_tts.hardware")

ERROR_AMEVA_TTS_E001 = (
    "[FAIL-FAST] [ERROR: AMEVA-TTS-E001] GPU acceleration requires 'ameva-runtime' and provisioned Vulkan assets.\n"
    "Cause: Hardware acceleration runtime or native Vulkan engine is not installed.\n"
    "Action Required: Run one-click provisioning via:\n"
    "  $ termux-tts install --tier high\n"
    "  (Or run without GPU: termux-tts synth -e dsp -t \"...\")\n"
    "Documentation: https://uno-km.vercel.app/lib/tts/"
)


def _resolve_ameva_runtime() -> Optional[Any]:
    """Check for ameva_runtime availability without top-level static dependency.

    Returns the ameva_runtime module if installed, otherwise None.
    """
    try:
        spec = importlib.util.find_spec("ameva_runtime")
        if spec is not None:
            import ameva_runtime
            return ameva_runtime
    except (ImportError, AttributeError):
        pass
    return None


GPU_RUNTIME_MIN_VERSION = "2.0.0"


def _require_gpu_runtime(requested_device: str) -> None:
    """Stop with the shared AMU_ERR_RUNTIME_REQUIRED explanation unless ameva-runtime is usable.

    Every explicit GPU request goes through here. There is no CPU fallback: the caller gets
    RuntimeRequiredError and the CLI exits with status 69.
    """
    from termux_tts.exceptions import RuntimeRequiredError
    try:
        from ameva_device import AmevaRuntimeRequiredError, require_ameva_runtime
    except ImportError as exc:
        raise RuntimeRequiredError(
            "[AMU_ERR_RUNTIME_REQUIRED] termux-tts: GPU execution (device='%s') needs ameva-runtime (AMUDA), "
            "and the shared check library 'ameva-device-sdk' is missing from this installation.\n"
            "Nothing was run, and no CPU fallback was used.\n"
            "Install:\n"
            "  pip install \"ameva-runtime>=%s\"\n"
            "  npm install -g @ameva/runtime\n"
            "  Releases and documentation: https://github.com/uno-km/ameva-runtime-releases\n"
            "To run on the CPU instead, use --device cpu.\n" % (requested_device, GPU_RUNTIME_MIN_VERSION)
        ) from exc
    try:
        require_ameva_runtime(
            "termux-tts", "GPU execution (device='%s')" % requested_device,
            min_version=GPU_RUNTIME_MIN_VERSION,
        )
    except AmevaRuntimeRequiredError as exc:
        raise RuntimeRequiredError(str(exc)) from exc


_NOTED_CPU_REASONS = set()


def _note_cpu_instead(reason: str) -> None:
    """auto mode: say on stderr (once per reason) that the GPU was not used, so the CPU run is never silent."""
    import sys
    if reason in _NOTED_CPU_REASONS:
        return
    _NOTED_CPU_REASONS.add(reason)
    sys.stderr.write("[termux-tts] GPU not used: %s. This run uses the CPU. "
                     "To enable the GPU: pip install ameva-runtime (https://github.com/uno-km/ameva-runtime-releases)\n" % reason)
    sys.stderr.flush()


def resolve_device_backend(
    requested_device: str,
    requested_engine: str = "auto",
) -> Tuple[str, str]:
    """Resolve user requested device/engine into (device, engine_type) with fail-fast compliance.

    Returns:
        Tuple[str, str]: (device, engine_type) e.g. ('vulkan', 'vulkan') or ('cpu', 'dsp')
    """
    req_dev = (requested_device or "auto").lower().strip()
    req_eng = (requested_engine or "auto").lower().strip()
    ameva_mod = _resolve_ameva_runtime()

    # Explicit OpenCL used to end on the CPU without a word; it needs the runtime and is not implemented here.
    if req_dev == "opencl":
        _require_gpu_runtime(req_dev)
        raise VulkanInitializationError(
            "[FAIL-FAST] termux-tts has no OpenCL synthesis engine. "
            "Use '--device vulkan' (needs ameva-runtime) or '--device cpu'."
        )
    # Explicit GPU/Vulkan requested
    if req_dev in ("vulkan", "gpu") or req_eng in ("vulkan", "gpu", "ncnn"):
        # Every explicit GPU request needs ameva-runtime (AMUDA); no CPU fallback.
        _require_gpu_runtime(req_dev if req_dev in ("vulkan", "gpu") else req_eng)

        # Check Vulkan doctor from ameva_runtime TtsAdapter
        try:
            from ameva_runtime.adapters import TtsAdapter
            adapter = TtsAdapter()
            report = adapter.resolve_diagnostic_report()
            is_vk = getattr(report, "overall_success", False) or getattr(report, "recommended_backend", "") == "vulkan"
            if not is_vk:
                raise VulkanInitializationError(
                    f"[FAIL-FAST] [ERROR: AMEVA-TTS-E002] Vulkan hardware acceleration is not supported on this device.\n"
                    f"Cause: No usable Vulkan physical device or ICD driver library found.\n"
                    f"Action Required: Use CPU or DSP synthesis via '--device cpu' or '--engine dsp'."
                )
        except VulkanInitializationError:
            raise
        except Exception as e:
            logger.debug("TtsAdapter diagnostic exception: %s", e)
            raise VulkanInitializationError(
                f"[FAIL-FAST] [ERROR: AMEVA-TTS-E002] Vulkan hardware acceleration check failed: {e}\n"
                f"Action Required: Use CPU or DSP synthesis via '--device cpu' or '--engine dsp'."
            ) from e

        effective_eng = req_eng if req_eng not in ("auto", "gpu") else "vulkan"
        return "vulkan", effective_eng

    # Auto mode: probe if ameva-runtime is available
    if req_dev == "auto":
        if ameva_mod is None:
            _note_cpu_instead("ameva-runtime (AMUDA) is not installed; operating in Tier 1 (Parametric DSP Formant) mode")
            return "cpu", req_eng

        try:
            from ameva_runtime.adapters import TtsAdapter
            adapter = TtsAdapter()
            report = adapter.resolve_diagnostic_report()
            is_vk = getattr(report, "overall_success", False) or getattr(report, "recommended_backend", "") == "vulkan"
            bin_path = adapter.resolve_binary_path()
            if is_vk and bin_path and req_eng in ("vulkan", "gpu"):
                return "vulkan", "vulkan"
        except Exception as e:
            logger.debug("TtsAdapter auto-routing probe exception: %s", e)
            _note_cpu_instead("GPU routing failed (%s)" % e)

        return "cpu", req_eng

    # Explicit CPU
    return "cpu", req_eng


def bind_tts_hardware(engine: Any, requested_device: str) -> Optional[Any]:
    """Safely invoke AMEVA-Runtime TtsAdapter if present to configure engine instance."""
    ameva_mod = _resolve_ameva_runtime()
    if ameva_mod is None:
        return None

    try:
        from ameva_runtime.adapters.tts import TtsAdapter
        binding = TtsAdapter.bind(engine=engine, requested_backend=requested_device)
        return binding
    except Exception as e:
        logger.debug("Hardware adapter binding skipped: %s", e)
        return None


def get_unified_model_search_dirs(submodule: str = "tts") -> list:
    """
    Returns unified model search paths adhering to AMEVA Ecosystem Shared Storage Specification.
    Enables zero-redundancy model sharing across STT, TTS, LLaMA, Vision, and Diffusion.
    """
    import os
    from pathlib import Path

    home = Path.home()
    dirs = []

    env_dir = os.environ.get("AMEVA_MODELS_DIR")
    if env_dir:
        p = Path(env_dir)
        dirs.extend([p / submodule, p])

    prefixes = [home]
    prefix_env = os.environ.get("PREFIX")
    if prefix_env:
        prefixes.append(Path(prefix_env).parent / "home")

    for base in prefixes:
        dirs.extend([
            base / "models" / submodule,
            base / "models",
            base / "ameva-models" / submodule,
            base / "ameva-models",
            base / ".cache" / "ameva" / "models" / submodule,
            base / ".cache" / "ameva" / "models",
            base / ".cache" / f"termux-{submodule}" / "models",
        ])

    # Deduplicate while preserving order
    seen = set()
    unique_dirs = []
    for d in dirs:
        resolved = str(d)
        if resolved not in seen:
            seen.add(resolved)
            unique_dirs.append(d)

    return unique_dirs


def get_clean_execution_env(extra_env: Optional[dict[str, str]] = None) -> dict[str, str]:
    """Assemble a clean environment dictionary conforming to Gate 1 safety rules.

    Guarantees:
    - Never injects /system/lib64, /vendor/lib64, /apex/ or other OS system paths into LD_LIBRARY_PATH.
    - Uses AMEVA-Runtime TtsAdapter.get_execution_env() when available.
    - Sanitizes existing LD_LIBRARY_PATH against forbidden system prefixes to prevent dual C++ runtime collisions.
    """
    import os
    from pathlib import Path

    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)

    try:
        from ameva_runtime.adapters.tts import TtsAdapter
        return TtsAdapter.get_execution_env(extra_env=env)
    except Exception:
        pass

    try:
        from ameva_runtime.adapters.base import get_vulkan_env
        return get_vulkan_env(base_env=env)
    except Exception:
        pass

    # Standalone Gate 1 fallback without ameva_runtime
    current_ld = env.get("LD_LIBRARY_PATH", "")
    forbidden_prefixes = ("/system/", "/vendor/", "/apex/", "/system_ext/", "/odm/", "/product/")
    existing_parts = [p for p in current_ld.split(":") if p and not any(p.startswith(fp) for fp in forbidden_prefixes)]

    home = Path.home()
    engine_dirs = []
    eng_lib = home / ".local" / "share" / "ameva" / "current" / "tts" / "lib"
    if eng_lib.is_dir():
        engine_dirs.append(str(eng_lib))
    local_lib = home / ".local" / "lib"
    if local_lib.is_dir():
        engine_dirs.append(str(local_lib))

    merged = []
    for d in engine_dirs + existing_parts:
        if d not in merged and not any(d.startswith(fp) for fp in forbidden_prefixes):
            merged.append(d)

    if merged:
        env["LD_LIBRARY_PATH"] = ":".join(merged)
    elif "LD_LIBRARY_PATH" in env:
        env["LD_LIBRARY_PATH"] = ""

    return env


# Standard Unified Hardware Interface Bridges
from dataclasses import dataclass, field
from typing import List

@dataclass
class HardwareProfile:
    is_termux: bool = False
    is_android: bool = False
    is_arm64: bool = False
    cpu_count: int = 4
    recommended_threads: int = 4
    ram_total_mb: float = 0.0
    ram_available_mb: float = 0.0
    has_neon: bool = True
    has_fp16: bool = False
    has_vulkan: bool = False
    gpu_name: Optional[str] = None
    soc_model: Optional[str] = None
    features: List[str] = field(default_factory=list)

    @property
    def soc_name(self) -> str:
        return self.soc_model or "Unknown SoC"

    @property
    def cpu_cores(self) -> int:
        return self.cpu_count

    @property
    def threads(self) -> int:
        return self.recommended_threads


def is_termux() -> bool:
    import os
    if os.environ.get("TERMUX_VERSION") or os.environ.get("TERMUX_APP_PID"):
        return True
    prefix = os.environ.get("PREFIX", "")
    if "com.termux" in prefix:
        return True
    return Path("/data/data/com.termux").is_dir()


def is_android() -> bool:
    """Check whether running on Android (Termux execution implies Android runtime)."""
    return is_termux()



def detect_hardware() -> HardwareProfile:
    import multiprocessing
    cores = multiprocessing.cpu_count()
    return HardwareProfile(
        is_termux=is_termux(),
        is_android=is_android(),
        cpu_count=cores,
        recommended_threads=max(1, cores // 2) if cores > 2 else cores,
    )


def resolve_device(requested_device: str = "auto") -> Tuple[str, int]:
    device, _ = resolve_device_backend(requested_device)
    return device, 32 if device == "vulkan" else 4


def get_optimal_threads() -> int:
    """Standard Unified Optimal Threads Calculator for termux-tts."""
    return detect_hardware().recommended_threads


def bind_hardware(engine: Any, requested_device: str = "auto", **kwargs) -> Optional[Any]:
    """Standard Unified Hardware Binding Interface for termux-tts."""
    return bind_tts_hardware(engine, requested_device)

