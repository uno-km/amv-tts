# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
Strict Dual Routing & Zero-Fallback Fail-Fast Verification for termux-tts.
Tests:
1. Explicit Vulkan mode (--device vulkan) -> Strict Fail-Fast on unavailable systems.
2. Auto mode (--device auto) -> Auto-detection and seamless transition to CPU.
3. Explicit CPU mode (--device cpu) -> Pure CPU execution.
"""

import pytest
import termux_tts as tts
from termux_tts.exceptions import VulkanInitializationError, TTSInferenceError, TTSModelLoadError
from termux_tts.engine import load

def test_explicit_cpu_mode():
    try:
        engine = load(language="ko", device="cpu")
    except (TTSModelLoadError, TTSInferenceError):
        pytest.skip("Neural models not installed in local host environment")
    with engine:
        res = engine.synthesize("CPU 전용 모드 테스트입니다.")
        assert "CPU" in res.backend
        assert res.duration_sec > 0.3
        print(f"\n[PASS CPU MODE] Backend: {res.backend}")

def test_auto_routing_mode():
    try:
        engine = load(language="ko", device="auto")
    except (TTSModelLoadError, TTSInferenceError):
        pytest.skip("Neural models not installed in local host environment")
    with engine:
        res = engine.synthesize("자동 라우팅 모드 테스트입니다.")
        assert res.backend in ["VULKAN_GPU", "ARM64_NEON_CPU", "X86_64_AVX2_CPU", "X86_CPU"]
        print(f"\n[PASS AUTO ROUTING] Resolved Backend: {res.backend}")

def test_explicit_vulkan_fail_fast_when_disabled(monkeypatch):
    try:
        from ameva_runtime.doctor import DiagnosticReport
        from ameva_runtime.adapters import TtsAdapter
    except ImportError:
        pytest.skip("ameva-runtime not installed in current environment")
    
    disabled_report = DiagnosticReport(
        device_name="None",
        vendor_id=0,
        overall_success=False,
        recommended_backend="cpu_neon",
        passed_stages=0,
        total_stages=12,
    )
    monkeypatch.setattr(TtsAdapter, "resolve_diagnostic_report", lambda *args, **kwargs: disabled_report)

    # 1. Explicit Vulkan MUST RAISE VulkanInitializationError (No silent fallback!)
    with pytest.raises(VulkanInitializationError) as exc_info:
        load(language="ko", device="vulkan")
    assert "FAIL-FAST" in str(exc_info.value)
    assert "--device cpu" in str(exc_info.value)
    print(f"\n[PASS FAIL-FAST GUARD] Correctly rejected with: {exc_info.value}")

    # 2. Auto mode MUST gracefully route to CPU
    try:
        engine = load(language="ko", device="auto")
    except (TTSModelLoadError, TTSInferenceError):
        pytest.skip("Neural models not installed in local host environment")
    with engine:
        res = engine.synthesize("Vulkan 없을 때 자동 CPU 전환 테스트.")
        assert "CPU" in res.backend
        print(f"\n[PASS AUTO DEGRADE TO CPU] Gracefully resolved: {res.backend}")
