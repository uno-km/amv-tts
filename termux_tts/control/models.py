# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
termux_tts.control.models
ModelRegistry 연동 — TTS 모델 + Voice 분리 관리.

TTS 특이사항:
  - model_id:  "dsp-ko", "piper-ko", "fastspeech-ko" 등 엔진/언어 조합
  - voice_id:  "ko-speaker-01", "en-speaker-01" 등 화자 ID (모델에 종속)
  - 내장 엔진(dsp/native)은 파일 없이도 항상 사용 가능 → missing 상태로 표기 금지
  - ONNX 모델은 파일 존재만으로 installed 처리 금지 → Registry 기반
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from ameva_component import ModelRegistry
from ameva_component.manifest import ModelState

COMPONENT_ID = "termux-tts"

# ------------------------------------------------------------------
# 내장(Built-in) 엔진 정의 — 파일 불필요, 항상 사용 가능
# ------------------------------------------------------------------

# format: {model_id: {description, engine, voices, backend_key}}
BUILTIN_MODELS: dict[str, dict[str, Any]] = {
    "dsp-ko": {
        "model_id": "dsp-ko",
        "description": "Pure Parametric DSP Formant Synthesizer (Korean, zero-dependency)",
        "engine": "dsp",
        "format": "builtin",
        "language": "ko",
        "voices": [
            {"voice_id": "ko-default", "description": "Default Korean DSP voice", "language": "ko"},
        ],
        "state": ModelState.ACTIVE.value,   # 내장 엔진은 항상 사용 가능
    },
    "dsp-en": {
        "model_id": "dsp-en",
        "description": "Pure Parametric DSP Formant Synthesizer (English, zero-dependency)",
        "engine": "dsp",
        "format": "builtin",
        "language": "en",
        "voices": [
            {"voice_id": "en-default", "description": "Default English DSP voice", "language": "en"},
        ],
        "state": ModelState.ACTIVE.value,
    },
    "native-system": {
        "model_id": "native-system",
        "description": "Android System Native Voice (Samsung/Google TTS bridge)",
        "engine": "native",
        "format": "builtin",
        "language": "ko,en",
        "voices": [
            {"voice_id": "system-default", "description": "Android system TTS voice", "language": "ko,en"},
        ],
        "state": ModelState.ACTIVE.value,
    },
}


class TTSModelRegistry:
    """
    ModelRegistry를 감싸는 TTS 전용 레지스트리.

    내장 엔진(dsp, native)은 파일 없이도 항상 사용 가능하며,
    ONNX 모델만 ModelRegistry를 통해 상태를 추적합니다.
    """

    def __init__(self, registry_dir: Path | None = None) -> None:
        self._reg = ModelRegistry(COMPONENT_ID, registry_dir=registry_dir)

    # ------------------------------------------------------------------
    # 전체 목록 조회
    # ------------------------------------------------------------------

    def list_all(self) -> list[dict]:
        """
        모든 모델 레코드를 반환합니다.
        - 내장 엔진: BUILTIN_MODELS에서 직접 반환
        - ONNX 모델: ModelRegistry에서 실제 상태 조회
        """
        result: list[dict] = list(BUILTIN_MODELS.values())
        # Registry에서 외부 ONNX 모델 추가
        for rec in self._reg.list_all():
            result.append(self._enrich_onnx_record(rec))
        return result

    def list_voices_for_model(self, model_id: str) -> list[dict]:
        """특정 model_id의 voice 목록을 반환합니다."""
        if model_id in BUILTIN_MODELS:
            return BUILTIN_MODELS[model_id].get("voices", [])
        rec = self._reg.get(model_id)
        if rec is None:
            return []
        return rec.get("voices", [])

    def get(self, model_id: str) -> dict | None:
        """단일 모델 레코드를 반환합니다. 없으면 None."""
        if model_id in BUILTIN_MODELS:
            return BUILTIN_MODELS[model_id]
        rec = self._reg.get(model_id)
        if rec is None:
            return None
        return self._enrich_onnx_record(rec)

    def get_state(self, model_id: str) -> ModelState | None:
        """모델 상태를 반환합니다. 없으면 None."""
        if model_id in BUILTIN_MODELS:
            return ModelState.ACTIVE
        rec = self._reg.get(model_id)
        if rec is None:
            return None
        try:
            return ModelState(rec.get("state", "missing"))
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # 상태 변경
    # ------------------------------------------------------------------

    def set_onnx_active(self, model_id: str) -> None:
        """ONNX 모델을 active 상태로 변경합니다."""
        self._reg.set_state(model_id, ModelState.ACTIVE)

    def set_onnx_inactive(self, model_id: str) -> None:
        """ONNX 모델을 inactive 상태로 변경합니다."""
        self._reg.set_state(model_id, ModelState.INACTIVE)

    def record_onnx_install(self, model_id: str, manifest, verified_at: float) -> None:
        """ONNX 모델 설치 완료를 기록합니다."""
        self._reg.record_install(model_id, manifest, verified_at)

    def set_onnx_state(self, model_id: str, state: ModelState, *, last_error: str | None = None) -> None:
        """ONNX 모델 상태를 직접 갱신합니다."""
        self._reg.set_state(model_id, state, last_error=last_error)

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _enrich_onnx_record(self, rec: dict) -> dict:
        """ONNX 레코드에 engine/voices 필드가 없으면 기본값을 추가합니다."""
        enriched = dict(rec)
        enriched.setdefault("engine", "onnx")
        enriched.setdefault("voices", [])
        return enriched
