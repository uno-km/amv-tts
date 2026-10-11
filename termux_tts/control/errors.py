# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
termux_tts.control.errors
termux-tts 패키지 고유 오류 정의.

공통 AmevaError 하위 클래스로 정의하여 render_from_exception()과 호환됩니다.
"""
from __future__ import annotations

from ameva_component.exceptions import AmevaError


class VoiceNotFound(AmevaError):
    """요청한 voice_id가 Registry 또는 내장 목록에 없을 때."""

    code = "VOICE_NOT_FOUND"
    exit_code = 65

    def __init__(self, voice_id: str, model_id: str | None = None) -> None:
        where = f" in model '{model_id}'" if model_id else ""
        super().__init__(
            f"Voice '{voice_id}' not found{where}",
            details={"voice_id": voice_id, "model_id": model_id},
        )
        self.voice_id = voice_id
        self.model_id = model_id


class EngineNotAvailable(AmevaError):
    """요청한 엔진 백엔드(dsp/onnx/native)가 현재 환경에서 사용 불가능할 때."""

    code = "ENGINE_NOT_AVAILABLE"
    exit_code = 69

    def __init__(self, engine_name: str, reason: str) -> None:
        super().__init__(
            f"Engine '{engine_name}' is not available: {reason}",
            details={"engine_name": engine_name, "reason": reason},
        )
        self.engine_name = engine_name


class ModelPathRequired(AmevaError):
    """ONNX 엔진 사용 시 model_path가 없을 때."""

    code = "MODEL_PATH_REQUIRED"
    exit_code = 64

    def __init__(self) -> None:
        super().__init__(
            "model_path is required for ONNX engine activation",
            details={"hint": "Pass model_path in the activation request"},
        )
