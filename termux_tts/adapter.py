# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
termux_tts.adapter
===================
AMEVA Component Protocol v1 — Orchestrator Adapter (v0.8.1 호환)

P0-2: infer() fallback yield _not_supported → raise OperationNotSupported
P0-4: except Exception → retryable 분류, 원본 오류 코드 보존
"""
from __future__ import annotations

from typing import Any, AsyncIterator

from ameva_component.adapter_base import BaseOrchestratorAdapter
from ameva_component.exceptions import (
    ComponentError,
    OperationNotSupported,
    redact_details,
    redact_text,
)
from termux_tts.control.component import TTSControl


class TTSOrchestratorAdapter(BaseOrchestratorAdapter):
    """TTS (Text-to-Speech) Orchestrator Adapter.

    합성은 파일 기반으로 수행됩니다.
    infer(): text를 받아 audio 파일 경로를 반환합니다.
    요청 모델과 실행 모델이 다르면 fallback_used=True를 명시합니다.
    """

    COMPONENT_ID = "termux-tts"

    _RETRYABLE_CODES: frozenset[str] = frozenset({
        "REMOTE_TIMEOUT",
        "MODEL_BUSY",
        "TEMPORARY_RESOURCE_UNAVAILABLE",
    })
    _NON_RETRYABLE_CODES: frozenset[str] = frozenset({
        "VOICE_NOT_FOUND",
        "TEXT_TOO_LONG",
        "UNSUPPORTED_LANGUAGE",
        "OUTPUT_PATH_FORBIDDEN",
        "ENCODING_FAILED",
    })

    def __init__(self, control: TTSControl | None = None) -> None:
        self._control = control or TTSControl()

    async def infer(self, request: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """TTS synthesis: text → audio file path.

        request 키:
            text (str): 합성할 텍스트 (필수, 비어있으면 안 됨)
            voice_id (str): 목소리 ID (선택)
            model_id (str): 모델 ID (선택)
            output_path (str): 출력 파일 경로 (선택)

        반환 프레임:
            {"type": "audio", "audio_path": str, "final": bool, "fallback_used": bool}
            {"type": "error", "ok": False, "error": {...}}
        """
        text = request.get("text", "").strip()
        if not text:
            yield {
                "type": "error",
                "ok": False,
                "error": {
                    "code": "TEXT_EMPTY",
                    "message": "text is required and must not be empty",
                    "operation": "infer",
                    "component_id": self.COMPONENT_ID,
                    "retryable": False,
                },
            }
            return

        if not hasattr(self._control, "synthesize"):
            # P0-2: fallback yield → raise
            raise OperationNotSupported(operation="infer.synthesize", component_id=self.COMPONENT_ID)

        try:
            result = await self._control.synthesize(request)

            if not isinstance(result, dict):
                raise ValueError(f"synthesize() must return dict, got {type(result).__name__}")

            if result.get("ok") is not True:
                err_payload = result.get("error") if isinstance(result.get("error"), dict) else {}
                yield {
                    "type": "error",
                    "ok": False,
                    "error": {
                        "code": err_payload.get("code", "ADAPTER_RESULT_NOT_SUCCESS"),
                        "message": err_payload.get("message", "synthesize() did not return ok=True"),
                        "operation": "infer",
                        "component_id": self.COMPONENT_ID,
                        "retryable": False,
                        "details": {"result_keys": sorted(result.keys())},
                    },
                }
                return

            audio_path = result.get("audio_path")
            if not audio_path:
                raise ValueError("synthesize() result missing required field 'audio_path'")
            if not isinstance(audio_path, str):
                raise TypeError(f"synthesize() audio_path must be str, got {type(audio_path).__name__}")

            yield {
                "type": "audio",
                "audio_path": audio_path,
                "final": True,
                "fallback_used": result.get("fallback_used", False),
                "requested_voice": result.get("requested_voice"),
                "executed_voice": result.get("executed_voice"),
                "ok": True,
            }

        except ComponentError as component_err:
            # HIGH 1: ComponentError public_message와 redact_details로 외부 노출 보안 격리
            import logging
            logging.getLogger(__name__).exception(
                "TTS component operation failed",
                extra={
                    "component_id": self.COMPONENT_ID,
                    "operation": "infer",
                    "code": getattr(component_err, "code", "COMPONENT_ERROR"),
                },
            )
            if hasattr(component_err, "to_public_dict"):
                err_dict = component_err.to_public_dict()
            else:
                err_dict = {
                    "code": getattr(component_err, "code", "COMPONENT_ERROR"),
                    "message": redact_text(getattr(component_err, "public_message", "Component operation failed")),
                    "retryable": getattr(component_err, "retryable", False),
                    "details": redact_details(getattr(component_err, "details", {})),
                }
            yield {
                "type": "error",
                "ok": False,
                "error": {
                    **err_dict,
                    "operation": "infer",
                    "component_id": self.COMPONENT_ID,
                    "retryable": self._classify_retryable(
                        err_dict.get("code", ""), default=err_dict.get("retryable", False)
                    ),
                },
            }

        except (ValueError, TypeError) as contract_err:
            # P0-3/4 & HIGH 1: 계약 위반 (str(contract_err) 직접 노출 금지, 내부 로그 격리)
            import logging
            logging.getLogger(__name__).exception("TTS contract validation error during infer: %s", contract_err)
            yield {
                "type": "error",
                "ok": False,
                "error": {
                    "code": "ADAPTER_CONTRACT_ERROR",
                    "message": "Adapter contract validation failed",
                    "operation": "infer",
                    "component_id": self.COMPONENT_ID,
                    "retryable": False,
                    "details": {
                        "cause_type": type(contract_err).__name__,
                        "operation": "infer",
                    },
                },
            }

        except Exception as unexpected_err:
            import logging
            logging.getLogger(__name__).exception("TTS adapter unexpected error during infer: %s", unexpected_err)
            code = getattr(unexpected_err, "code", "ADAPTER_INTERNAL_ERROR")
            yield {
                "type": "error",
                "ok": False,
                "error": {
                    "code": code if isinstance(code, str) else "ADAPTER_INTERNAL_ERROR",
                    "message": "Unexpected adapter failure",
                    "operation": "infer",
                    "component_id": self.COMPONENT_ID,
                    "retryable": False,
                    "details": {
                        "cause_type": type(unexpected_err).__name__,
                        "operation": "infer",
                    },
                },
            }

    def _classify_retryable(self, code: str, *, default: bool = False) -> bool:
        if code in self._RETRYABLE_CODES:
            return True
        if code in self._NON_RETRYABLE_CODES:
            return False
        return default


def create_adapter() -> TTSOrchestratorAdapter:
    """Entry Point Factory."""
    return TTSOrchestratorAdapter()
