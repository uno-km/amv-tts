# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
termux_tts.control.instances
InstanceRegistry 연동 — DSP / ONNX / Native 엔진을 각각 Instance로 표현.

TTS Instance 정책:
  - DSP 엔진   → instance_id = "tts-dsp"
  - ONNX 엔진  → instance_id = "tts-onnx"
  - Native 엔진 → instance_id = "tts-native"
  - 각 엔진은 in-process ControlMode로 동작 (별도 Worker 프로세스 없음)
  - active_jobs 측정 실패를 0으로 변환하지 않음 → None 반환 후 호출자가 처리
"""
from __future__ import annotations

import time
from pathlib import Path

from ameva_component import InstanceRegistry
from ameva_component.instance import ControlMode, InstanceState, InstanceStatus

COMPONENT_ID = "termux-tts"

# TTS 엔진별 고정 Instance ID
INSTANCE_DSP    = "tts-dsp"
INSTANCE_ONNX   = "tts-onnx"
INSTANCE_NATIVE = "tts-native"

ALL_INSTANCE_IDS = (INSTANCE_DSP, INSTANCE_ONNX, INSTANCE_NATIVE)


def _make_default_instance(instance_id: str, model_id: str, backend: str) -> InstanceStatus:
    """엔진별 기본 InstanceStatus를 생성합니다."""
    return InstanceStatus(
        instance_id=instance_id,
        component_id=COMPONENT_ID,
        model_id=model_id,
        state=InstanceState.HOT,
        active_jobs=0,
        queue_depth=0,
        max_concurrency=1,
        backend=backend,
        started_at=time.time(),
        last_heartbeat=time.time(),
        last_error=None,
        control_mode=ControlMode.IN_PROCESS,
        endpoint=None,
    )


class TTSInstanceRegistry:
    """
    InstanceRegistry를 감싸는 TTS 전용 레지스트리.

    DSP/ONNX/Native 3개 엔진을 각각 독립 Instance로 추적합니다.
    Registry 파일이 없으면 기본 InstanceStatus를 생성하여 등록합니다.
    """

    def __init__(self, registry_dir: Path | None = None) -> None:
        self._reg = InstanceRegistry(COMPONENT_ID, registry_dir=registry_dir)
        self._ensure_defaults()

    # ------------------------------------------------------------------
    # 초기화
    # ------------------------------------------------------------------

    def _ensure_defaults(self) -> None:
        """레지스트리에 없는 엔진 Instance를 기본값으로 등록합니다."""
        defaults = {
            INSTANCE_DSP:    ("dsp-ko",        "dsp-cpu"),
            INSTANCE_ONNX:   ("native-system",  "onnx-cpu"),   # 활성 모델이 없을 때 기본값
            INSTANCE_NATIVE: ("native-system",  "android-native"),
        }
        for inst_id, (model_id, backend) in defaults.items():
            if self._reg.get(inst_id) is None:
                self._reg.register(_make_default_instance(inst_id, model_id, backend))

    # ------------------------------------------------------------------
    # 조회
    # ------------------------------------------------------------------

    def list_all(self) -> list[dict]:
        """등록된 모든 Instance를 dict 목록으로 반환합니다."""
        return [
            inst.to_dict()
            for inst in self._reg.list_all()
        ]

    def get(self, instance_id: str) -> InstanceStatus | None:
        """단일 Instance 상태를 반환합니다. 없으면 None."""
        return self._reg.get(instance_id)

    def total_active_jobs(self) -> int | None:
        """
        전체 Instance의 active_jobs 합계를 반환합니다.
        측정 실패 시 None — 0으로 변환 금지.
        """
        total = 0
        any_read = False
        for inst in self._reg.list_all():
            any_read = True
            total += inst.active_jobs
        return total if any_read else None

    # ------------------------------------------------------------------
    # 상태 변경
    # ------------------------------------------------------------------

    def on_model_activated(self, instance_id: str, model_id: str, backend: str) -> None:
        """모델 활성화 시 Instance model_id와 backend를 갱신합니다."""
        inst = self._reg.get(instance_id)
        if inst is None:
            inst = _make_default_instance(instance_id, model_id, backend)
        inst.model_id = model_id
        inst.backend = backend
        inst.state = InstanceState.HOT
        inst.last_error = None
        inst.last_heartbeat = time.time()
        self._reg.register(inst)

    def on_job_start(self, instance_id: str) -> None:
        """Job 시작 시 active_jobs++."""
        self._reg.increment_jobs(instance_id)

    def on_job_end(self, instance_id: str) -> None:
        """Job 종료 시 active_jobs-- (음수 방지)."""
        self._reg.decrement_jobs(instance_id)

    def on_engine_error(self, instance_id: str, error: str) -> None:
        """엔진 오류 시 Instance 상태를 FAILED로 변경합니다."""
        self._reg.update_state(instance_id, InstanceState.FAILED, last_error=error)

    def on_heartbeat(self, instance_id: str) -> None:
        """Heartbeat 갱신."""
        self._reg.update_heartbeat(instance_id)

    def update_state(self, instance_id: str, state: InstanceState, *, last_error: str | None = None) -> None:
        """Instance 상태를 직접 변경합니다."""
        self._reg.update_state(instance_id, state, last_error=last_error)
