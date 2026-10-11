# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
termux_tts.control.component
AMEVA Component Protocol v1 — TTSControl

기존 TTSEngine.doctor() Adapter 연결.
Model/Voice 분리 추적 (신규 — 기존 코드에 없음).
doctor_lite: 상태파일 + PID만 (12단계 Vulkan Doctor는 doctor_full에만).
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from ameva_component import (
    ActivationLock, ComponentInfo, ComponentStateFile,
    ControlMode, InstanceRegistry, InstanceState, InstanceStatus,
    ModelRegistry, ModelState, ModelNotFound, ModelLoadFailed,
    OperationNotSupported, now_timestamps, log_stderr, PROTOCOL_COMPONENT,
)
from ameva_component.control import ComponentControl


class TTSControl(ComponentControl):
    """
    termux-tts ComponentControl.
    Model/Voice 분리: model_id=piper-ko / voice_id=ko-speaker-01
    기존 engine.py TTSEngine은 Adapter로 연결합니다.
    """

    COMPONENT_ID   = "termux-tts"
    COMPONENT_TYPE = "tts"
    CAPABILITIES   = ("audio.synthesize", "voice.list")

    DEFAULT_MODELS_DIR = Path.home() / ".cache" / "termux-tts" / "models"
    DEFAULT_PID_FILE   = Path.home() / ".local" / "run" / "termux-tts.pid"

    def __init__(self, models_dir: Path | None = None) -> None:
        self._models_dir = models_dir or self.DEFAULT_MODELS_DIR
        self._state_file = ComponentStateFile(self.COMPONENT_ID)
        self._model_reg  = ModelRegistry(self.COMPONENT_ID)
        self._inst_reg   = InstanceRegistry(self.COMPONENT_ID)
        self._act_lock   = ActivationLock()
        # Phase 4: Heartbeat Writer
        from termux_tts.control.status import TTSStatusWriter
        self._heartbeat = TTSStatusWriter(self)

    def _get_version(self) -> str:
        try:
            from termux_tts import __version__; return __version__
        except Exception: return "1.1.2"

    def component_info(self) -> dict:
        info = ComponentInfo(
            protocol=PROTOCOL_COMPONENT, component_id=self.COMPONENT_ID,
            component_type=self.COMPONENT_TYPE, version=self._get_version(),
            capabilities=self.CAPABILITIES,
        )
        info.validate()
        return info.to_dict()

    def doctor_lite(self) -> dict:
        """
        경량 진단.
        12단계 Vulkan Doctor 금지 — doctor_full()에서만 호출.
        """
        ts = now_timestamps()
        state_data = self._state_file.read()
        stale = self._state_file.is_stale(threshold_ms=30_000)
        pid_info = self._check_pid()
        pid = pid_info.get("pid")
        pid_alive = pid_info.get("alive")
        instances = self._inst_reg.list_all()
        hot = [i for i in instances if i.state == InstanceState.HOT]

        # DSP 백엔드는 항상 가용 (Zero Dependency)
        dsp_available = True
        onnx_available = False
        try:
            from termux_tts.engine_onnx import ONNXNeuralEngine
            onnx_available = True
        except ImportError as _imp_err:
            import logging
            logging.getLogger(__name__).debug("[tts] ONNXNeuralEngine not importable: %s", _imp_err)

        ready = dsp_available  # DSP는 Zero Dependency이므로 항상 최소 가용
        degraded = stale or (pid_alive is not True)

        proc_dict: dict[str, Any] = {
            "running": pid_alive,
            "pid": pid,
            "verified": pid_info.get("verified", False),
        }
        if "inspection_error" in pid_info:
            proc_dict["inspection_error"] = pid_info["inspection_error"]
        if "reason" in pid_info:
            proc_dict["reason"] = pid_info["reason"]

        return {
            "protocol":       "ameva-component-status/1",
            "component_id":   self.COMPONENT_ID,
            "component_type": self.COMPONENT_TYPE,
            "version":        self._get_version(),
            "ready":          ready,
            "degraded":       degraded,
            **ts,
            "process":        proc_dict,
            "capabilities":   list(self.CAPABILITIES),
            "active_models":  [i.model_id for i in hot],
            "backends": {
                "dsp":    dsp_available,
                "onnx":   onnx_available,
                "native": None,  # Android 런타임에서만 확인 가능
            },
            "errors":   [state_data.get("last_error")] if state_data and state_data.get("last_error") else [],
            "state_file": {
                "path":       str(self._state_file.path),
                "stale":      stale,
                "updated_at": state_data.get("updated_at") if state_data else None,
            },
        }

    def _check_pid(self) -> dict[str, Any]:
        """BLOCKER 1: PID 파일 기반 프로세스 활성 여부 확인.
        PermissionError/OSError 발생 시 alive=None, verified=False, inspection_error 반환."""
        import logging
        _log = logging.getLogger(__name__)

        if self.DEFAULT_PID_FILE.exists():
            try:
                raw = self.DEFAULT_PID_FILE.read_text().strip()
                pid = int(raw)
            except (ValueError, OSError) as parse_err:
                _log.warning("[tts] PID file parse/read error: %s", parse_err)
                return {
                    "pid": None,
                    "alive": None,
                    "verified": False,
                    "inspection_error": {
                        "code": "PID_PARSE_ERROR",
                        "message": str(parse_err),
                    },
                }

            try:
                os.kill(pid, 0)
                return {"pid": pid, "alive": True, "verified": True}
            except ProcessLookupError:
                return {
                    "pid": pid,
                    "alive": False,
                    "verified": True,
                    "reason": "process_lookup_failed",
                }
            except PermissionError as perm_err:
                _log.warning("[tts] PID alive check PermissionError: %s", perm_err)
                return {
                    "pid": pid,
                    "alive": None,
                    "verified": False,
                    "inspection_error": {
                        "code": "PROCESS_INSPECTION_PERMISSION_DENIED",
                        "message": str(perm_err),
                    },
                }
            except OSError as os_err:
                _log.warning("[tts] PID alive check OSError: %s", os_err)
                return {
                    "pid": pid,
                    "alive": None,
                    "verified": False,
                    "inspection_error": {
                        "code": "PROCESS_INSPECTION_OS_ERROR",
                        "message": str(os_err),
                    },
                }

        state_data = self._state_file.read()
        if state_data:
            pid = state_data.get("process", {}).get("pid")
            if pid:
                try:
                    os.kill(pid, 0)
                    return {"pid": pid, "alive": True, "verified": True}
                except ProcessLookupError:
                    return {
                        "pid": pid,
                        "alive": False,
                        "verified": True,
                        "reason": "process_lookup_failed",
                    }
                except PermissionError as perm_err:
                    _log.warning("[tts] State-file PID %d PermissionError: %s", pid, perm_err)
                    return {
                        "pid": pid,
                        "alive": None,
                        "verified": False,
                        "inspection_error": {
                            "code": "PROCESS_INSPECTION_PERMISSION_DENIED",
                            "message": str(perm_err),
                        },
                    }
                except OSError as os_err:
                    _log.warning("[tts] State-file PID %d OSError: %s", pid, os_err)
                    return {
                        "pid": pid,
                        "alive": None,
                        "verified": False,
                        "inspection_error": {
                            "code": "PROCESS_INSPECTION_OS_ERROR",
                            "message": str(os_err),
                        },
                    }

        return {
            "pid": None,
            "alive": False,
            "verified": True,
            "reason": "pid_file_missing",
        }

    def doctor_full(self) -> dict:
        """기존 TTSEngine doctor() 호출 — 12단계 Vulkan Doctor 포함."""
        lite = self.doctor_lite()
        try:
            from termux_tts.engine import doctor
            doc_rep = doctor()
            lite["doctor"] = doc_rep
            lite["vulkan"] = doc_rep.get("doctor_report") or {"status": doc_rep.get("status")}
        except Exception as e:
            lite["vulkan_error"] = str(e)
        lite["doctor_level"] = "full"
        return lite

    def list_models(self) -> dict:
        """ModelRegistry 기반 + 설치된 파일 스캔."""
        reg_models = self._model_reg.list_all()
        reg_map = {m["model_id"]: m for m in reg_models}

        # models_dir 스캔 (파일만 존재 = unverified)
        if self._models_dir.exists():
            for p in self._models_dir.glob("*"):
                if p.is_file():
                    mid = p.stem
                    if mid not in reg_map:
                        reg_map[mid] = {
                            "model_id":    mid,
                            "state":       "unverified",
                            "format":      p.suffix.lstrip("."),
                            "note":        "File found on disk but not verified by AMEVA registry",
                            "verified_at": None,
                        }

        return {"models": list(reg_map.values()), "total": len(reg_map),
                "models_dir": str(self._models_dir)}

    def model_status(self, model_id: str | None = None) -> dict:
        if model_id:
            rec = self._model_reg.get(model_id)
            if rec is None: raise ModelNotFound(model_id)
            return {"model": rec}
        return self.list_models()

    def install_model(self, request: dict) -> dict:
        from ameva_component import ModelInstaller
        url = request.get("url", ""); filename = request.get("filename", "")
        sha256 = request.get("sha256", "")
        expected_bytes = int(request.get("expected_bytes", 0))
        model_id = request.get("model_id") or Path(filename).stem
        self._models_dir.mkdir(parents=True, exist_ok=True)
        installer = ModelInstaller(self.COMPONENT_ID, self._models_dir, self._model_reg)
        return installer.install(url=url, filename=filename, sha256=sha256,
                                 expected_bytes=expected_bytes, model_id=model_id)

    async def activate_model(self, request: dict) -> dict:
        model_id = request.get("model_id", "")
        voice_id = request.get("voice_id")  # Model/Voice 분리
        rec = self._model_reg.get(model_id)
        if rec is None: raise ModelNotFound(model_id)
        if ModelState.from_str(rec.get("state", "missing")) not in (ModelState.INSTALLED, ModelState.INACTIVE):
            raise ModelLoadFailed(model_id, f"State is '{rec.get('state')}'")
        with self._act_lock.acquire(timeout=60.0):
            self._model_reg.set_state(model_id, ModelState.ACTIVE)
            self._write_state()
        return {"activated": True, "model_id": model_id, "voice_id": voice_id,
                "rollback": {"attempted": False, "succeeded": False}}

    async def deactivate_model(self, request: dict) -> dict:
        model_id = request.get("model_id", "")
        self._model_reg.set_state(model_id, ModelState.INACTIVE)
        self._write_state()
        return {"deactivated": True, "model_id": model_id}

    def list_instances(self) -> dict:
        instances = self._inst_reg.list_all()
        return {"instances": [i.to_dict() for i in instances], "total": len(instances)}

    async def start_instance(self, request: dict) -> dict:
        backend = request.get("backend", "dsp")
        model_id = request.get("model_id", "dsp-default")
        instance_id = request.get("instance_id") or f"tts-{backend}-{int(time.time())}"
        inst = InstanceStatus(
            instance_id=instance_id, component_id=self.COMPONENT_ID,
            model_id=model_id, state=InstanceState.HOT,
            active_jobs=0, queue_depth=0, max_concurrency=4,
            backend=backend, started_at=time.time(), last_heartbeat=time.time(),
            last_error=None, control_mode=ControlMode.IN_PROCESS,
        )
        self._inst_reg.register(inst)
        self._write_state()
        # Phase 4: Heartbeat 시작 (Worker 시작 트리거)
        self._heartbeat.start()
        return {"instance_id": instance_id, "state": InstanceState.HOT.value, "backend": backend}

    async def drain_instance(self, instance_id: str) -> dict:
        from ameva_component import InstanceNotFound
        if not self._inst_reg.get(instance_id): raise InstanceNotFound(instance_id)
        self._inst_reg.update_state(instance_id, InstanceState.DRAINING)
        return {"instance_id": instance_id, "state": InstanceState.DRAINING.value}

    async def stop_instance(self, instance_id: str) -> dict:
        from ameva_component import InstanceNotFound
        if not self._inst_reg.get(instance_id): raise InstanceNotFound(instance_id)
        self._inst_reg.update_state(instance_id, InstanceState.STOPPED)
        self._inst_reg.remove(instance_id)
        # Phase 4: Heartbeat 중단 (정상 종료 트리거)
        remaining = self._inst_reg.list_all()
        if not remaining:
            self._heartbeat.stop()
        else:
            self._write_state()
        return {"instance_id": instance_id, "state": InstanceState.STOPPED.value}

    def _write_state(self, *, ready: bool | None = None, last_error: str | None = None) -> None:
        ts = now_timestamps()
        hot = [i for i in self._inst_reg.list_all() if i.state == InstanceState.HOT]
        _, pid_alive = self._check_pid()
        _ready = True if ready is None else ready  # DSP는 항상 최소 가용
        self._state_file.write({
            "protocol": "ameva-component-status/1", "component_id": self.COMPONENT_ID,
            "component_type": self.COMPONENT_TYPE, "version": self._get_version(),
            "ready": _ready, "degraded": not _ready, **ts,
            "active_models": [i.model_id for i in hot], "last_error": last_error,
        })

    async def synthesize(self, request: dict[str, Any]) -> dict[str, Any]:
        """AMEVA Component Protocol v1 synthesize method."""
        text = request.get("text", "").strip()
        if not text:
            return {
                "ok": False,
                "error": {
                    "code": "TEXT_EMPTY",
                    "message": "Input text cannot be empty",
                }
            }

        voice_id = request.get("voice_id")
        model_id = request.get("model_id")
        output_path = request.get("output_path")
        engine_type = request.get("engine", "auto")
        device = request.get("device", "auto")

        try:
            from termux_tts.engine import TTSEngine
            model_file = str(self._models_dir / model_id) if model_id else None
            engine = TTSEngine(
                model_path=model_file,
                engine_type=engine_type,
                device=device,
            )
            res = engine.synthesize(text, output=output_path)

            out_file = output_path
            if not out_file and hasattr(res, "save"):
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    out_file = f.name
                res.save(out_file)

            return {
                "ok": True,
                "audio_path": out_file or getattr(res, "output_path", None),
                "duration_sec": getattr(res, "duration_sec", 0.0),
                "sample_rate": getattr(res, "sample_rate", 22050),
                "model_id": getattr(res, "model_name", model_id or "default"),
                "backend": getattr(res, "backend", "unknown"),
                "fallback_used": False,
            }
        except Exception as e:
            return {
                "ok": False,
                "error": {
                    "code": "SYNTHESIZE_FAILED",
                    "message": str(e),
                }
            }
