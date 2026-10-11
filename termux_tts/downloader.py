# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
AMEVA Unified Model Downloader for termux-tts.
"""
from pathlib import Path
from typing import Optional, List, Dict, Any
from .hardware import get_unified_model_search_dirs

AVAILABLE_MODELS = {
    "vits-piper-ko": {"name": "vits-piper-ko.onnx", "lang": "ko", "size_mb": 65, "desc": "Korean Piper VITS"},
    "melo-tts-ko": {"name": "melo-ko", "lang": "ko", "size_mb": 120, "desc": "Korean MeloTTS"},
    "vits-piper-en": {"name": "vits-piper-en.onnx", "lang": "en", "size_mb": 60, "desc": "English Piper VITS"},
}

def resolve_model_path(model_name: str = "vits-piper-ko") -> Path:
    search_dirs = get_unified_model_search_dirs("tts")
    for d in search_dirs:
        for cand in [d / model_name, d / f"{model_name}.onnx", d / f"{model_name}.tar.gz"]:
            if cand.exists():
                return cand.resolve()
    from .installer import get_install_paths
    paths = get_install_paths()
    return Path(paths.get("models_dir", search_dirs[0])) / model_name

def download_model(model_name: str = "vits-piper-ko", output_dir: Optional[Path] = None, force: bool = False) -> Path:
    from .installer import provision_neural_model_archive
    target_id = "ko" if "ko" in model_name else ("melo" if "melo" in model_name else "en")
    return Path(provision_neural_model_archive(target_id, force=force))

def list_models() -> List[Dict[str, Any]]:
    return [{"id": k, **v} for k, v in AVAILABLE_MODELS.items()]
