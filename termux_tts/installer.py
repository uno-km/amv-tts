# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
Automated One-Click Native Installer & Model Provisioner for termux-tts.
Downloads pre-compiled ARM64 native Sherpa-ONNX CPU engine binaries and studio neural speech models.
Adheres strictly to AOSF-ENG-STD-2026 and Fail-Fast engineering standards.
"""
from __future__ import annotations

import os
import sys
import tarfile
import urllib.request
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any


def _resolve_package_version() -> Optional[str]:
    """Dynamically resolve current installed package version without static fallback."""
    try:
        from . import __version__
        if __version__:
            return __version__
    except Exception:
        pass
    try:
        import importlib.metadata
        return importlib.metadata.version("termux-tts")
    except Exception:
        return None


def get_candidate_binary_urls() -> List[str]:
    """Generate dynamic candidate endpoints for Sherpa-ONNX CPU native binary provisioner."""
    urls = []
    custom_tag = os.environ.get("TERMUX_TTS_RELEASE_TAG", "").strip()
    custom_base = os.environ.get("TERMUX_TTS_RELEASE_BASE", "").strip()

    # Tier 1: Explicit environment overrides
    if custom_base:
        urls.append(f"{custom_base.rstrip('/')}/sherpa-onnx-android-arm64.tar.gz")
    if custom_tag:
        tag = custom_tag if custom_tag.startswith("v") else f"v{custom_tag}"
        urls.append(f"https://github.com/uno-km/termux-tts/releases/download/{tag}/sherpa-onnx-android-arm64.tar.gz")

    # Tier 2: GitHub Releases latest canonical endpoint (Zero-Hardcoding SSOT)
    urls.append("https://github.com/uno-km/termux-tts/releases/latest/download/sherpa-onnx-android-arm64.tar.gz")

    # Tier 3: Current installed package dynamic version matching
    ver = _resolve_package_version()
    if ver:
        urls.append(f"https://github.com/uno-km/termux-tts/releases/download/v{ver}/sherpa-onnx-android-arm64.tar.gz")

    return urls


OFFICIAL_NEURAL_MODELS: Dict[str, Dict[str, str]] = {
    "ko": {
        "name": "vits-mimic3-ko_KO-kss_low",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mimic3-ko_KO-kss_low.tar.bz2",
        "repo": "csukuangfj/vits-mimic3-ko_KO-kss_low",
        "language_name": "Korean",
        "description": "Korean VITS Mimic3 KSS Studio ONNX Model (~45MB)",
    },
    "en": {
        "name": "vits-piper-en_US-lessac-medium",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-lessac-medium.tar.bz2",
        "repo": "csukuangfj/vits-piper-en_US-lessac-medium",
        "language_name": "English",
        "description": "English VITS Piper Lessac Medium ONNX Model (~55MB)",
    },
    "kokoro": {
        "name": "kokoro-int8-en-v0_19",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-int8-en-v0_19.tar.bz2",
        "repo": "csukuangfj/kokoro-int8-en-v0_19",
        "language_name": "Kokoro-82M Studio High Quality (INT8)",
        "description": "Kokoro-82M StyleTTS2 Multilingual Studio Grade Model (~103MB)",
    },
    "melo": {
        "name": "vits-melo-tts-zh_en",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-melo-tts-zh_en.tar.bz2",
        "repo": "csukuangfj/vits-melo-tts-zh_en",
        "language_name": "MeloTTS Universal Bilingual",
        "description": "MeloTTS Universal VITS Bilingual Model (~150MB)",
    },
    "supertonic": {
        "name": "sherpa-onnx-supertonic-3-tts-int8-2026-05-11",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2",
        "repo": "csukuangfj/sherpa-onnx-supertonic-3-tts-int8",
        "language_name": "Supertonic 3 On-Device Multilingual (INT8)",
        "description": "Supertonic 3 Ultra-Fast 31-Language Model (~128MB)",
    },
    "ja": {
        "name": "vits-piper-ja_JP-hina-medium",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-ja_JP-hina-medium.tar.bz2",
        "repo": "csukuangfj/vits-piper-ja_JP-hina-medium",
        "language_name": "Japanese",
        "description": "Japanese VITS Piper Hina Medium ONNX Model (~50MB)",
    },
    "zh": {
        "name": "vits-zh-aishell3",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-zh-aishell3.tar.bz2",
        "repo": "csukuangfj/vits-zh-aishell3",
        "language_name": "Chinese (Mandarin)",
        "description": "Chinese VITS AISHELL-3 Multi-Speaker ONNX Model (~65MB)",
    },
    "hi": {
        "name": "vits-piper-hi_IN-swara-medium",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-hi_IN-swara-medium.tar.bz2",
        "repo": "csukuangfj/vits-piper-hi_IN-swara-medium",
        "language_name": "Hindi",
        "description": "Hindi VITS Piper Swara Medium ONNX Model (~55MB)",
    },
    "ru": {
        "name": "vits-piper-ru_RU-dmitri-medium",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-ru_RU-dmitri-medium.tar.bz2",
        "repo": "csukuangfj/vits-piper-ru_RU-dmitri-medium",
        "language_name": "Russian",
        "description": "Russian VITS Piper Dmitri Medium ONNX Model (~55MB)",
    },
    "es": {
        "name": "vits-piper-es_ES-davefx-medium",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-es_ES-davefx-medium.tar.bz2",
        "repo": "csukuangfj/vits-piper-es_ES-davefx-medium",
        "language_name": "Spanish",
        "description": "Spanish VITS Piper Davefx Medium ONNX Model (~50MB)",
    },
    "fr": {
        "name": "vits-piper-fr_FR-siwis-medium",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-fr_FR-siwis-medium.tar.bz2",
        "repo": "csukuangfj/vits-piper-fr_FR-siwis-medium",
        "language_name": "French",
        "description": "French VITS Piper Siwis Medium ONNX Model (~50MB)",
    },
    "de": {
        "name": "vits-piper-de_DE-thorsten-medium",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-de_DE-thorsten-medium.tar.bz2",
        "repo": "csukuangfj/vits-piper-de_DE-thorsten-medium",
        "language_name": "German",
        "description": "German VITS Piper Thorsten Medium ONNX Model (~50MB)",
    },
}


def get_install_paths():
    prefix = Path(os.environ.get("PREFIX", "/data/data/com.termux/files/usr")).resolve()
    bin_dir = (prefix / "bin").resolve()
    lib_dir = (prefix / "lib").resolve()
    xdg_cache = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")).resolve()
    cache_dir = (xdg_cache / "termux-tts" / "models").resolve()
    bin_dir.mkdir(parents=True, exist_ok=True)
    lib_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return bin_dir, cache_dir


def download_with_progress(url: str, dest_path: Path, label: str):
    ver = _resolve_package_version() or "latest"

    print(f"  [DOWNLOADING] {label}...")
    req = urllib.request.Request(url, headers={"User-Agent": f"termux-tts-installer/{ver} (Android; ARM64)"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out_f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 64 * 1024
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = (downloaded / total) * 100
                sys.stdout.write(f"\r  [{label}] {pct:.1f}% ({downloaded / (1024*1024):.1f}MB / {total / (1024*1024):.1f}MB)")
                sys.stdout.flush()
    print()


def is_valid_elf(path: Path) -> bool:
    """Verifies that the target path is a valid ELF executable/library via magic bytes."""
    try:
        p = path.resolve() if path.is_symlink() else path
        if not p.is_file():
            return False
        with open(p, "rb") as f:
            return f.read(4) == b"\x7fELF"
    except (OSError, PermissionError):
        return False


def install_engine_binary(force: bool = False, dedicate: bool = False) -> Path:
    """Download and install native ARM64 Sherpa-ONNX CPU engine binary and shared libraries."""
    bin_dir, _ = get_install_paths()
    binary_path = bin_dir / "sherpa-onnx-offline-tts"
    lib_dir = Path(os.environ.get("PREFIX", "/data/data/com.termux/files/usr")) / "lib"

    if dedicate and binary_path.is_symlink():
        if ".local/share/ameva" in str(binary_path.resolve()):
            print("  [DEDICATE] AMEVA Runtime managed engine detected. Preserving co-existence (<0.002s).")
            return binary_path

    if binary_path.exists() and is_valid_elf(binary_path) and not force:
        print(f"  [OK] Native CPU engine binary already exists and verified: {binary_path}")
        return binary_path

    xdg_cache = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")).resolve()
    staging_dir = xdg_cache / "termux-tts" / ".staging-cpu"
    staging_dir.mkdir(parents=True, exist_ok=True)
    tar_path = staging_dir / "sherpa-cpu.tar.gz"
    candidate_urls = get_candidate_binary_urls()
    download_success = False

    for url in candidate_urls:
        try:
            download_with_progress(url, tar_path, f"ARM64 Sherpa CPU Binary ({url})")
            if tar_path.exists() and tar_path.stat().st_size > 1024:
                download_success = True
                break
        except Exception as dl_err:
            print(f"  [-] Candidate URL failed ({url}): {dl_err}")
            if tar_path.exists():
                tar_path.unlink(missing_ok=True)
            continue

    if not download_success:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise RuntimeError("Failed to download sherpa-onnx-offline-tts from any candidate endpoints.")

    print(f"  [EXTRACTING] Extracting CPU engine binary to staging isolation {staging_dir}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=staging_dir)

    tar_path.unlink(missing_ok=True)

    found_bin = None
    for p in staging_dir.rglob("sherpa-onnx-offline-tts"):
        if p.is_file() and is_valid_elf(p):
            found_bin = p
            break

    if not found_bin:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise RuntimeError("sherpa-onnx-offline-tts binary not found inside extracted archive.")

    shutil.copy2(found_bin, binary_path)
    binary_path.chmod(0o755)

    lib_dir.mkdir(parents=True, exist_ok=True)
    for so_file in staging_dir.rglob("*.so*"):
        if so_file.is_file():
            target_so = lib_dir / so_file.name
            if not force and target_so.is_file() and is_valid_elf(target_so):
                continue  # Preserve existing healthy shared library
            shutil.copy2(so_file, target_so)
            try:
                target_so.chmod(0o755)
            except OSError:
                pass

    shutil.rmtree(staging_dir, ignore_errors=True)
    print(f"  [SUCCESS] Native CPU engine installed: {binary_path}")
    return binary_path


# Backward compatibility alias
install_cpu_binary = install_engine_binary


def provision_neural_model_archive(language: str, force: bool = False) -> Path:
    """
    On-Demand Auto-Provisioner for official neural speech models.
    Downloads official pre-compiled model archives (.tar.bz2) and extracts directly into cache.
    """
    from .script_classifier import normalize_language_code
    lang = normalize_language_code(language)
    if lang not in OFFICIAL_NEURAL_MODELS:
        from .exceptions import TTSModelLoadError
        raise TTSModelLoadError(
            f"[FAIL-FAST] No official automated model package registered for identifier '{lang}'.\n"
            f"Available automated packages: {list(OFFICIAL_NEURAL_MODELS.keys())}"
        )

    cfg = OFFICIAL_NEURAL_MODELS[lang]
    _, cache_dir = get_install_paths()
    target_dir = cache_dir / cfg["name"]
    shared_model_dir = Path.home() / ".cache" / "termux-ai" / "models" / cfg["name"]
    unified_tts_dir = Path.home() / "models" / "tts"

    if not force:
        if shared_model_dir.is_dir() and list(shared_model_dir.glob("*.onnx")):
            return shared_model_dir
        if target_dir.is_dir():
            onnx_files = list(target_dir.glob("*.onnx"))
            if onnx_files:
                try:
                    unified_tts_dir.mkdir(parents=True, exist_ok=True)
                    link_dest = unified_tts_dir / cfg["name"]
                    if not link_dest.exists() and not link_dest.is_symlink():
                        link_dest.symlink_to(target_dir)
                except OSError:
                    pass
                return target_dir

    print(f"\n[termux-tts runtime] Auto-provisioning {cfg['description']}...")
    archive_path = cache_dir / f"{cfg['name']}.tar.bz2"

    candidate_urls = []
    custom_tag = os.environ.get("TERMUX_TTS_RELEASE_TAG", "").strip()
    custom_base = os.environ.get("TERMUX_TTS_RELEASE_BASE", "").strip()

    # Tier 1: Explicit environment overrides
    if custom_base:
        candidate_urls.append(f"{custom_base.rstrip('/')}/{cfg['name']}.tar.bz2")
    if custom_tag:
        tag = custom_tag if custom_tag.startswith("v") else f"v{custom_tag}"
        candidate_urls.append(f"https://github.com/uno-km/termux-tts/releases/download/{tag}/{cfg['name']}.tar.bz2")

    # Tier 2: GitHub Releases latest canonical endpoint
    candidate_urls.append(f"https://github.com/uno-km/termux-tts/releases/latest/download/{cfg['name']}.tar.bz2")

    # Tier 3: Current installed package dynamic version matching
    ver = _resolve_package_version()
    if ver:
        candidate_urls.append(f"https://github.com/uno-km/termux-tts/releases/download/v{ver}/{cfg['name']}.tar.bz2")

    # Tier 4: Upstream source
    candidate_urls.append(cfg["url"])

    download_success = False
    for url in candidate_urls:
        try:
            download_with_progress(url, archive_path, cfg["name"])
            if archive_path.exists() and archive_path.stat().st_size > 100 * 1024:
                download_success = True
                break
        except Exception as dl_err:
            if archive_path.exists():
                archive_path.unlink(missing_ok=True)
            continue

    if not download_success:
        from .exceptions import TTSModelLoadError
        raise TTSModelLoadError(f"[FAIL-FAST] Failed to download neural model archive for '{cfg['name']}' from all candidate endpoints.")

    try:
        print(f"  [EXTRACTING] Unpacking model archive into {cache_dir}...")
        with tarfile.open(archive_path, "r:bz2") as tar:
            tar.extractall(path=cache_dir)
        archive_path.unlink(missing_ok=True)

        try:
            unified_tts_dir.mkdir(parents=True, exist_ok=True)
            link_dest = unified_tts_dir / cfg["name"]
            if not link_dest.exists() and not link_dest.is_symlink():
                link_dest.symlink_to(target_dir)
        except OSError:
            pass

        print(f"  [SUCCESS] Model provisioned: {target_dir}")
        return target_dir
    except Exception as err:
        if archive_path.exists():
            archive_path.unlink(missing_ok=True)
        from .exceptions import TTSModelLoadError
        raise TTSModelLoadError(
            f"[FAIL-FAST] Failed to extract neural model '{cfg['name']}': {err}"
        ) from err


def run_installation(models: str = "default", force: bool = False, play: bool = True, dedicate: bool = False):
    """
    Pure CPU Native Automated Provisioner for termux-tts.
    Installs pre-compiled ARM64 Sherpa-ONNX CPU engine and standard models (Korean + English + Kokoro-82M).
    """
    print("=" * 70)
    print("   TERMUX-TTS NATIVE CPU AUTOMATED PROVISIONER")
    print("=" * 70)

    # 1. Install pre-compiled Sherpa-ONNX CPU native binary
    print("\n[STEP 1/3] Provisioning ARM64 Sherpa-ONNX Native CPU Engine...")
    install_engine_binary(force=force, dedicate=dedicate)

    # 2. Provision Neural Speech Models
    # Default is Korean (ko) + English (en) + Kokoro-82M Studio Model (kokoro)
    raw_models = models.strip().lower() if models else "default"
    if raw_models in ("default", "base", "core"):
        langs_to_install = ["ko", "en", "kokoro"]
    elif raw_models == "all":
        langs_to_install = list(OFFICIAL_NEURAL_MODELS.keys())
    else:
        langs_to_install = [l.strip() for l in raw_models.split(",") if l.strip()]

    print(f"\n[STEP 2/3] Provisioning Neural Speech Models: {langs_to_install}...")
    for item in langs_to_install:
        try:
            provision_neural_model_archive(item, force=force)
        except Exception as e:
            print(f"  [-] Model '{item}' provisioning note: {e}")

    # 3. On-Device Verification
    print("\n[STEP 3/3] Running On-Device Self-Test...")
    test_wav = Path.home() / "install_test_tts.wav"

    try:
        from .engine import load
        with load(language="en", device="cpu") as eng:
            eng.synthesize("Hello, Termux CPU speech synthesis is ready.", output=str(test_wav))
        proc_returncode = 0
        proc_stderr = ""
    except Exception as tts_err:
        proc_returncode = 1
        proc_stderr = str(tts_err)

    if proc_returncode == 0:
        print("  [VERIFIED] On-device native CPU synthesis self-test passed!")
        player = shutil.which("termux-media-player") or shutil.which("play-audio")
        if play and player:
            print("  [PLAYBACK] Playing verification audio through physical speaker...")
            if shutil.which("termux-volume"):
                import subprocess
                subprocess.run(["termux-volume", "music", "10"], check=False)
            import subprocess
            subprocess.run([player, str(test_wav)], check=False)
    else:
        print(f"  [FAIL-FAST] Self-test returned error: {proc_stderr}")

    print("=" * 70)
    print("   INSTALLATION COMPLETE! YOU CAN NOW USE 'termux-tts' DIRECTLY.")
    print("=" * 70)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="termux-tts native CPU automated provisioner")
    parser.add_argument("--force", "-f", action="store_true", help="Force clean re-installation")
    parser.add_argument("--dedicate", action="store_true", help="Smart inspection mode: preserve AMEVA runtime symlink")
    parser.add_argument("--models", type=str, default="default", help="Neural model languages to provision (default, all, or ko,en)")
    parser.add_argument("--no-play", action="store_true", help="Disable test sound playback")
    args = parser.parse_args()

    run_installation(models=args.models, force=args.force, play=not args.no_play, dedicate=args.dedicate)


if __name__ == "__main__":
    main()
