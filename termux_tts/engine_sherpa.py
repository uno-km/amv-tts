# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
Sherpa-ONNX C++ Native Subprocess Inference Engine for termux-tts (Tier 3-Neural & Tier 4-Expressive).
Directly executes high-performance C++ sherpa-onnx binaries for VITS, Kokoro, and Matcha models
with zero Python onnxruntime dependency, strict crash isolation, and full fail-fast compliance.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

from .audio import AudioBuffer
from .exceptions import TTSModelLoadError, TTSInferenceError
from .hardware import get_clean_execution_env
from .cluster import parse_cluster_rpc_spec, verify_rpc_cluster_nodes

logger = logging.getLogger("termux_tts.engine_sherpa")


@dataclass
class SherpaResult:
    text: str
    audio_buffer: AudioBuffer
    sample_rate: int
    duration_sec: float
    elapsed_ms: float
    rtf: float
    model_name: str
    backend: str
    device_model: str

    def save(self, filepath: str) -> str:
        return self.audio_buffer.save(filepath)

    @property
    def wav_bytes(self) -> bytes:
        return self.audio_buffer.to_wav_bytes()


class SherpaNeuralEngine:
    """Production C++ Subprocess-Isolated Neural Speech Synthesis Engine."""

    CANDIDATE_BINARIES = [
        str(Path.home() / ".local" / "share" / "ameva" / "rpc_llama" / "bin" / "sherpa-onnx-offline-tts"),
        str(Path.home() / ".local" / "share" / "ameva" / "rpc_llama" / "bin" / "llama-cli"),
        "sherpa-onnx-offline-tts",
        str(Path.home() / ".local" / "bin" / "sherpa-onnx-offline-tts"),
        str(Path.home() / "sherpa-onnx-offline-tts"),
        "/data/data/com.termux/files/home/sherpa-onnx-offline-tts",
        "/data/data/com.termux/files/home/.local/bin/sherpa-onnx-offline-tts",
        "/data/data/com.termux/files/usr/bin/sherpa-onnx-offline-tts",
    ]

    STANDARD_MODEL_DIRS = [
        Path.home() / ".cache" / "termux-tts" / "models",
        Path.home() / "vits-mimic3-ko_KO-kss_low",
        Path("/data/data/com.termux/files/home/vits-mimic3-ko_KO-kss_low"),
        Path("/data/data/com.termux/files/home/.cache/termux-tts/models"),
    ]

    def __init__(
        self,
        model_path: Optional[str] = None,
        language: str = "ko",
        device: str = "auto",
        threads: int = 4,
        sample_rate: int = 22050,
        model_type: str = "vits",
        rpc: Optional[str] = None,
        tensor_split: Optional[str] = None,
        cluster_rpc_servers: Optional[Any] = None,
        cluster_split_mode: Optional[str] = None,
        cluster_tensor_split: Optional[str] = None,
        cluster_vram_budget: Optional[dict] = None,
    ):
        self.language = language.lower()
        self.requested_device = device.lower()
        self.threads = threads
        self.sample_rate = sample_rate
        self.model_type = model_type.lower()
        self.cluster_rpc_servers = parse_cluster_rpc_spec(cluster_rpc_servers or rpc)
        self.rpc = ",".join(self.cluster_rpc_servers) if self.cluster_rpc_servers else None
        self.cluster_split_mode = cluster_split_mode
        self.cluster_tensor_split = cluster_tensor_split or tensor_split
        self.tensor_split = self.cluster_tensor_split
        self.cluster_vram_budget = cluster_vram_budget
        self._is_closed = False

        # 1. Locate C++ binary
        self.binary = self._find_binary()

        # 2. Resolve Model assets (Fail-Fast)
        self.model_assets = self._resolve_model_assets(model_path)
        self.model_name = self.model_assets["model_name"]
        self.backend = f"SHERPA_{self.model_type.upper()}_ARM64"

    def _find_binary(self) -> str:
        """Locate sherpa-onnx-offline-tts executable or fail-fast."""
        for candidate in self.CANDIDATE_BINARIES:
            found = shutil.which(candidate) if not os.path.isabs(candidate) else candidate
            if found and os.path.isfile(found) and os.access(found, os.X_OK):
                return str(found)
        raise TTSModelLoadError(
            "[FAIL-FAST] 'sherpa-onnx-offline-tts' binary not found. "
            "Please ensure sherpa-onnx is installed in PATH or '~/.local/bin/sherpa-onnx-offline-tts'."
        )

    def _resolve_model_assets(self, model_path: Optional[str]) -> Dict[str, str]:
        """Locate model onnx and auxiliary metadata assets or fail-fast."""
        search_dirs: List[Path] = []
        if model_path:
            p = Path(model_path).expanduser().resolve()
            if not p.exists():
                raise TTSModelLoadError(
                    f"[FAIL-FAST] Explicitly specified model path does not exist: '{model_path}'"
                )
            if p.is_dir():
                search_dirs = [p]
            elif p.is_file():
                search_dirs = [p.parent]
        else:
            from .hardware import get_unified_model_search_dirs
            search_dirs = list(get_unified_model_search_dirs("tts"))
            search_dirs.extend([Path.home(), Path("/data/data/com.termux/files/home")])
            search_dirs.extend(self.STANDARD_MODEL_DIRS)

        # 1. Kokoro Model Search
        if self.model_type == "kokoro":
            for sdir in search_dirs:
                if not sdir.exists():
                    continue
                candidates = [sdir] + [sdir / name for name in ["kokoro-int8-en-v0_19", "kokoro-en-v0_19", "kokoro-int8-multi-lang-v1_1", "kokoro-multi-lang-v1_0", "kokoro"]]
                for cand in candidates:
                    if not cand.is_dir():
                        continue
                    voices = cand / "voices.bin"
                    tokens = cand / "tokens.txt"
                    data_dir = cand / "espeak-ng-data"
                    onnx_files = list(cand.glob("*.onnx"))
                    if onnx_files and voices.exists() and tokens.exists():
                        return {
                            "model": str(onnx_files[0]),
                            "voices": str(voices),
                            "tokens": str(tokens),
                            "data_dir": str(data_dir) if data_dir.exists() else "",
                            "model_name": cand.name,
                        }
            raise TTSModelLoadError(
                f"[FAIL-FAST] Kokoro model assets (model.onnx, voices.bin, tokens.txt) not found.\n"
                f"  Run 'termux-tts install --models kokoro' to download and provision Kokoro-82M."
            )

        # 2. Supertonic Model Search
        if self.model_type == "supertonic":
            for sdir in search_dirs:
                if not sdir.exists():
                    continue
                candidates = [sdir] + [sdir / name for name in ["sherpa-onnx-supertonic-3-tts-int8-2026-05-11", "sherpa-onnx-supertonic-tts-int8-2026-03-06", "supertonic"]]
                for cand in candidates:
                    if not cand.is_dir():
                        continue
                    dp = cand / "duration_predictor.int8.onnx" if (cand / "duration_predictor.int8.onnx").exists() else cand / "duration_predictor.onnx"
                    te = cand / "text_encoder.int8.onnx" if (cand / "text_encoder.int8.onnx").exists() else cand / "text_encoder.onnx"
                    ve = cand / "vector_estimator.int8.onnx" if (cand / "vector_estimator.int8.onnx").exists() else cand / "vector_estimator.onnx"
                    voc = cand / "vocoder.int8.onnx" if (cand / "vocoder.int8.onnx").exists() else cand / "vocoder.onnx"
                    tts_json = cand / "tts.json"
                    ui = cand / "unicode_indexer.bin"
                    vs = cand / "voice_styles.bin"
                    if not vs.exists():
                        vs = cand / "voice.bin"
                    if dp.exists() and te.exists() and ve.exists() and voc.exists() and tts_json.exists() and ui.exists() and vs.exists():
                        return {
                            "duration_predictor": str(dp),
                            "text_encoder": str(te),
                            "vector_estimator": str(ve),
                            "vocoder": str(voc),
                            "tts_json": str(tts_json),
                            "unicode_indexer": str(ui),
                            "voice_style": str(vs),
                            "model_name": cand.name,
                        }
            raise TTSModelLoadError(
                f"[FAIL-FAST] Supertonic model assets (duration_predictor, text_encoder, vocoder, tts.json) not found.\n"
                f"  Run 'termux-tts install --models supertonic' to download and provision Supertonic."
            )

        # 3. MeloTTS Model Search
        if self.model_type == "melo":
            for sdir in search_dirs:
                if not sdir.exists():
                    continue
                candidates = [sdir] + [sdir / name for name in ["vits-melo-tts-zh_en", "melo-tts", "melotts"]]
                for cand in candidates:
                    if not cand.is_dir():
                        continue
                    tokens = cand / "tokens.txt"
                    lexicon = cand / "lexicon.txt"
                    onnx_files = list(cand.glob("*.onnx"))
                    if onnx_files and tokens.exists() and lexicon.exists():
                        return {
                            "model": str(onnx_files[0]),
                            "tokens": str(tokens),
                            "lexicon": str(lexicon),
                            "model_name": cand.name,
                        }
            raise TTSModelLoadError(
                f"[FAIL-FAST] MeloTTS model assets (model.onnx, tokens.txt, lexicon.txt) not found.\n"
                f"  Run 'termux-tts install --models melo' to download and provision MeloTTS."
            )

        # 4. Standard VITS ONNX model search
        from .script_classifier import normalize_language_code
        norm_lang = normalize_language_code(self.language)

        for sdir in search_dirs:
            if not sdir.exists():
                continue

            chosen_dir = sdir
            onnx_files = list(sdir.glob("*.onnx"))

            # If sdir does not directly contain .onnx, search language-specific subdirectories
            if not onnx_files:
                target_subdirs = []
                from .installer import OFFICIAL_NEURAL_MODELS
                if norm_lang in OFFICIAL_NEURAL_MODELS:
                    target_subdirs.append(OFFICIAL_NEURAL_MODELS[norm_lang]["name"])

                if norm_lang == "ko":
                    target_subdirs.extend(["vits-mimic3-ko_KO-kss_low", "vits-ko-kss", "vits-mimic3-ko"])
                elif norm_lang == "en":
                    target_subdirs.extend(["vits-piper-en_US-lessac-medium", "vits-piper-en_US-lessac-high", "vits-piper-en_US-amy-medium", "vits-en-lessac"])
                elif norm_lang == "ja":
                    target_subdirs.extend(["vits-piper-ja_JP-hina-medium", "vits-ja-hina"])
                elif norm_lang == "zh":
                    target_subdirs.extend(["vits-zh-aishell3", "vits-zh"])

                for sub in target_subdirs:
                    cand = sdir / sub
                    if cand.is_dir() and list(cand.glob("*.onnx")):
                        chosen_dir = cand
                        onnx_files = list(cand.glob("*.onnx"))
                        break

                # Fallback: scan any subdirectory matching language tag
                if not onnx_files:
                    for sub in sdir.iterdir():
                        if sub.is_dir() and norm_lang in sub.name.lower():
                            cand_onnx = list(sub.glob("*.onnx"))
                            if cand_onnx:
                                chosen_dir = sub
                                onnx_files = cand_onnx
                                break

            if onnx_files:
                onnx_model = str(onnx_files[0])
                tokens_file = chosen_dir / "tokens.txt"
                espeak_dir = chosen_dir / "espeak-ng-data"
                lexicon_file = chosen_dir / "lexicon.txt"

                if not tokens_file.exists():
                    tokens_file = chosen_dir.parent / "tokens.txt"
                if not espeak_dir.exists():
                    espeak_dir = chosen_dir.parent / "espeak-ng-data"
                if not lexicon_file.exists():
                    lexicon_file = chosen_dir.parent / "lexicon.txt"

                if tokens_file.exists() and (espeak_dir.exists() or lexicon_file.exists()):
                    res = {
                        "model": str(onnx_model),
                        "tokens": str(tokens_file),
                        "model_name": Path(onnx_model).stem,
                    }
                    if espeak_dir.exists():
                        res["data_dir"] = str(espeak_dir)
                    if lexicon_file.exists():
                        res["lexicon"] = str(lexicon_file)
                    return res

        raise TTSModelLoadError(
            f"[FAIL-FAST] Required TTS model assets (onnx model, tokens.txt, espeak-ng-data) "
            f"not found for language '{self.language}' in provided path '{model_path}' or standard locations: "
            f"{[str(d) for d in self.STANDARD_MODEL_DIRS]}. "
            f"Please deploy the model package (e.g. 'vits-mimic3-ko_KO-kss_low') before synthesis."
        )

    def synthesize(
        self,
        text: str,
        output: Optional[str] = None,
        speed: float = 1.0,
        preset: Optional[str] = None,
        speaker_id: Optional[int] = None,
        speaker_profile: Optional[Union[Any, str]] = None,
        profile: Optional[Union[Any, str]] = None,
    ) -> SherpaResult:
        """Synthesize natural speech audio from text using C++ isolated subprocess."""
        if self._is_closed:
            raise TTSInferenceError("Cannot synthesize: Engine session is closed.")

        clean_text = text.strip()
        if not clean_text:
            raise TTSInferenceError("Cannot synthesize empty or whitespace-only text.")

        # Acoustic prosody fix: Triple dots '...' in Korean cause espeak-ng glottal elision.
        # Replacing with comma preserves initial vowels and rhythm.
        normalized_text = clean_text.replace("...", ", ").replace("…", ", ")

        t0 = time.perf_counter()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            temp_wav = tmp_file.name

        try:
            if self.model_type == "kokoro":
                cmd = [
                    self.binary,
                    f"--kokoro-model={self.model_assets['model']}",
                    f"--kokoro-voices={self.model_assets['voices']}",
                    f"--kokoro-tokens={self.model_assets['tokens']}",
                    f"--output-filename={temp_wav}",
                    f"--num-threads={self.threads}",
                    f"--speed={speed:.2f}",
                ]
                if self.model_assets.get("data_dir"):
                    cmd.append(f"--kokoro-data-dir={self.model_assets['data_dir']}")
                cmd.append(normalized_text)
            elif self.model_type == "supertonic":
                cmd = [
                    self.binary,
                    f"--supertonic-duration-predictor={self.model_assets['duration_predictor']}",
                    f"--supertonic-text-encoder={self.model_assets['text_encoder']}",
                    f"--supertonic-vector-estimator={self.model_assets['vector_estimator']}",
                    f"--supertonic-vocoder={self.model_assets['vocoder']}",
                    f"--supertonic-tts-json={self.model_assets['tts_json']}",
                    f"--supertonic-unicode-indexer={self.model_assets['unicode_indexer']}",
                    f"--supertonic-voice-style={self.model_assets['voice_style']}",
                    f"--lang={self.language if self.language not in ('auto', '') else 'ko'}",
                    f"--output-filename={temp_wav}",
                    f"--num-threads={self.threads}",
                    f"--speed={speed:.2f}",
                    normalized_text,
                ]
            elif self.model_type == "melo":
                cmd = [
                    self.binary,
                    f"--vits-model={self.model_assets['model']}",
                    f"--vits-tokens={self.model_assets['tokens']}",
                    f"--vits-lexicon={self.model_assets['lexicon']}",
                    f"--output-filename={temp_wav}",
                    f"--num-threads={self.threads}",
                    f"--speed={speed:.2f}",
                ]
                if speaker_id is not None:
                    cmd.append(f"--sid={speaker_id}")
                cmd.append(normalized_text)
            else:
                cmd = [
                    self.binary,
                    f"--vits-model={self.model_assets['model']}",
                    f"--vits-tokens={self.model_assets['tokens']}",
                    f"--output-filename={temp_wav}",
                    f"--num-threads={self.threads}",
                    f"--speed={speed:.2f}",
                ]
                if self.model_assets.get("data_dir"):
                    cmd.append(f"--vits-data-dir={self.model_assets['data_dir']}")
                elif self.model_assets.get("lexicon"):
                    cmd.append(f"--vits-lexicon={self.model_assets['lexicon']}")
                if speaker_id is not None:
                    cmd.append(f"--sid={speaker_id}")
                cmd.append(normalized_text)

            # Cluster Distributed Inference Routing & Pre-Flight Verification
            if self.cluster_rpc_servers:
                verify_rpc_cluster_nodes(self.cluster_rpc_servers)
                cmd.extend(["--rpc", ",".join(self.cluster_rpc_servers)])
            elif self.rpc:
                parsed_servers = parse_cluster_rpc_spec(self.rpc)
                if parsed_servers:
                    verify_rpc_cluster_nodes(parsed_servers)
                    cmd.extend(["--rpc", ",".join(parsed_servers)])

            if self.tensor_split:
                cmd.extend(["--tensor-split", str(self.tensor_split).strip()])

            env = get_clean_execution_env({
                "PYTHONIOENCODING": "utf-8",
                "LANG": "en_US.UTF-8",
                "LC_ALL": "en_US.UTF-8",
            })
            rpc_lib_dir = Path.home() / ".local" / "share" / "ameva" / "rpc_llama" / "bin"
            if rpc_lib_dir.is_dir():
                cur_ld = env.get("LD_LIBRARY_PATH", "")
                if str(rpc_lib_dir) not in cur_ld:
                    env["LD_LIBRARY_PATH"] = f"{rpc_lib_dir}:{cur_ld}" if cur_ld else str(rpc_lib_dir)

            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=60,
            )

            if res.returncode != 0:
                raise TTSInferenceError(
                    f"[FAIL-FAST] sherpa-onnx-offline-tts execution failed (code {res.returncode}):\n"
                    f"{res.stderr.strip()}"
                )

            if not os.path.exists(temp_wav) or os.path.getsize(temp_wav) == 0:
                raise TTSInferenceError(
                    f"[FAIL-FAST] sherpa-onnx exited with 0 but generated no audio file or empty file: {res.stderr.strip()}"
                )

            # Load raw WAV and apply acoustic voice adaptation if speaker profile is provided
            raw_buf = AudioBuffer.from_wav_file(temp_wav)
            resolved_prof = speaker_profile or profile
            if resolved_prof is not None:
                from .acoustic_adaptor import adapt_waveform_to_profile
                adapted_samples = adapt_waveform_to_profile(raw_buf.samples, raw_buf.sample_rate, resolved_prof)
                raw_buf = AudioBuffer(adapted_samples, sample_rate=raw_buf.sample_rate)

            # Apply DAC ramp-up silence padding (200ms lead-in, 150ms lead-out)
            padded_buf = raw_buf.pad_silence(lead_in_ms=200, lead_out_ms=150)

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            dur_sec = padded_buf.duration_seconds
            rtf = (elapsed_ms / 1000.0) / max(0.001, dur_sec)

            if output:
                padded_buf.save(output)

            return SherpaResult(
                text=text,
                audio_buffer=padded_buf,
                sample_rate=padded_buf.sample_rate,
                duration_sec=dur_sec,
                elapsed_ms=elapsed_ms,
                rtf=rtf,
                model_name=self.model_name,
                backend=self.backend,
                device_model=f"Cortex-A78_{self.threads}T",
            )
        finally:
            if os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except OSError:
                    pass

    def close(self) -> None:
        self._is_closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
