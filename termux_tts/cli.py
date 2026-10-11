# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
Command-Line Interface for termux-tts:
- termux-tts synth : Option A Deep Learning Neural Synthesis to File
- termux-tts speak : Option B Android System Native Immediate Voice Output
- termux-tts doctor: 12-Stage Vulkan GPU Hardware Diagnostics
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
from .engine import load, doctor
from termux_tts.exceptions import RuntimeRequiredError

def main():
    """CLI entry point.

    A GPU request without a usable ameva-runtime ends here with the shared explanation on stderr
    and exit status 69, whichever sub-command raised it.
    """
    from termux_tts.exceptions import RuntimeRequiredError
    try:
        return _main_impl()
    except RuntimeRequiredError as err:
        sys.stderr.write(str(err))
        sys.stderr.flush()
        return 69


def _main_impl():
    # Route to default subcommand 'synth' if no recognized subcommand is given
    KNOWN_COMMANDS = {"synth", "speak", "doctor", "install", "profile", "component", "model", "instance"}
    raw_args = sys.argv[1:]
    if raw_args and raw_args[0] not in KNOWN_COMMANDS and raw_args[0] not in ("-h", "--help", "-v", "--version"):
        sys.argv.insert(1, "synth")

    parser = argparse.ArgumentParser(
        prog="termux-tts",
        description="Termux Neural & Native Text-to-Speech Engine"
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # 1. Synth (4-Tier Speech Synthesis)
    synth_parser = subparsers.add_parser("synth", help="Synthesize text to audio WAV file (Synth, Neural, Expressive)")
    synth_parser.add_argument("text_pos", nargs="*", default=[], help="Input text to synthesize (positional)")
    synth_parser.add_argument("-t", "--text", default=None, help="Input text to synthesize (flag)")
    synth_parser.add_argument("-o", "--output", default="output.wav", help="Output WAV filepath")
    synth_parser.add_argument("-l", "--lang", default="auto", help="Language code (auto=Multi-language auto switch, ko/kor=Korean only, en/eng=English only, ja/jpn=Japanese only)")
    synth_parser.add_argument(
        "-e", "--engine", default="auto",
        choices=["auto", "vulkan", "ncnn", "gpu", "native", "neural", "onnx", "expressive", "multilingual", "hybrid", "kokoro", "melo", "supertonic", "cosyvoice", "cosyvoice2", "f5", "f5tts", "chattts", "sota"],
        help="Synthesis engine tier (auto=Smart Routing, cosyvoice=Alibaba 0.5B DiT Flow, f5tts=Non-autoregressive DiT, chattts=Conversational Auto-regressive, vulkan=GPU NCNN, native=Android voice, neural=VITS C++)"
    )
    synth_parser.add_argument("-m", "--model", default=None, help="Path to model file or directory")
    synth_parser.add_argument("-p", "--preset", default="balanced", choices=["fast", "balanced", "expressive", "ultra", "cinematic", "sota"])
    synth_parser.add_argument("-d", "--device", default="auto", choices=["auto", "gpu", "vulkan", "opencl", "cpu"], help="Compute target device")
    synth_parser.add_argument("-b", "--backend", dest="device", choices=["auto", "gpu", "vulkan", "opencl", "cpu"], help="Alias for --device")
    synth_parser.add_argument("--gpu", dest="device", action="store_const", const="gpu", help="Enable hardware GPU acceleration")
    synth_parser.add_argument("--cpu", dest="device", action="store_const", const="cpu", help="Force CPU compute mode")
    synth_parser.add_argument("--tier", default=None, choices=["high", "medium", "balanced", "fast", "ultra"], help="Target model tier (high=Studio FP16, medium=Balanced)")
    synth_parser.add_argument("-s", "--speed", type=float, default=1.0, help="Speech speed multiplier (0.5 to 2.0)")
    synth_parser.add_argument("--mode", default="unified", choices=["unified", "stitch"], help="Multilingual synthesis mode (unified=Single-pass G2P transliteration [BigTech Standard], stitch=Multi-model chunk concatenation)")
    synth_parser.add_argument("--threads", type=int, default=4, help="Compute worker threads (ARM NEON)")
    synth_parser.add_argument("--volume", type=int, default=None, help="Set Android media volume (1 to 15)")
    synth_parser.add_argument("--play", action="store_true", help="Play synthesized audio through physical speaker immediately")
    synth_parser.add_argument("--rpc", type=str, default=None, help="Distributed RPC server addresses (e.g. '192.168.0.220:50052,192.168.0.253:50052')")
    synth_parser.add_argument("-ts", "--tensor-split", type=str, default=None, help="Fraction of the model to offload across devices (e.g. '50,50' or '60,40')")
    synth_parser.add_argument("--cluster-rpc-servers", type=str, default=None, help="AMEVA cluster RPC server endpoints (comma-separated host:port)")
    synth_parser.add_argument("--cluster-split-mode", type=str, default=None, help="AMEVA cluster split mode")
    synth_parser.add_argument("--cluster-tensor-split", type=str, default=None, help="AMEVA cluster tensor split ratios")
    synth_parser.add_argument("--cluster-vram-budget", type=str, default=None, help="AMEVA cluster VRAM budget")
    synth_parser.add_argument("--ref-audio", type=str, default=None, help="Reference audio file (WAV/MP3) for zero-shot voice cloning")
    synth_parser.add_argument("--ref-text", type=str, default=None, help="Reference audio transcription for high-fidelity cloning")
    synth_parser.add_argument("--profile", type=str, default=None, help="Registered speaker profile name or path (.vec) for zero-shot personalization")
    synth_parser.add_argument("--sid", "--speaker-id", dest="speaker_id", type=int, default=None, help="Speaker ID for multi-speaker neural models (e.g. 0, 1, 2...)")
    synth_parser.add_argument("--prompt", type=str, default=None, help="Stylistic or emotional prompt (e.g. 'dramatic documentary narrator')")

    # 2. Speak (Option B: Native Samsung/Google System Voice)
    speak_parser = subparsers.add_parser("speak", help="Speak text directly through device speaker (Option B: Native)")
    speak_parser.add_argument("-t", "--text", required=True, help="Input text to speak")
    speak_parser.add_argument("-l", "--lang", default="ko", help="Language code (ko, en)")
    speak_parser.add_argument("-s", "--stream", default="MUSIC", help="Audio stream (MUSIC, NOTIFICATION, ALARM)")
    speak_parser.add_argument("--volume", type=int, default=None, help="Set Android media volume (1 to 15)")

    # 3. Doctor (Diagnostics)
    subparsers.add_parser("doctor", help="Run 12-stage Vulkan GPU hardware diagnostics")

    # 4. Install (One-Click Automated Provisioner - Pure CPU Baseline)
    install_parser = subparsers.add_parser("install", help="1-Click download and provision native CPU Sherpa neural engine and models")
    install_parser.add_argument(
        "--models", default="default",
        help="Language model packages to provision: default (Korean, English, Kokoro-82M), specific language (ko, en, ja, zh, etc.), or specific model (melo, kokoro, supertonic, all)"
    )
    install_parser.add_argument("--force", action="store_true", help="Force overwrite existing binary and model assets")
    install_parser.add_argument("--no-play", action="store_true", help="Skip playback verification during self-test")

    # 5. Profile (Personalized Speaker Profile Management)
    profile_parser = subparsers.add_parser("profile", help="Manage 1-shot voice profiles (~/.termux-tts/profiles/)")
    prof_subs = profile_parser.add_subparsers(dest="profile_action", help="Profile actions")

    prof_create = prof_subs.add_parser("create", help="Create a 192-dim speaker profile from a 3~7s reference audio")
    prof_create.add_argument("name", help="Profile identifier name (e.g. 'my_voice')")
    prof_create.add_argument("-f", "--file", required=True, help="Input reference WAV file")

    prof_subs.add_parser("list", help="List all registered speaker profiles")

    prof_inspect = prof_subs.add_parser("inspect", help="Inspect profile details and acoustic descriptors")
    prof_inspect.add_argument("name", help="Profile name to inspect")

    prof_del = prof_subs.add_parser("delete", help="Delete a registered speaker profile")
    prof_del.add_argument("name", help="Profile name to delete")

    prof_cmp = prof_subs.add_parser("compare", help="Compare cosine similarity between two voice profiles or audio files")
    prof_cmp.add_argument("target1", help="First profile name or WAV file")
    prof_cmp.add_argument("target2", help="Second profile name or WAV file")

    # ── AMEVA Component Protocol v1 ─────────────────────────────────────────
    _protocol_available = False
    try:
        from ameva_component.cli_support import build_protocol_subcommands
        build_protocol_subcommands(subparsers)
        _protocol_available = True
    except ImportError as _proto_err:
        _protocol_available = False
    # ────────────────────────────────────────────────────────────────────────

    args = parser.parse_args()

    import subprocess
    import shutil

    if getattr(args, "volume", None) is not None:
        if shutil.which("termux-volume"):
            subprocess.run(["termux-volume", "music", str(args.volume)], check=False)

    if args.command == "synth":
        pos_text = " ".join(args.text_pos).strip() if getattr(args, "text_pos", None) else None
        target_text = args.text or (pos_text if pos_text else None)
        if not target_text:
            synth_parser.error("the following arguments are required: text (as positional arguments or -t/--text)")

        out_path = os.path.expanduser(args.output) if args.output else None
        effective_lang = args.lang
        if effective_lang == "auto":
            from termux_tts.script_classifier import MultilingualTokenizer
            detected = list(MultilingualTokenizer.detect_languages(target_text))
            if len(detected) == 1:
                effective_lang = detected[0]

        with load(
            model=args.model,
            language=effective_lang,
            preset=args.preset,
            device=args.device,
            threads=args.threads,
            engine=args.engine,
            tier=getattr(args, "tier", None),
            rpc=getattr(args, "rpc", None),
            tensor_split=getattr(args, "tensor_split", None),
            cluster_rpc_servers=getattr(args, "cluster_rpc_servers", None),
            cluster_split_mode=getattr(args, "cluster_split_mode", None),
            cluster_tensor_split=getattr(args, "cluster_tensor_split", None),
            cluster_vram_budget=getattr(args, "cluster_vram_budget", None),
        ) as engine:
            synth_kwargs = {}
            if getattr(args, "ref_audio", None) is not None:
                synth_kwargs["ref_audio"] = args.ref_audio
            if getattr(args, "ref_text", None) is not None:
                synth_kwargs["ref_text"] = args.ref_text
            if getattr(args, "profile", None) is not None:
                synth_kwargs["profile"] = args.profile
            if getattr(args, "speaker_id", None) is not None:
                synth_kwargs["speaker_id"] = args.speaker_id
            if getattr(args, "prompt", None) is not None:
                synth_kwargs["prompt"] = args.prompt

            res = engine.synthesize(
                target_text,
                output=out_path,
                speed=args.speed,
                language=effective_lang,
                mode=getattr(args, "mode", "unified"),
                **synth_kwargs
            )
            backend_name = getattr(res, "backend", "UNKNOWN")
            model_name = getattr(res, "model_name", "model")
            dur = getattr(res, "duration_sec", 0.0)
            elapsed = getattr(res, "elapsed_ms", 0.0)
            rtf = getattr(res, "rtf", 0.0)
            print(f"[SUCCESS] Synthesized via {backend_name} ({model_name}) -> {out_path or args.output}")
            print(f"  Duration: {dur:.2f}s | Elapsed: {elapsed:.1f}ms | RTF: {rtf:.4f}x")

            if args.play and out_path and os.path.exists(out_path):
                if shutil.which("termux-media-player"):
                    subprocess.run(["termux-media-player", "play", out_path], check=False)
                elif shutil.which("play-audio"):
                    subprocess.run(["play-audio", out_path], check=False)

    elif args.command == "speak":
        with load(language=args.lang) as engine:
            res = engine.speak(args.text, stream=args.stream)
            print(f"[SUCCESS] Spoken via {res.engine_name} on stream {args.stream} ({res.elapsed_ms:.1f}ms)")

    elif args.command == "doctor":
        diag = doctor()
        print("=" * 60)
        print("   TERMUX-TTS 12-STAGE VULKAN HARDWARE DIAGNOSTICS")
        print("=" * 60)
        for k, v in diag.items():
            print(f"  - {k:30s}: {v}")
        print("=" * 60)

    elif args.command == "install":
        from .installer import run_installation
        run_installation(
            models=getattr(args, "models", "default"),
            force=args.force,
            play=not args.no_play
        )

    elif args.command == "profile":
        from .speaker_profile import SpeakerProfile, SpeakerProfileEngine, ProfileNotFoundError
        prof_engine = SpeakerProfileEngine()

        action = getattr(args, "profile_action", None)
        if action == "create":
            try:
                prof = prof_engine.extract_from_file(args.file, name=args.name)
                saved_path = prof.save()
                print(f"[SUCCESS] Speaker profile '{prof.name}' created successfully.")
                print(f"  Saved path: {saved_path}")
                print(f"  Duration: {prof.duration_sec:.2f}s | F0 Pitch: {prof.pitch_mean_hz:.1f}Hz | RMS: {prof.energy_rms_db:.1f}dB")
            except RuntimeRequiredError:
                raise
            except Exception as e:
                print(f"[ERROR] Failed to create profile: {e}", file=sys.stderr)
                sys.exit(1)

        elif action == "list":
            profiles = prof_engine.list_profiles()
            print(f"Registered Voice Profiles ({len(profiles)}):")
            print("-" * 75)
            print(f"{'Name':<20} {'Duration':<10} {'Pitch (Hz)':<12} {'Energy (dB)':<12} {'Created'}")
            print("-" * 75)
            for p in profiles:
                print(f"{p.name:<20} {p.duration_sec:>6.2f}s    {p.pitch_mean_hz:>8.1f}Hz   {p.energy_rms_db:>8.1f}dB   {p.created_at}")
            print("-" * 75)

        elif action == "inspect":
            try:
                p = prof_engine.get_profile(args.name)
                print(f"Profile: {p.name}")
                print(f"  - Embedding Dimensions : {len(p.embedding)} (L2 Norm: {np.linalg.norm(p.embedding):.4f})")
                print(f"  - Duration             : {p.duration_sec:.2f}s")
                print(f"  - Pitch Mean           : {p.pitch_mean_hz:.1f} Hz")
                print(f"  - Energy RMS           : {p.energy_rms_db:.1f} dB")
                print(f"  - Created At           : {p.created_at}")
                print(f"  - Extractor Type       : {p.extractor_type}")
                print(f"  - Source Audio         : {p.source_file or 'N/A'}")
            except RuntimeRequiredError:
                raise
            except Exception as e:
                print(f"[ERROR] {e}", file=sys.stderr)
                sys.exit(1)

        elif action == "delete":
            target = prof_engine.profiles_dir / f"{args.name}.vec"
            target_meta = prof_engine.profiles_dir / f"{args.name}.meta.json"
            if not target.is_file():
                print(f"[ERROR] Profile '{args.name}' not found at {target}", file=sys.stderr)
                sys.exit(1)
            target.unlink(missing_ok=True)
            target_meta.unlink(missing_ok=True)
            print(f"[SUCCESS] Deleted profile '{args.name}'.")

        elif action == "compare":
            try:
                def _get_or_load(target_str: str) -> SpeakerProfile:
                    if Path(target_str).is_file():
                        return prof_engine.extract_from_file(target_str)
                    return prof_engine.get_profile(target_str)

                p1 = _get_or_load(args.target1)
                p2 = _get_or_load(args.target2)
                sim = SpeakerProfileEngine.compute_similarity(p1, p2)
                verdict = "MATCH (Same Speaker)" if sim >= 0.80 else ("BORDERLINE" if sim >= 0.65 else "DIFFERENT")
                print(f"Speaker Verification Similarity Analysis:")
                print(f"  - Target 1 : {p1.name} (Pitch: {p1.pitch_mean_hz:.1f}Hz)")
                print(f"  - Target 2 : {p2.name} (Pitch: {p2.pitch_mean_hz:.1f}Hz)")
                print(f"  - Cosine Similarity: {sim:.4f} (-1.0 to +1.0)")
                print(f"  - Verdict          : [{verdict}]")
            except RuntimeRequiredError:
                raise
            except Exception as e:
                print(f"[ERROR] Comparison failed: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            profile_parser.print_help()

    elif args.command in ("component", "model", "instance") and _protocol_available:
        from ameva_component.cli_support import dispatch_protocol
        from termux_tts.control import TTSControl
        dispatch_protocol(args, TTSControl())
    elif args.command in ("component", "model", "instance"):
        print("[ERROR] ameva-component-sdk not installed.", file=sys.stderr)
        sys.exit(1)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

