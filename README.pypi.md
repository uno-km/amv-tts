# Termux-TTS (Python)

[![PyPI](https://img.shields.io/pypi/v/termux-tts.svg?style=flat-square&color=0369a1)](https://pypi.org/project/termux-tts/)
[![Python](https://img.shields.io/pypi/pyversions/termux-tts.svg?style=flat-square)](https://pypi.org/project/termux-tts/)
[![License](https://img.shields.io/badge/License-Apache_2.0-004499.svg?style=flat-square)](https://github.com/uno-km/termux-tts)

> Production-Grade 4-Tier On-Device Speech Synthesis Framework for Mobile & Edge (Zero-Dependency DSP Formant, C++ Vulkan GPU Neural Engine & Android Native Voice Bridge)

## Installation

```bash
pip install termux-tts
```

### 1-Click Automated Engine Provisioning
```bash
termux-tts install --tier high
```

## Quickstart

```python
import termux_tts as tts

# 1. Studio Vulkan GPU Neural Engine
with tts.load(engine="vulkan", model_tier="high") as engine:
    result = engine.synthesize("Neural speech synthesis on mobile GPU.", output="speech.wav")
    print(f"Elapsed: {result.elapsed_ms:.1f}ms (RTF: {result.rtf:.4f}x)")

# 2. Zero-Dependency DSP Formant Mode
with tts.load(engine="dsp", preset="balanced") as engine:
    result = engine.synthesize("Instant speech generation.", output="dsp.wav")
    print(f"DSP Latency: {result.elapsed_ms:.1f}ms")

# 3. Direct Android Native Speaker Output
with tts.load(engine="native", language="en") as engine:
    engine.speak("Direct hardware speaker output.")
```

## Benchmarks (Physical Devices)

| Target Device | Hardware Architecture | Synthesis Engine | Real-Time Factor (RTF) | Status |
| :--- | :--- | :--- | :---: | :---: |
| **Galaxy S25** | Snapdragon 8 Elite / Adreno 830 | Vulkan GPU (`lessac-high-fp16`) | **0.993x** | Validated |
| **Galaxy S25** | Snapdragon 8 Elite / Adreno 830 | Vulkan GPU (`lessac-medium`) | **0.264x** | Validated |
| **Galaxy A35** | Exynos 1380 / Mali-G68 MP5 | Vulkan GPU (`lessac-medium`) | **1.146x** | Validated |
| **ARM64 CPU** | All Core Profiles | Parametric DSP Formant | **0.0130x** | Validated |

## Documentation
- [Official Documentation & API Reference](https://uno-km.vercel.app/lib/tts/)
- [GitHub Repository](https://github.com/uno-km/termux-tts)

## License
Apache-2.0 License. Copyright (c) 2026 Eunho Kim (@uno-km).


## What's New in v1.7.0 — Tier 5 SOTA Neural Pipelines & Zero-Shot Voice Cloning

- Integrated Tier 5 SOTA ONNX neural synthesis pipelines: CosyVoice 2, F5-TTS, and ChatTTS INT8.
- Enabled on-device zero-shot voice cloning and conversational prosody control.
- Eliminated legacy sinusoidal fallback stubs in favor of strict fail-fast error reporting.
- Added speaker profile and acoustic adaptor CLI commands.
- Unified GPU neural vocoder dispatch through ameva-runtime HAL.
