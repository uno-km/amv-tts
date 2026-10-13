# Termux-TTS

[![PyPI](https://img.shields.io/pypi/v/termux-tts.svg?style=flat-square&color=0369a1)](https://pypi.org/project/termux-tts/)
[![Python](https://img.shields.io/pypi/pyversions/termux-tts.svg?style=flat-square)](https://pypi.org/project/termux-tts/)
[![npm](https://img.shields.io/npm/v/termux-tts.svg?style=flat-square&color=b91c1c)](https://www.npmjs.com/package/termux-tts)
[![License](https://img.shields.io/badge/License-Apache_2.0-004499.svg?style=flat-square)](https://github.com/uno-km/termux-tts)

> Production-Grade 4-Tier On-Device Speech Synthesis Framework (Multilingual Neural Orchestrator, Zero-Dependency DSP Formant & C++ Acceleration)

---

## Architecture & Overview

Termux-TTS is an enterprise-grade, on-device text-to-speech framework optimized for mobile edge hardware and Android Termux environments. Built to eliminate heavy dependency stacks and fragile driver behaviors, it features a resilient 4-Tier architecture:

- **Multilingual Neural Orchestrator**: Dynamic cross-language code-switching mesh covering 9 official languages (Korean, English, Japanese, Chinese, Hindi, Russian, Spanish, French, German) with Unicode script tokenization and 50ms context-aware silence padding.
- **Resident C-API In-Memory Engine**: Direct C-API memory residency with `SherpaResidentManager` achieving sub-0.18x real-time factor with ARM NEON SIMD acceleration and zero subprocess lag.
- **Tier 1: Zero-Dependency Parametric DSP Formant Vocoder**: 0MB disk footprint, Rosenberg glottal pulse formulation, and 5-band biquad formant filters providing deterministic speech synthesis in under 50 milliseconds (RTF 0.013x).
- **Tier 2: Android System Native Voice Bridge**: Direct IPC integration to physical Samsung and Google speech engines via the Termux-API service layer.
- **Tier 3: Subprocess-Isolated Sherpa C++ CPU Engine**: Subprocess-isolated VITS acoustic modeling on ARM64 NEON with memory leak protection.
- **Tier 4: Pure Vulkan GPU Hardware Neural Engine**: High-performance GPU tensor synthesis via precompiled native C++ binaries (`sherpa-ncnn-offline-tts-vulkan`) running high-resolution studio models (`vits-piper-en_US-lessac-high-fp16`) with zero silent CPU fallback.

---

## Empirical Hardware Benchmarks (Physical Devices)

Measurements gathered on physical Android 16 hardware running Termux ARM64:

| Target Device | Hardware Architecture | Synthesis Engine | Model Profile | Audio Length | Synthesis Time | Real-Time Factor (RTF) | Status |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **Galaxy A53** | Exynos 1280 / ARM64 NEON | Multilingual Neural C-API | `KSS + Lessac` (Bilingual) | 10.50 s | **1.82 s** | **0.173x** | Validated (5.78x RT) |
| **Galaxy S25** | Snapdragon 8 Elite / Adreno 830 | Vulkan GPU Neural | `lessac-high-fp16` | 6.70 s | **6.65 s** | **0.993x** | Validated |
| **Galaxy S25** | Snapdragon 8 Elite / Adreno 830 | Vulkan GPU Neural | `lessac-medium` | 4.59 s | **1.21 s** | **0.264x** | Validated (3.79x RT) |
| **Galaxy A35** | Exynos 1380 / Mali-G68 MP5 | Vulkan GPU Neural | `lessac-medium` | 4.52 s | **5.18 s** | **1.146x** | Validated |
| **ARM64 CPU** | Cortex-A78 / A55 | Parametric DSP | 5-Band Biquad | 4.15 s | **0.054 s** | **0.0130x** | Validated (76x RT) |

---

## Installation & Automated Provisioning

### 1. Package Installation
```bash
# Python SDK & CLI
pip install termux-tts

# Node.js / TypeScript
npm install termux-tts
```

### 2. Automated Model Provisioning
Automate downloading and linking precompiled models to the official immutable path (`/data/data/com.termux/files/home/models/tts/`):
```bash
# Default lightweight installation (Korean KSS + English Lessac)
termux-tts install

# On-demand provision specific language models:
termux-tts install --models hi    # Hindi (Piper Swara)
termux-tts install --models ja    # Japanese (Piper Hina)
termux-tts install --models ru    # Russian (Piper Dmitri)
termux-tts install --models zh    # Chinese (AISHELL3)
termux-tts install --models all   # All 9 official languages

# Provision Studio Vulkan GPU engine:
termux-tts install --tier high
```

---

## Quickstart

### Global Command-Line Interface (Zero-Config)
```bash
# 1. Zero-config synthesis with physical speaker playback
termux-tts "Hello 방가방가 나는 parrot 이라고 해." -o ~/out.wav --play

# 2. Flag-based syntax
termux-tts -t "Multilingual neural speech synthesis on device." --play

# 3. Force language pinning
termux-tts -l en -t "Pure English text output." --play
termux-tts -l ko -t "한국어 단독 신경망 음성 합성." --play

# 4. Instant DSP Formant synthesis (0MB footprint)
termux-tts synth -e dsp -t "Zero dependency DSP synthesis." -o dsp.wav

# 5. Direct Android system native speaker broadcast
termux-tts speak -t "Hardware speaker broadcast via Android service."

# 6. Hardware diagnostics
termux-tts doctor
```

### Python SDK
```python
import termux_tts as tts

# 1. Zero-Config Multilingual Neural Synthesis (Korean + English Code-Switching)
with tts.load() as engine:
    result = engine.synthesize(
        "Hello 방가방가 키키키키 나는 parrot 이라고 해.",
        output="multilingual.wav"
    )
    print(f"Generated {result.duration_sec:.2f}s in {result.elapsed_ms:.1f}ms (RTF: {result.rtf:.4f}x)")

# 2. Pure Vulkan GPU Neural Synthesis (Studio Tier)
with tts.load(engine="vulkan", model_tier="high") as engine:
    result = engine.synthesize("Pure Vulkan neural execution on mobile.", output="speech.wav")
    print(f"Synthesized in {result.elapsed_ms:.1f}ms (RTF: {result.rtf:.4f}x)")

# 3. Zero-Dependency DSP Formant Synthesis
with tts.load(engine="dsp", preset="balanced") as engine:
    result = engine.synthesize("Instant speech without model downloads.", output="dsp.wav")
    print(f"DSP Latency: {result.elapsed_ms:.1f}ms")
```

### Node.js / TypeScript
```typescript
import * as tts from 'termux-tts';

async function main() {
  const engine = tts.load({ engine: 'vulkan', tier: 'high' });
  const res = await engine.synthesize("High performance speech synthesis.", { output: "speech.wav" });
  console.log(`Synthesized in ${res.elapsedMs}ms`);
}
main();
```

---

## Official Documentation & Benchmarks
- [Official Architecture & API Reference](https://uno-km.vercel.app/lib/tts/)
- [Ecosystem Metrics & Registry Stats](https://uno-km.vercel.app/foundation/metrics)
- [AMEVA Open-Source Foundation Portal](https://uno-km.vercel.app/foundation/index.html)

---

## License
Licensed under the Apache-2.0 License. Copyright (c) 2026 Eunho Kim ([@uno-km](https://github.com/uno-km)).
