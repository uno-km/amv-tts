# v1.7.0 (2026-10-13) — Tier 5 SOTA Neural Pipelines & Zero-Shot Voice Cloning

Integrated Tier 5 SOTA neural speech synthesis pipelines (CosyVoice 2, F5-TTS, ChatTTS INT8 ONNX) with zero-shot voice cloning, purged legacy sinusoidal stubs for strict fail-fast execution, and added acoustic adaptor CLI.

### Key Changes
- Integrated Tier 5 SOTA ONNX neural synthesis pipelines: CosyVoice 2, F5-TTS, and ChatTTS INT8.
- Enabled on-device zero-shot voice cloning and conversational prosody control.
- Eliminated legacy sinusoidal fallback stubs in favor of strict fail-fast error reporting.
- Added speaker profile and acoustic adaptor CLI commands.
- Unified GPU neural vocoder dispatch through ameva-runtime HAL.

---

# Release Notes - termux-tts v1.5.5

**Release Tag**: `v1.5.5`  
**Distribution Channels**: PyPI (`termux-tts`), NPM (`termux-tts`), GitHub Releases  
**Target Platform**: Android Termux (ARM64 / aarch64 Bionic)  
**License**: Apache-2.0  

---

## Highlights & Key Architectural Changes

### 1. Unified 5-Backend Standardization
- **Ecosystem Parameter Whitelist**: Aligned synthesis command choices to standard 5-set `["auto", "gpu", "vulkan", "opencl", "cpu"]`.
- **Fail-Fast Policy**: Eliminates unsupported or ambiguous arguments immediately.

### 2. Full Test Pass Verification
- **Test Suite Integrity**: Validated 54 passing unit tests across G2P phonemizers, DSP formant synthesizers, and multilingual neural vocoders.
