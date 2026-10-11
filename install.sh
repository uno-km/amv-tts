#!/bin/bash
# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
# ==============================================================================
# termux-tts: Zero-Drift One-Touch Unified Installer (Android Termux / Linux)
# Open-Source under Apache License 2.0 (AMEVA Foundation)
# ==============================================================================

set -e

echo "========================================================="
echo "        🚀 Initializing termux-tts Installer             "
echo "========================================================="

# 1. System Package Manager Provisioning (pkg / apt)
if command -v pkg >/dev/null 2>&1; then
    echo "[1/4] Updating Termux packages and installing dependencies..."
    pkg update -y
    pkg install -y clang git python python-numpy termux-api nodejs
elif command -v apt-get >/dev/null 2>&1; then
    echo "[1/4] Updating Debian/Ubuntu packages..."
    apt-get update -y
    apt-get install -y build-essential git python3 python3-pip python3-numpy nodejs npm
else
    echo "[!] Unknown package manager. Please ensure python3, numpy, and nodejs are installed."
fi

# 2. Python Toolchain & Package Installation (pip)
echo "[2/4] Installing Python SDK and CLI via pip..."
pip install setuptools wheel
if pip install ameva-runtime 2>/dev/null; then
    echo "  -> ameva-runtime hardware diagnostics bound."
else
    echo "  -> ameva-runtime optional hardware acceleration bridge skipped."
fi
pip install --no-build-isolation -e .

# 3. Node.js Dual Engine CLI Installation (npm)
echo "[3/4] Linking Node.js SDK and npm global CLI..."
if command -v npm >/dev/null 2>&1; then
    npm install -g . || npm link
fi

# 4. Diagnostics & Verification
echo "[4/4] Running Vulkan GPU Hardware Probe..."
termux-tts doctor

echo "========================================================="
echo "  ✅ termux-tts successfully installed!"
echo "========================================================="
echo "  Dual-Engine Verification:"
echo "    * Python CLI:   termux-tts synth -t 'Hello' -o test.wav"
echo "    * Native Speak: termux-tts speak -t '안녕하세요' -l ko"
echo "    * Node.js CLI:  npx termux-tts doctor"
echo "========================================================="
