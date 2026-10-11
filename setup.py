# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
import os
from setuptools import setup, find_packages

setup(
    name="termux-tts",
    version="1.5.4",
    description="Ultra-Fast On-Device 4-Tier Text-to-Speech Framework (DSP Synth, Android Native, C++ Sherpa-ONNX Neural & Expressive)",
    long_description=open("README.pypi.md", encoding="utf-8").read() if os.path.exists("README.pypi.md") else open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="AMEVA Foundation",
    license="Apache-2.0",
    url="https://github.com/uno-km/termux-tts",
    packages=find_packages(include=["termux_tts", "termux_tts.*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "onnx": ["onnxruntime>=1.15.0"],
        "neural": ["onnxruntime>=1.15.0"],
        "dev": ["pytest>=7.0.0", "pytest-asyncio"],
    },
    entry_points={
        "console_scripts": [
            "termux-tts=termux_tts.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Android",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
)
