# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
from pathlib import Path

vocab_file = Path.home() / ".cache/termux-tts/models/sota/f5tts/vocab.txt"
with open(vocab_file, "r", encoding="utf-8") as f:
    vocab = {line.strip("\r\n"): i for i, line in enumerate(f)}

text = "The secret of how Joseon built royal palaces on swamp land."
print(f"Vocab size: {len(vocab)}")
for c in sorted(set(text)):
    status = f"ID: {vocab[c]}" if c in vocab else "MISSING!"
    print(f"Char '{c}' ({ord(c)}): {status}")
