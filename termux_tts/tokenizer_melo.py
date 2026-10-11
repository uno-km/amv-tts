# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
MeloTTS Tokenizer and Phonetic Normalizer for termux-tts.
Maps input text to MeloTTS model input tensors:
- x: [1, L] (token IDs)
- x_lengths: [1] (sequence length)
- tones: [1, L] (phoneme tones 0-5)
- sid: [1] (speaker ID)
- noise_scale: [1]
- length_scale: [1]
- noise_scale_w: [1]
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


class MeloTokenizer:
    """MeloTTS Phonetic Tokenizer with tokens.txt / lexicon.txt mapping."""

    def __init__(
        self,
        tokens_path: Optional[str] = None,
        lexicon_path: Optional[str] = None,
    ):
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.lexicon: Dict[str, List[str]] = {}

        if tokens_path and os.path.isfile(tokens_path):
            self._load_tokens(tokens_path)
        else:
            self._load_default_tokens()

        if lexicon_path and os.path.isfile(lexicon_path):
            self._load_lexicon(lexicon_path)

    def _load_tokens(self, filepath: str) -> None:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if len(parts) >= 2 and parts[-1].isdigit():
                        token = " ".join(parts[:-1])
                        idx = int(parts[-1])
                    else:
                        token = parts[0]
                        idx = len(self.token_to_id)
                    self.token_to_id[token] = idx
                    self.id_to_token[idx] = token
        except Exception:
            self._load_default_tokens()

    def _load_default_tokens(self) -> None:
        # Standard fallback tokens for common phonemes and symbols
        default_vocab = [
            "_", "AA", "E", "EE", "En", "N", "OO", "V", "a", "a:", "ai", "an", "ang", "ao",
            "b", "c", "ch", "d", "e", "ei", "en", "eng", "er", "f", "g", "h", "i", "i0",
            "ia", "ian", "iang", "iao", "ie", "in", "ing", "iong", "iu", "j", "k", "l",
            "m", "n", "o", "ong", "ou", "p", "q", "r", "s", "sh", "t", "u", "u:", "ua",
            "uai", "uan", "uang", "ui", "un", "uo", "v", "van", "ve", "vn", "w", "x",
            "y", "z", "zh", "!", "?", "…", ",", ".", "—", "\"", "#", "$", "%", "&", "'",
            "(", ")", "*", "+", "-", "/", ":", ";", "<", "=", ">", "@", "[", "\\", "]",
            "^", "`", "{", "|", "}", "~", " "
        ]
        for idx, sym in enumerate(default_vocab):
            self.token_to_id[sym] = idx
            self.id_to_token[idx] = sym

    def _load_lexicon(self, filepath: str) -> None:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        word = parts[0].upper()
                        phones = parts[1:]
                        self.lexicon[word] = phones
        except Exception:
            pass

    def text_to_tokens(self, text: str) -> Tuple[List[int], List[int]]:
        """Converts raw text into token IDs and tone IDs."""
        clean = text.strip()
        tokens: List[int] = []
        tones: List[int] = []

        # Punctuation / pause start
        pad_id = self.token_to_id.get("_", 0)
        tokens.append(pad_id)
        tones.append(0)

        # Split words while keeping punctuation
        words = re.findall(r"[\w']+|[.,!?;—]", clean)

        for w in words:
            w_upper = w.upper()
            if w in self.token_to_id:
                tokens.append(self.token_to_id[w])
                tones.append(0)
            elif w_upper in self.lexicon:
                phones = self.lexicon[w_upper]
                for p in phones:
                    m = re.match(r"^([A-Za-z:]+)([0-5]?)$", p)
                    if m:
                        phone_sym, tone_num = m.groups()
                        tone_val = int(tone_num) if tone_num else 0
                    else:
                        phone_sym = p
                        tone_val = 0

                    tid = self.token_to_id.get(phone_sym, self.token_to_id.get(phone_sym.lower(), pad_id))
                    tokens.append(tid)
                    tones.append(tone_val)
            else:
                for ch in w:
                    tid = self.token_to_id.get(ch, self.token_to_id.get(ch.lower(), pad_id))
                    tokens.append(tid)
                    tones.append(0)

            # Insert inter-word pause
            tokens.append(pad_id)
            tones.append(0)

        if not tokens:
            tokens = [pad_id]
            tones = [0]

        return tokens, tones

    def build_inputs(
        self,
        text: str,
        speed: float = 1.0,
        sid: int = 0,
        noise_scale: float = 0.667,
        noise_scale_w: float = 0.8,
    ) -> Dict[str, np.ndarray]:
        """Produce exact input tensors ready for MeloTTS inference."""
        tokens, tones = self.text_to_tokens(text)
        seq_len = len(tokens)

        speed = max(0.5, min(2.0, float(speed)))
        length_scale = 1.0 / speed

        return {
            "x": np.array([tokens], dtype=np.int32),
            "x_lengths": np.array([seq_len], dtype=np.int32),
            "tones": np.array([tones], dtype=np.int32),
            "sid": np.array([sid], dtype=np.int32),
            "noise_scale": np.array([noise_scale], dtype=np.float32),
            "length_scale": np.array([length_scale], dtype=np.float32),
            "noise_scale_w": np.array([noise_scale_w], dtype=np.float32),
        }
