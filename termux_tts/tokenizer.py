# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
Grapheme-to-Phoneme (G2P) Phonetic Tokenizer for Korean and English.
Includes Number-to-Speech Normalizer and Inline Expressive Tag Parser.
"""

import re
from typing import List, Dict
from .exceptions import TTSLanguageNotSupportedError
from .g2p_korean import korean_text_to_phonemes

HANGUL_BASE = 0xAC00
HANGUL_END = 0xD7A3

CHO = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"
]
JUNG = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
    "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"
]
JONG = [
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
    "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"
]

EXPRESSIVE_TAGS = {
    "[laugh]": 1001,
    "[sigh]": 1002,
    "[breath]": 1003,
    "[uv_break]": 1004,
    "[clears_throat]": 1005,
    "[pause]": 1006
}

VOCAB: List[str] = [
    "_", " ", "!", "?", ",", ".", "~", "-",
    *CHO, *JUNG, *[j for j in JONG if j],
    *"abcdefghijklmnopqrstuvwxyz"
]
VOCAB_TO_ID: Dict[str, int] = {sym: idx for idx, sym in enumerate(VOCAB)}
PAD_ID: int = VOCAB_TO_ID["_"]
SPACE_ID: int = VOCAB_TO_ID[" "]

KOREAN_DIGITS = ["", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]
SMALL_UNITS = ["", "십", "백", "천"]
BIG_UNITS = ["", "만", "억", "조", "경"]

ENGLISH_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen", "14": "fourteen",
    "15": "fifteen", "16": "sixteen", "17": "seventeen", "18": "eighteen", "19": "nineteen",
    "20": "twenty", "30": "thirty", "40": "forty", "50": "fifty",
    "60": "sixty", "70": "seventy", "80": "eighty", "90": "ninety",
    "100": "hundred", "1000": "thousand"
}

def decompose_hangul(char: str) -> List[str]:
    code = ord(char)
    if HANGUL_BASE <= code <= HANGUL_END:
        offset = code - HANGUL_BASE
        cho_idx = offset // (21 * 28)
        jung_idx = (offset % (21 * 28)) // 28
        jong_idx = offset % 28
        res = [CHO[cho_idx], JUNG[jung_idx]]
        if jong_idx > 0:
            res.append(JONG[jong_idx])
        return res
    return [char]

def _convert_4digits_korean(chunk: str) -> str:
    num = int(chunk)
    if num == 0:
        return ""
    res = []
    str_num = str(num).zfill(4)
    for i, ch in enumerate(str_num):
        d = int(ch)
        if d > 0:
            unit = SMALL_UNITS[3 - i]
            if d == 1 and unit != "":
                res.append(unit)
            else:
                res.append(KOREAN_DIGITS[d] + unit)
    return "".join(res)

def number_to_korean_sino(num_str: str) -> str:
    """Convert integer string to authentic Sino-Korean place-value numerals (e.g. 1234 -> 천이백삼십사, 10000 -> 만)."""
    try:
        n = int(num_str)
    except ValueError:
        return num_str
    if n == 0:
        return "영"
    rev_str = str(n)[::-1]
    chunks = [rev_str[i:i+4][::-1] for i in range(0, len(rev_str), 4)]
    parts = []
    for i, chunk in enumerate(chunks):
        c_korean = _convert_4digits_korean(chunk)
        if c_korean:
            unit = BIG_UNITS[i]
            # 10000일 때 '일만' 대신 '만' (단, 210000 -> 이십일만)
            if c_korean == "일" and unit != "" and len(chunks) == i + 1:
                parts.append(unit)
            else:
                parts.append(c_korean + unit)
    return "".join(reversed(parts))

def normalize_numbers_korean(text: str) -> str:
    """Convert digit sequences into spoken Korean place-value numerals."""
    return re.sub(r"\d+", lambda m: number_to_korean_sino(m.group(0)), text)

def normalize_numbers_english(text: str) -> str:
    """Convert digit sequences into spoken English words."""
    def _en_repl(m):
        num_str = m.group(0)
        if num_str in ENGLISH_WORDS:
            return " " + ENGLISH_WORDS[num_str] + " "
        # Digit-by-digit for phone numbers / codes
        return " " + " ".join(ENGLISH_WORDS.get(d, d) for d in num_str) + " "
    return re.sub(r"\d+", _en_repl, text)

class PhoneticTokenizer:
    def __init__(self, language: str = "ko"):
        self.language = language.lower()
        if self.language in ("auto", "multilingual"):
            self.language = "ko"
        self.vocab = VOCAB
        if self.language not in ["ko", "korean", "en", "english"]:
            raise TTSLanguageNotSupportedError(f"Language '{language}' is not supported. Supported: ['ko', 'en', 'auto', 'multilingual']")

    def normalize_text(self, text: str) -> str:
        if not text or not text.strip():
            return ""
        # 1. Normalize linebreaks and tabs
        text = re.sub(r"[\r\n\t]+", " ", text)
        
        # 2. Digits to spoken words
        if self.language in ["ko", "korean"]:
            text = normalize_numbers_korean(text)
            text = korean_text_to_phonemes(text)
        else:
            text = normalize_numbers_english(text)

        # 3. Collapse multiple spaces
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    def tokenize(self, text: str) -> List[int]:
        normalized = self.normalize_text(text)
        if not normalized:
            return []

        # Parse expressive tags first
        tag_pattern = re.compile(r"(\[[a-zA-Z_]+\])")
        parts = tag_pattern.split(normalized)

        tokens: List[int] = []
        for part in parts:
            if not part:
                continue
            if part in EXPRESSIVE_TAGS:
                tokens.append(EXPRESSIVE_TAGS[part])
                continue

            if self.language in ["ko", "korean"]:
                for char in part:
                    if char == " ":
                        tokens.append(SPACE_ID)
                    elif char in [".", ",", "!", "?", "~", "-"]:
                        if char in VOCAB_TO_ID:
                            tokens.append(VOCAB_TO_ID[char])
                    else:
                        jamos = decompose_hangul(char)
                        for j in jamos:
                            if j in VOCAB_TO_ID:
                                tokens.append(VOCAB_TO_ID[j])
            else: # en
                for char in part.lower():
                    if char in VOCAB_TO_ID:
                        tokens.append(VOCAB_TO_ID[char])

        return tokens

