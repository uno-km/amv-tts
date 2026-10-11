# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
Extensible Unicode Script Classifier and Multilingual Tokenizer.
Supports Korean (ko), English/Latin (en), Japanese (ja), Chinese (zh),
and dynamic extension for future languages.
Adheres to strict Zero-Silent-Fallback and AOSF-ENG-STD-2026 Governance.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set, Dict, Callable


@dataclass
class LanguageChunk:
    text: str
    language: str
    pause_after: float
    is_affix: bool = False

    def __repr__(self) -> str:
        return f"LanguageChunk(lang='{self.language}', text='{self.text}', pause={self.pause_after:.3f}s, affix={self.is_affix})"


class ScriptRegistry:
    """
    Extensible Unicode Range Registry for multi-script linguistic detection.
    Allows dynamic registration of new languages without modifying core dispatch logic.
    """
    _CUSTOM_MATCHERS: Dict[str, Callable[[int], bool]] = {}

    @classmethod
    def register_language(cls, lang_code: str, matcher: Callable[[int], bool]) -> None:
        cls._CUSTOM_MATCHERS[lang_code.lower()] = matcher

    @classmethod
    def classify_code_point(cls, code: int) -> str:
        # 1. Custom registered languages
        for lang_code, matcher in cls._CUSTOM_MATCHERS.items():
            if matcher(code):
                return lang_code.lower()

        # 2. Korean (Hangul Syllables, Jamo, Compatibility Jamo, Extended Jamo A/B)
        if ((0xAC00 <= code <= 0xD7A3) or
            (0x1100 <= code <= 0x11FF) or
            (0x3130 <= code <= 0x318F) or
            (0xA960 <= code <= 0xA97F) or
            (0xD7B0 <= code <= 0xD7FF)):
            return "ko"

        # 3. Japanese Hiragana & Katakana (distinctly Japanese)
        if ((0x3040 <= code <= 0x309F) or  # Hiragana
            (0x30A0 <= code <= 0x30FF) or  # Katakana
            (0x31F0 <= code <= 0x31FF)):   # Katakana Phonetic Extensions
            return "ja"

        # 4. English & Latin Alphabet (Basic Latin & Latin-1 Supplement)
        if ((0x0041 <= code <= 0x005A) or  # A-Z
            (0x0061 <= code <= 0x007A) or  # a-z
            (0x00C0 <= code <= 0x024F)):   # Latin Extended-A & B
            return "en"

        # 5. CJK Unified Ideographs (Common Hanzi/Kanji/Hanja)
        if ((0x4E00 <= code <= 0x9FFF) or
            (0x3400 <= code <= 0x4DBF) or
            (0x20000 <= code <= 0x2A6DF)):
            return "cjk_ideograph"

        # 6. Hindi (Devanagari script)
        if 0x0900 <= code <= 0x097F:
            return "hi"

        # 7. Russian (Cyrillic script)
        if (0x0400 <= code <= 0x04FF) or (0x0500 <= code <= 0x052F):
            return "ru"

        # 8. Arabic script
        if (0x0600 <= code <= 0x06FF) or (0x0750 <= code <= 0x077F):
            return "ar"

        # 9. Spaces, Controls
        if code in (0x20, 0x09, 0x0A, 0x0D):
            return "space"

        # 10. Punctuations
        ch = chr(code)
        if ch in ".?!":
            return "punct_sentence"
        elif ch in ",;:":
            return "punct_clause"
        elif ch in "\"'`-~…—()[]{}":
            return "punct_inline"
        elif 0x0030 <= code <= 0x0039:
            return "digit"

        return "other"


def normalize_language_code(lang: Optional[str]) -> str:
    """Normalizes various user language inputs (e.g. 'eng', 'kor', 'hindi', 'russian') to ISO 2-letter codes."""
    if not lang:
        return "auto"
    cleaned = lang.strip().lower()
    mapping = {
        "ko": "ko", "kor": "ko", "korean": "ko", "한국어": "ko", "한글": "ko",
        "en": "en", "eng": "en", "english": "en", "영어": "en",
        "ja": "ja", "jpn": "ja", "japanese": "ja", "일본어": "ja", "일어": "ja",
        "zh": "zh", "cmn": "zh", "chinese": "zh", "중국어": "zh", "중문": "zh",
        "hi": "hi", "hin": "hi", "hindi": "hi", "힌디어": "hi",
        "ru": "ru", "rus": "ru", "russian": "ru", "러시아어": "ru", "노어": "ru",
        "es": "es", "spa": "es", "spanish": "es", "스페인어": "es",
        "fr": "fr", "fra": "fr", "fre": "fr", "french": "fr", "프랑스어": "fr",
        "de": "de", "deu": "de", "ger": "de", "german": "de", "독일어": "de",
        "ar": "ar", "ara": "ar", "arabic": "ar", "아랍어": "ar",
        "auto": "auto", "all": "auto", "multi": "auto", "multilingual": "auto",
    }
    return mapping.get(cleaned, cleaned)


class MultilingualTokenizer:
    """
    Intelligent context-aware Code-Switching Tokenizer.
    Partitions input text into language chunks and calculates optimal acoustic pauses
    for seamless, artifact-free multi-speaker concatenation.
    """

    DEFAULT_PAUSE_SENTENCE = 0.250   # . ? ! (Sentence boundary breathing pause)
    DEFAULT_PAUSE_CLAUSE = 0.150     # , ; : (Clause boundary short breath)
    DEFAULT_PAUSE_AFFIX = 0.020      # Direct word+particle affix (e.g. 'parrot'+'이라고')
    DEFAULT_PAUSE_WORD = 0.060       # Normal inter-word space

    def __init__(self, default_language: str = "ko"):
        self.default_language = default_language.lower()

    @staticmethod
    def detect_languages(text: str) -> Set[str]:
        """Returns the set of natural languages detected in the text, ignoring expressive bracket markup."""
        cleaned = re.sub(r"\[[a-zA-Z가-힣_]+\]", "", text)
        detected = set()
        for ch in cleaned:
            script = ScriptRegistry.classify_code_point(ord(ch))
            if script in ("ko", "en", "ja", "hi", "ru", "ar") or script in ScriptRegistry._CUSTOM_MATCHERS:
                detected.add(script)
            elif script == "cjk_ideograph":
                detected.add("zh")
        return detected

    def tokenize(self, text: str, force_language: Optional[str] = None) -> List[LanguageChunk]:
        """
        Dynamically partitions mixed-script text into language chunks with
        contextual silence duration.
        If force_language is specified ('ko', 'en', etc.), the entire text is
        routed to that language without multi-script splitting.
        """
        if not text or not text.strip():
            return []

        norm_forced = normalize_language_code(force_language)
        if norm_forced != "auto":
            clean_str = text.strip()
            if any(clean_str.endswith(p) for p in (".", "?", "!")):
                pause = self.DEFAULT_PAUSE_SENTENCE
            elif any(clean_str.endswith(p) for p in (",", ";", ":")):
                pause = self.DEFAULT_PAUSE_CLAUSE
            else:
                pause = self.DEFAULT_PAUSE_WORD

            return [LanguageChunk(text=clean_str, language=norm_forced, pause_after=pause, is_affix=False)]

        # Contextual disambiguation for CJK Ideographs (Kanji vs Hanzi vs Hanja)
        has_kana = any(ScriptRegistry.classify_code_point(ord(c)) == "ja" for c in text)
        has_hangul = any(ScriptRegistry.classify_code_point(ord(c)) == "ko" for c in text)

        tokens: List[LanguageChunk] = []
        current_lang: Optional[str] = None
        buffer_chars: List[str] = []
        has_trailing_space: bool = False

        def flush_chunk(is_affix: bool = False):
            nonlocal buffer_chars, current_lang
            raw_chunk = "".join(buffer_chars).strip()
            if raw_chunk and current_lang:
                if any(raw_chunk.endswith(p) for p in (".", "?", "!")):
                    pause = self.DEFAULT_PAUSE_SENTENCE
                elif any(raw_chunk.endswith(p) for p in (",", ";", ":")):
                    pause = self.DEFAULT_PAUSE_CLAUSE
                elif is_affix:
                    pause = self.DEFAULT_PAUSE_AFFIX
                else:
                    pause = self.DEFAULT_PAUSE_WORD

                tokens.append(LanguageChunk(
                    text=raw_chunk,
                    language=current_lang,
                    pause_after=pause,
                    is_affix=is_affix
                ))
            buffer_chars = []

        for ch in text:
            script = ScriptRegistry.classify_code_point(ord(ch))

            # Resolve effective language for ideographs
            resolved_lang: Optional[str] = None
            if script in ("ko", "en", "ja") or script in ScriptRegistry._CUSTOM_MATCHERS:
                resolved_lang = script
            elif script == "cjk_ideograph":
                if has_kana and not has_hangul:
                    resolved_lang = "ja"  # Japanese Kanji
                elif has_hangul and not has_kana:
                    resolved_lang = "ko"  # Korean Hanja context
                else:
                    resolved_lang = "zh"  # Chinese Hanzi

            if resolved_lang is not None:
                # Check sentence boundary even within same language (e.g., '알지? 뛰어난')
                ends_with_sentence_punct = any("".join(buffer_chars).strip().endswith(p) for p in (".", "?", "!"))
                
                if current_lang is None:
                    current_lang = resolved_lang
                    buffer_chars.append(ch)
                    has_trailing_space = False
                elif current_lang != resolved_lang:
                    # Language switch boundary!
                    is_affix = not has_trailing_space
                    flush_chunk(is_affix=is_affix)
                    current_lang = resolved_lang
                    buffer_chars.append(ch)
                    has_trailing_space = False
                elif ends_with_sentence_punct and has_trailing_space:
                    # Sentence boundary split within same language!
                    flush_chunk(is_affix=False)
                    current_lang = resolved_lang
                    buffer_chars.append(ch)
                    has_trailing_space = False
                else:
                    buffer_chars.append(ch)
                    has_trailing_space = False

            elif script == "space":
                if buffer_chars:
                    buffer_chars.append(ch)
                    has_trailing_space = True
            else:
                # Punctuation / digits / symbols belong to the active language chunk
                if buffer_chars:
                    buffer_chars.append(ch)

        # Flush final chunk
        if buffer_chars and current_lang:
            flush_chunk(is_affix=False)

        return tokens
