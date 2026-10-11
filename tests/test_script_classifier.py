# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
Tests for Extensible Unicode Script Classifier and Multilingual Tokenizer.
"""
import pytest
from termux_tts.script_classifier import (
    ScriptRegistry,
    MultilingualTokenizer,
    LanguageChunk,
)

def test_single_script_detection():
    assert MultilingualTokenizer.detect_languages("안녕하세요 반갑습니다") == {"ko"}
    assert MultilingualTokenizer.detect_languages("Hello world welcome") == {"en"}
    assert MultilingualTokenizer.detect_languages("こんにちは ありがとう") == {"ja"}
    assert MultilingualTokenizer.detect_languages("你好 谢谢你") == {"zh"}

def test_multilingual_code_switching_tokenization():
    tokenizer = MultilingualTokenizer()
    text = "Hello 반가워 나는 parrot이라고 해, your face 는 정말 아름다워 .like ferrari  알지? 뛰어난 glass에 비친 grass는 정말 grath 해"
    chunks = tokenizer.tokenize(text)

    # Validate chunks
    assert len(chunks) >= 8
    assert chunks[0].language == "en"
    assert chunks[0].text == "Hello"
    assert chunks[1].language == "ko"
    assert "반가워" in chunks[1].text
    
    # Validate intra-word affix attachment (e.g. 'parrot' + '이라고')
    parrot_chunk = next(c for c in chunks if c.text == "parrot")
    assert parrot_chunk.is_affix is True
    assert parrot_chunk.pause_after == 0.020

    # Validate clause punctuation pause (e.g., '해,')
    clause_chunk = next(c for c in chunks if c.text.endswith(","))
    assert clause_chunk.pause_after == 0.150

    # Validate sentence punctuation pause (e.g., '?')
    sentence_chunk = next(c for c in chunks if c.text.endswith("?"))
    assert sentence_chunk.pause_after == 0.250

def test_japanese_kanji_disambiguation():
    tokenizer = MultilingualTokenizer()
    text = "今日は良い天気ですね"
    chunks = tokenizer.tokenize(text)
    assert len(chunks) == 1
    assert chunks[0].language == "ja"

def test_chinese_hanzi_disambiguation():
    tokenizer = MultilingualTokenizer()
    text = "今天天气很好"
    chunks = tokenizer.tokenize(text)
    assert len(chunks) == 1
    assert chunks[0].language == "zh"

def test_custom_language_registration():
    # Register Arabic Unicode range \u0600-\u06FF
    ScriptRegistry.register_language("ar", lambda code: 0x0600 <= code <= 0x06FF)
    
    cat = ScriptRegistry.classify_code_point(ord("\u0645"))
    assert cat == "ar"

    tokenizer = MultilingualTokenizer()
    chunks = tokenizer.tokenize("Hello مرحبا")
    assert len(chunks) == 2
    assert chunks[0].language == "en"
    assert chunks[1].language == "ar"


def test_normalize_language_code():
    from termux_tts.script_classifier import normalize_language_code
    assert normalize_language_code("ko") == "ko"
    assert normalize_language_code("kor") == "ko"
    assert normalize_language_code("korean") == "ko"
    assert normalize_language_code("한국어") == "ko"
    assert normalize_language_code("en") == "en"
    assert normalize_language_code("eng") == "en"
    assert normalize_language_code("english") == "en"
    assert normalize_language_code("영어") == "en"
    assert normalize_language_code("ja") == "ja"
    assert normalize_language_code("jpn") == "ja"
    assert normalize_language_code("auto") == "auto"
    assert normalize_language_code(None) == "auto"


def test_force_language_tokenization():
    tokenizer = MultilingualTokenizer()
    text = "Hello 방가방가 parrot"
    
    # 1. Force English: whole sentence treated as en
    en_chunks = tokenizer.tokenize(text, force_language="en")
    assert len(en_chunks) == 1
    assert en_chunks[0].language == "en"
    assert en_chunks[0].text == "Hello 방가방가 parrot"

    # 2. Force Korean: whole sentence treated as ko
    ko_chunks = tokenizer.tokenize(text, force_language="ko")
    assert len(ko_chunks) == 1
    assert ko_chunks[0].language == "ko"
    assert ko_chunks[0].text == "Hello 방가방가 parrot"

    # 3. Auto: splits by language
    auto_chunks = tokenizer.tokenize(text, force_language="auto")
    assert len(auto_chunks) == 3
    assert auto_chunks[0].language == "en"
    assert auto_chunks[1].language == "ko"
    assert auto_chunks[2].language == "en"
