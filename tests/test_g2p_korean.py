# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
Unit and Granular Test Suite for Korean Grapheme-to-Phoneme (G2P) Engine.
Validates adherence to National Institute of Korean Language Standard Pronunciation Rules.
"""

import pytest
from termux_tts.g2p_korean import korean_text_to_phonemes, KoreanG2PEngine
from termux_tts.tokenizer import PhoneticTokenizer

def test_palatalization_rule():
    """Verify 구개음화 (제17항): ㄷ, ㅌ + ㅣ -> ㅈ, ㅊ"""
    assert korean_text_to_phonemes("굳이") == "구지"
    assert korean_text_to_phonemes("같이") == "가치"
    assert korean_text_to_phonemes("미닫이") == "미다지"
    assert korean_text_to_phonemes("붙이다") == "부치다"
    assert korean_text_to_phonemes("핥이다") == "할치다"

def test_aspiration_rule():
    """Verify 격음화 / 거센소리되기 (제12항): ㄱ,ㄷ,ㅂ,ㅈ + ㅎ -> ㅋ,ㅌ,ㅍ,ㅊ"""
    assert korean_text_to_phonemes("축하") == "추카"
    assert korean_text_to_phonemes("좋다") == "조타"
    assert korean_text_to_phonemes("입학") == "이팍"
    assert korean_text_to_phonemes("맞히다") == "마치다"
    assert korean_text_to_phonemes("좋은") == "조은"

def test_nasalization_rule():
    """Verify 비음화 (제18/19항): ㄱ,ㄷ,ㅂ + ㄴ,ㅁ -> ㅇ,ㄴ,ㅁ & ㅁ,ㅇ + ㄹ -> ㄴ"""
    assert korean_text_to_phonemes("국물") == "궁물"
    assert korean_text_to_phonemes("닫는") == "단는"
    assert korean_text_to_phonemes("밥먹다") == "밤먹따"
    assert korean_text_to_phonemes("독립") == "동닙"
    assert korean_text_to_phonemes("백로") == "뱅노"
    assert korean_text_to_phonemes("종로") == "종노"
    assert korean_text_to_phonemes("심리") == "심니"

def test_liquidization_rule():
    """Verify 유음화 (제20항): ㄴ+ㄹ, ㄹ+ㄴ -> ㄹ+ㄹ"""
    assert korean_text_to_phonemes("신라") == "실라"
    assert korean_text_to_phonemes("난로") == "날로"
    assert korean_text_to_phonemes("칼날") == "칼랄"
    assert korean_text_to_phonemes("줄넘기") == "줄럼기"

def test_tensification_rule():
    """Verify 경음화 / 된소리되기 (제23항): ㄱ,ㄷ,ㅂ + ㄱ,ㄷ,ㅂ,ㅅ,ㅈ -> ㄲ,ㄸ,ㅃ,ㅆ,ㅉ"""
    assert korean_text_to_phonemes("국밥") == "국빱"
    assert korean_text_to_phonemes("학교") == "학꾜"
    assert korean_text_to_phonemes("옷고름") == "옫꼬름"
    assert korean_text_to_phonemes("옆집") == "엽찝"

def test_liaison_rule():
    """Verify 연음 규칙 (제13/14항): 받침 + 모음"""
    assert korean_text_to_phonemes("옷이") == "오시"
    assert korean_text_to_phonemes("닭을") == "달글"
    assert korean_text_to_phonemes("값을") == "갑슬"

def test_sino_korean_exceptions():
    """Verify 특수 한자어 및 복합어 사전 예외 발음"""
    assert korean_text_to_phonemes("생산량") == "생산냥"
    assert korean_text_to_phonemes("결단력") == "결딴녁"
    assert korean_text_to_phonemes("금융") == "금늉"
    assert korean_text_to_phonemes("솜이불") == "솜니불"
    assert korean_text_to_phonemes("맨입") == "맨닙"
    assert korean_text_to_phonemes("꽃잎") == "꼰닙"

def test_tokenizer_integration_with_g2p():
    """Verify PhoneticTokenizer end-to-end normalization with G2P and numbers."""
    tok = PhoneticTokenizer(language="ko")
    # 100원 -> 백원 -> 배권 (연음/자음동화)
    norm = tok.normalize_text("100원 국물 같이 굳이 먹자")
    assert "배권" in norm or "백원" in norm
    assert "궁물" in norm
    assert "가치" in norm
    assert "구지" in norm
