# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
"""
Production-Grade Korean Grapheme-to-Phoneme (G2P) Engine for termux-tts.
Strict adherence to National Institute of Korean Language (국립국어원) Standard Pronunciation Rules.
Implements:
1. Neutralization (평폐쇄음화 및 겹받침 단순화, 제8/9항)
2. Liaison (연음 및 ㅎ 탈락, 제13/14/12항)
3. Palatalization (구개음화: ㄷ,ㅌ+ㅣ -> ㅈ,ㅊ, 제17항)
4. Aspiration (격음화/거센소리되기: ㄱ,ㄷ,ㅂ,ㅈ+ㅎ -> ㅋ,ㅌ,ㅍ,ㅊ, 제12항)
5. Nasalization (비음화: ㄱ,ㄷ,ㅂ+ㄴ,ㅁ -> ㅇ,ㄴ,ㅁ 및 ㅁ,ㅇ+ㄹ -> ㄴ, 제18/19항)
6. Liquidization (유음화: ㄴ+ㄹ, ㄹ+ㄴ -> ㄹ+ㄹ, 제20항)
7. Tensification (경음화/된소리되기: ㄱ,ㄷ,ㅂ+ㄱ,ㄷ,ㅂ,ㅅ,ㅈ -> ㄲ,ㄸ,ㅃ,ㅆ,ㅉ, 제23항)
8. Sino-Korean & Compound Exceptions (한자어/사이시옷 특수 발음 사전)
"""

from typing import Tuple, Dict

HANGUL_BASE = 0xAC00
HANGUL_END = 0xD7A3

CHO_LIST = [
    'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ',
    'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
]
JUNG_LIST = [
    'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
    'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ'
]
JONG_LIST = [
    '', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ',
    'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
    'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
]

CHO_MAP = {c: i for i, c in enumerate(CHO_LIST)}
JUNG_MAP = {c: i for i, c in enumerate(JUNG_LIST)}
JONG_MAP = {c: i for i, c in enumerate(JONG_LIST)}

# 대표 종성 중화 맵 (제8항)
NEUTRAL_JONG = {
    'ㄲ': 'ㄱ', 'ㄳ': 'ㄱ', 'ㅋ': 'ㄱ', 'ㄺ': 'ㄱ',
    'ㅅ': 'ㄷ', 'ㅆ': 'ㄷ', 'ㅈ': 'ㄷ', 'ㅊ': 'ㄷ', 'ㅌ': 'ㄷ', 'ㅎ': 'ㄷ', 'ㄵ': 'ㄴ', 'ㄶ': 'ㄴ',
    'ㄼ': 'ㄹ', 'ㄽ': 'ㄹ', 'ㄾ': 'ㄹ', 'ㅀ': 'ㄹ',
    'ㄻ': 'ㅁ',
    'ㄿ': 'ㅂ', 'ㅄ': 'ㅂ', 'ㅍ': 'ㅂ'
}

# 겹받침 연음 시 분리 맵 (제14항: 닭을 -> 달글, 값을 -> 갑슬)
DOUBLE_JONG_SPLIT = {
    'ㄳ': ('ㄱ', 'ㅅ'),
    'ㄵ': ('ㄴ', 'ㅈ'),
    'ㄶ': ('ㄴ', 'ㅎ'),
    'ㄺ': ('ㄹ', 'ㄱ'),
    'ㄻ': ('ㄹ', 'ㅁ'),
    'ㄼ': ('ㄹ', 'ㅂ'),
    'ㄽ': ('ㄹ', 'ㅅ'),
    'ㄾ': ('ㄹ', 'ㅌ'),
    'ㄿ': ('ㄹ', 'ㅍ'),
    'ㅀ': ('ㄹ', 'ㅎ'),
    'ㅄ': ('ㅂ', 'ㅅ')
}

# 한자어/특수 복합어 불규칙 예외 사전
IRREGULAR_WORDS = {
    '신라': '실라', '난로': '날로', '칼날': '칼랄', '물약': '물략',
    '생산량': '생산냥', '결단력': '결딴녁', '의견란': '의견난', '임진란': '임진난',
    '이원론': '이원논', '입원료': '이붠뇨', '동양루': '동양누', '구원투수': '구원투수',
    '금융': '금늉', '식용유': '식용뉴', '맨입': '맨닙', '솜이불': '솜니불',
    '눈요기': '눈뇨기', '남존여비': '남존녀비', '신여성': '신녀성', '꽃잎': '꼰닙',
    '깻잎': '깬닙', '나뭇잎': '나문닙', '학여울': '항녀울', '독립': '동닙',
    '백로': '뱅노', '협력': '혐녁', '국립': '궁닙', '막일': '망닐'
}

def is_hangul_syllable(ch: str) -> bool:
    return len(ch) == 1 and (HANGUL_BASE <= ord(ch) <= HANGUL_END)

def decompose_syllable(ch: str) -> Tuple[str, str, str]:
    code = ord(ch) - HANGUL_BASE
    cho = CHO_LIST[code // (21 * 28)]
    jung = JUNG_LIST[(code % (21 * 28)) // 28]
    jong = JONG_LIST[code % 28]
    return cho, jung, jong

def compose_syllable(cho: str, jung: str, jong: str = '') -> str:
    cho_i = CHO_MAP.get(cho, 0)
    jung_i = JUNG_MAP.get(jung, 0)
    jong_i = JONG_MAP.get(jong, 0)
    return chr(HANGUL_BASE + (cho_i * 21 * 28) + (jung_i * 28) + jong_i)

class KoreanG2PEngine:
    """High-Performance Bigram Korean Grapheme-to-Phoneme Engine."""

    def __init__(self):
        self.irregular_dict = IRREGULAR_WORDS

    def convert(self, text: str) -> str:
        if not text or not text.strip():
            return ''

        # 1. Dictionary-based irregular word replacement
        res_text = text
        for k, v in self.irregular_dict.items():
            if k in res_text:
                res_text = res_text.replace(k, v)

        # 2. Syllable-by-syllable phonological transformation
        tokens = list(res_text)
        n = len(tokens)
        if n <= 1:
            return self._neutralize_single(res_text)

        i = 0
        while i < n - 1:
            c1, c2 = tokens[i], tokens[i+1]
            if is_hangul_syllable(c1) and is_hangul_syllable(c2):
                t1, t2 = self._apply_bigram_rules(c1, c2)
                tokens[i] = t1
                tokens[i+1] = t2
            i += 1

        # 3. Final word-final neutralization on last character
        if n > 0 and is_hangul_syllable(tokens[-1]):
            cho, jung, jong = decompose_syllable(tokens[-1])
            if jong in NEUTRAL_JONG:
                tokens[-1] = compose_syllable(cho, jung, NEUTRAL_JONG[jong])

        return ''.join(tokens)

    def _neutralize_single(self, ch: str) -> str:
        if len(ch) == 1 and is_hangul_syllable(ch):
            cho, jung, jong = decompose_syllable(ch)
            if jong in NEUTRAL_JONG:
                return compose_syllable(cho, jung, NEUTRAL_JONG[jong])
        return ch

    def _apply_bigram_rules(self, c1: str, c2: str) -> Tuple[str, str]:
        cho1, jung1, jong1 = decompose_syllable(c1)
        cho2, jung2, jong2 = decompose_syllable(c2)

        if not jong1:
            return c1, c2

        # -------------------------------------------------------------
        # Rule 1: 구개음화 (Palatalization - 제17항)
        # ㄷ, ㅌ + 이, 여, 야, 유 -> ㅈ, ㅊ
        # -------------------------------------------------------------
        if jong1 in ('ㄷ', 'ㅌ', 'ㄾ') and cho2 == 'ㅇ' and jung2 in ('ㅣ', 'ㅑ', 'ㅕ', 'ㅛ', 'ㅠ'):
            if jong1 == 'ㄷ':
                return compose_syllable(cho1, jung1, ''), compose_syllable('ㅈ', jung2, jong2)
            elif jong1 in ('ㅌ', 'ㄾ'):
                new_jong1 = 'ㄹ' if jong1 == 'ㄾ' else ''
                return compose_syllable(cho1, jung1, new_jong1), compose_syllable('ㅊ', jung2, jong2)

        # -------------------------------------------------------------
        # Rule 2: 격음화 / 거센소리되기 (Aspiration - 제12항)
        # [ㄱ, ㄷ, ㅂ, ㅈ] + [ㅎ] -> [ㅋ, ㅌ, ㅍ, ㅊ]
        # [ㅎ, ㄶ, ㅀ] + [ㄱ, ㄷ, ㅂ, ㅈ] -> [ㅋ, ㅌ, ㅍ, ㅊ]
        # -------------------------------------------------------------
        if cho2 == 'ㅎ':
            asp_map = {'ㄱ': 'ㅋ', 'ㄷ': 'ㅌ', 'ㅂ': 'ㅍ', 'ㅈ': 'ㅊ', 'ㄺ': 'ㅋ', 'ㄼ': 'ㅍ', 'ㄵ': 'ㅊ'}
            if jong1 in asp_map:
                new_jong1 = 'ㄹ' if jong1 in ('ㄺ', 'ㄼ') else ('ㄴ' if jong1 == 'ㄵ' else '')
                return compose_syllable(cho1, jung1, new_jong1), compose_syllable(asp_map[jong1], jung2, jong2)
            elif jong1 in ('ㅅ', 'ㅆ', 'ㅊ', 'ㅌ'):
                return compose_syllable(cho1, jung1, ''), compose_syllable('ㅌ', jung2, jong2)

        if jong1 in ('ㅎ', 'ㄶ', 'ㅀ'):
            asp_map2 = {'ㄱ': 'ㅋ', 'ㄷ': 'ㅌ', 'ㅂ': 'ㅍ', 'ㅈ': 'ㅊ', 'ㅅ': 'ㅆ'}
            if cho2 in asp_map2:
                new_jong1 = 'ㄴ' if jong1 == 'ㄶ' else ('ㄹ' if jong1 == 'ㅀ' else '')
                return compose_syllable(cho1, jung1, new_jong1), compose_syllable(asp_map2[cho2], jung2, jong2)
            elif cho2 == 'ㅇ':  # ㅎ 탈락 (좋은 -> 조은)
                new_jong1 = 'ㄴ' if jong1 == 'ㄶ' else ('ㄹ' if jong1 == 'ㅀ' else '')
                return compose_syllable(cho1, jung1, new_jong1), compose_syllable('ㅇ', jung2, jong2)

        # -------------------------------------------------------------
        # Rule 3: 연음 규칙 (Liaison - 제13/14항)
        # 받침 뒤에 초성 'ㅇ'(모음)이 오는 경우
        # -------------------------------------------------------------
        if cho2 == 'ㅇ':
            if jong1 in DOUBLE_JONG_SPLIT:
                j_first, j_second = DOUBLE_JONG_SPLIT[jong1]
                # 겹받침 앞 글자는 남고 뒤 글자는 초성으로 이동
                return compose_syllable(cho1, jung1, j_first), compose_syllable(j_second, jung2, jong2)
            else:
                # 홑받침 전체가 다음 음절 초성으로 이동
                return compose_syllable(cho1, jung1, ''), compose_syllable(jong1, jung2, jong2)

        # -------------------------------------------------------------
        # Rule 4: 유음화 (Liquidization - 제20항)
        # ㄴ + ㄹ -> ㄹ + ㄹ / ㄹ + ㄴ -> ㄹ + ㄹ
        # -------------------------------------------------------------
        if jong1 == 'ㄴ' and cho2 == 'ㄹ':
            return compose_syllable(cho1, jung1, 'ㄹ'), compose_syllable('ㄹ', jung2, jong2)
        if jong1 in ('ㄹ', 'ㄾ', 'ㅀ', 'ㄼ') and cho2 == 'ㄴ':
            return compose_syllable(cho1, jung1, 'ㄹ'), compose_syllable('ㄹ', jung2, jong2)

        # -------------------------------------------------------------
        # Rule 5: 비음화 (Nasalization - 제18/19항)
        # ㄱ, ㄷ, ㅂ + ㄴ, ㅁ -> ㅇ, ㄴ, ㅁ + ㄴ, ㅁ
        # ㅁ, ㅇ + ㄹ -> ㅁ, ㅇ + ㄴ
        # ㄱ, ㅂ + ㄹ -> ㅇ, ㅁ + ㄴ (연쇄 비음화)
        # -------------------------------------------------------------
        effective_jong = NEUTRAL_JONG.get(jong1, jong1)

        # ㅁ, ㅇ + ㄹ -> ㅁ, ㅇ + ㄴ
        if jong1 in ('ㅁ', 'ㅇ') and cho2 == 'ㄹ':
            return c1, compose_syllable('ㄴ', jung2, jong2)

        # ㄱ, ㅂ + ㄹ -> ㅇ, ㅁ + ㄴ
        if effective_jong == 'ㄱ' and cho2 == 'ㄹ':
            return compose_syllable(cho1, jung1, 'ㅇ'), compose_syllable('ㄴ', jung2, jong2)
        if effective_jong == 'ㅂ' and cho2 == 'ㄹ':
            return compose_syllable(cho1, jung1, 'ㅁ'), compose_syllable('ㄴ', jung2, jong2)

        # ㄱ, ㄷ, ㅂ + ㄴ, ㅁ -> ㅇ, ㄴ, ㅁ
        if cho2 in ('ㄴ', 'ㅁ'):
            if effective_jong == 'ㄱ':
                return compose_syllable(cho1, jung1, 'ㅇ'), c2
            elif effective_jong == 'ㄷ':
                return compose_syllable(cho1, jung1, 'ㄴ'), c2
            elif effective_jong == 'ㅂ':
                return compose_syllable(cho1, jung1, 'ㅁ'), c2

        # -------------------------------------------------------------
        # Rule 6: 경음화 / 된소리되기 (Tensification - 제23항)
        # 받침 [ㄱ, ㄷ, ㅂ] + [ㄱ, ㄷ, ㅂ, ㅅ, ㅈ] -> [ㄲ, ㄸ, ㅃ, ㅆ, ㅉ]
        # -------------------------------------------------------------
        tense_map = {'ㄱ': 'ㄲ', 'ㄷ': 'ㄸ', 'ㅂ': 'ㅃ', 'ㅅ': 'ㅆ', 'ㅈ': 'ㅉ'}
        if effective_jong in ('ㄱ', 'ㄷ', 'ㅂ') and cho2 in tense_map:
            return compose_syllable(cho1, jung1, effective_jong), compose_syllable(tense_map[cho2], jung2, jong2)

        # 기본 자음 앞 겹받침 단순화 (제9항)
        if jong1 in NEUTRAL_JONG:
            return compose_syllable(cho1, jung1, NEUTRAL_JONG[jong1]), c2

        return c1, c2

_g2p_instance = KoreanG2PEngine()

def korean_text_to_phonemes(text: str) -> str:
    """Public API: Convert standard Korean orthographic text into phonetic Hangul sequence."""
    return _g2p_instance.convert(text)
