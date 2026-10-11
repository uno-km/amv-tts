# Copyright (c) 2026 Uno Kim / AMEVA Foundation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 OR Commercial-License
import re
from typing import Dict, List, Tuple

COMMON_EN_KO_LEXICON: Dict[str, str] = {
    # Greetings & Salutations
    "hello": "헬로",
    "hi": "하이",
    "bye": "바이",
    "good morning": "굿모닝",
    "good night": "굿나잇",
    "welcome": "웰컴",
    "thanks": "땡큐",
    "thank you": "땡큐",
    "sorry": "쏘리",
    "please": "플리즈",
    
    # Pronouns & Auxiliaries
    "i am": "아이엠",
    "i am sam": "아이엠 샘",
    "i'am": "아이엠",
    "i'm": "아임",
    "you are": "유아",
    "you're": "유어",
    "we are": "위아",
    "it's": "잇츠",
    "this is": "디스이즈",
    "that is": "댓이즈",
    "there is": "데어리즈",
    "my": "마이",
    "your": "유어",
    "your face": "유어 페이스",
    "our": "아워",
    "his": "히즈",
    "her": "허",
    "their": "데어",
    
    # Common Nouns & Slang
    "sam": "샘",
    "kate": "케이트",
    "john": "존",
    "david": "데이비드",
    "alex": "알렉스",
    "parrot": "패럿",
    "face": "페이스",
    "fake": "페이크",
    "hate": "헤이트",
    "love": "러브",
    "like": "라이크",
    "game": "게임",
    "play": "플레이",
    "player": "플레이어",
    "user": "유저",
    "friend": "프렌드",
    "team": "팀",
    "boss": "보스",
    "hero": "히어로",
    "monster": "몬스터",
    "star": "스타",
    "music": "뮤직",
    "song": "송",
    "voice": "보이스",
    "sound": "사운드",
    "audio": "오디오",
    "video": "비디오",
    "phone": "폰",
    "smart": "스마트",
    "camera": "카메라",
    
    # Tech, Hardware, AI Terms
    "ai": "에이아이",
    "api": "에이피아이",
    "cpu": "씨피유",
    "gpu": "지피유",
    "npu": "엔피유",
    "soc": "에스오씨",
    "vulkan": "벌칸",
    "opencl": "오픈씨엘",
    "onnx": "온닉스",
    "ncnn": "엔씨엔엔",
    "arm": "암",
    "qualcomm": "퀄컴",
    "snapdragon": "스냅드래곤",
    "adreno": "아드레노",
    "exynos": "엑시노스",
    "mali": "말리",
    "galaxy": "갤럭시",
    "samsung": "삼성",
    "apple": "애플",
    "google": "구글",
    "android": "안드로이드",
    "termux": "터먹스",
    "linux": "리눅스",
    "windows": "윈도우",
    "python": "파이썬",
    "code": "코드",
    "coding": "코딩",
    "model": "모델",
    "engine": "엔진",
    "system": "시스템",
    "server": "서버",
    "client": "클라이언트",
    "network": "네트워크",
    "data": "데이터",
    "memory": "메모리",
    "cache": "캐시",
    "buffer": "버퍼",
    "speed": "스피드",
    "fast": "패스트",
    "power": "파워",
    "powerful": "파워풀",
    "test": "테스트",
    "benchmark": "벤치마크",
    "debug": "디버그",
    "error": "에러",
    "bug": "버그",
    "patch": "패치",
    "update": "업데이트",
    "release": "릴리즈",
    "version": "버전",
    
    # Common Adjectives & Interjections
    "best": "베스트",
    "good": "굿",
    "bad": "배드",
    "cool": "쿨",
    "hot": "핫",
    "nice": "나이스",
    "great": "그레이트",
    "perfect": "퍼펙트",
    "super": "슈퍼",
    "ultra": "울트라",
    "pro": "프로",
    "max": "맥스",
    "mini": "미니",
    "lite": "라이트",
    "free": "프리",
    "open": "오픈",
    "close": "클로즈",
    "start": "스타트",
    "stop": "스톱",
    "reset": "리셋",
    "ok": "오케이",
    "okay": "오케이",
    "no": "노",
    "yes": "예스",
    "oh": "오",
    "wow": "와우",
    "oops": "웁스",
}

def fallback_g2p_word(word: str) -> str:
    w = word.lower().strip()
    if not w:
        return ""
    if w in COMMON_EN_KO_LEXICON:
        return COMMON_EN_KO_LEXICON[w]

    if len(w) <= 4 and w.isalpha():
        letter_map = {
            "a": "에이", "b": "비", "c": "씨", "d": "디", "e": "이",
            "f": "에프", "g": "지", "h": "에이치", "i": "아이", "j": "제이",
            "k": "케이", "l": "엘", "m": "엠", "n": "엔", "o": "오",
            "p": "피", "q": "큐", "r": "알", "s": "에스", "t": "티",
            "u": "유", "v": "브이", "w": "더블유", "x": "엑스", "y": "와이", "z": "지",
        }
        vowels = set("aeiou")
        if not any(c in vowels for c in w) or w.isupper():
            return "".join(letter_map.get(c, c) for c in w)

    out = []
    i = 0
    while i < len(w):
        matched = False
        for en, ko in [("tion", "션"), ("sion", "션"), ("ing", "잉"), ("er", "어"), ("or", "어"), ("ar", "아"), ("ee", "이"), ("oo", "우"), ("ea", "이"), ("ou", "아우"), ("ai", "에이"), ("ay", "에이"), ("oa", "오")]:
            if w[i:i+len(en)] == en:
                out.append(ko)
                i += len(en)
                matched = True
                break
        if matched:
            continue

        c = w[i]
        char_map = {
            "a": "아", "b": "브", "c": "크", "d": "드", "e": "에",
            "f": "프", "g": "그", "h": "흐", "i": "이", "j": "즈",
            "k": "크", "l": "르", "m": "므", "n": "느", "o": "오",
            "p": "프", "q": "큐", "r": "르", "s": "스", "t": "트",
            "u": "우", "v": "브", "w": "우", "x": "크스", "y": "이", "z": "즈",
        }
        out.append(char_map.get(c, c))
        i += 1

    return "".join(out)

class PhoneticTransliterationPipeline:
    def __init__(self, custom_lexicon: Dict[str, str] | None = None):
        self.lexicon = dict(COMMON_EN_KO_LEXICON)
        if custom_lexicon:
            self.lexicon.update(custom_lexicon)

        sorted_phrases = sorted(self.lexicon.keys(), key=len, reverse=True)
        escaped_phrases = [re.escape(p) for p in sorted_phrases]
        self.phrase_regex = re.compile(
            r"\b(" + "|".join(escaped_phrases) + r")\b",
            re.IGNORECASE
        )
        self.eng_word_regex = re.compile(r"\b([a-zA-Z]+)\b")

    def transliterate(self, text: str) -> str:
        if not text:
            return ""

        result = text

        def _replace_known(m: re.Match) -> str:
            key = m.group(1).lower()
            return self.lexicon.get(key, m.group(0))

        result = self.phrase_regex.sub(_replace_known, result)

        def _replace_unknown(m: re.Match) -> str:
            word = m.group(1)
            if not word.isascii():
                return word
            return fallback_g2p_word(word)

        result = self.eng_word_regex.sub(_replace_unknown, result)

        result = re.sub(r"키{3,}", "키키키", result)
        result = re.sub(r"하{3,}", "하하하", result)
        result = re.sub(r"크{3,}", "크크크", result)
        result = re.sub(r"ㅋ{2,}", "크크", result)
        result = re.sub(r"ㅎ{2,}", "하하", result)

        result = re.sub(r"\s+([,.?!;:])", r"\1", result)
        result = re.sub(r"[ \t]+", " ", result)

        return result.strip()

_GLOBAL_PIPELINE: PhoneticTransliterationPipeline | None = None

def get_transliteration_pipeline() -> PhoneticTransliterationPipeline:
    global _GLOBAL_PIPELINE
    if _GLOBAL_PIPELINE is None:
        _GLOBAL_PIPELINE = PhoneticTransliterationPipeline()
    return _GLOBAL_PIPELINE

def transliterate_mixed_text(text: str) -> str:
    return get_transliteration_pipeline().transliterate(text)
