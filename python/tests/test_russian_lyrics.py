"""Russian lyrics support: tokenisation, encodings, syllables, alignment,
language selection, words.json/chart.json compatibility.

Everything is synthetic or mocked — no Whisper model is loaded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from vocalchart import lyrics as L
from vocalchart.chart import Chart
from vocalchart.lyrics import (Word, _normalise, _tokens, align_to_known_lyrics,
                               assign_syllables_to_notes, detect_script_language,
                               prompt_from_lyrics, read_lyrics_file,
                               resolve_language, syllabify)

REPO = Path(__file__).resolve().parents[2]

LYRICS_RU = """Ночь, улица, фонарь, аптека,
Бессмысленный и тусклый свет.
Живи ещё хоть четверть века —
Всё будет так. Исхода нет."""


@dataclass
class _N:
    startSeconds: float
    durationSeconds: float
    syllable: object = None
    syllableRole: str = "none"


# --------------------------------------------------------------------------- #
# tokens / normalisation
# --------------------------------------------------------------------------- #

def test_cyrillic_tokens_strip_punctuation_and_dashes():
    assert _tokens("Ночь, улица, фонарь — аптека!") == ["ночь", "улица", "фонарь", "аптека"]


def test_cyrillic_hyphen_splits_and_guillemets_ignored():
    assert _tokens("«Кто-то» сказал: …") == ["кто", "то", "сказал"]


def test_yo_kept_for_display_but_normalised_for_matching():
    assert _tokens("Ещё ёлка") == ["ещё", "ёлка"]
    assert _normalise("ЕЩЁ,") == "еще"
    assert _normalise("еще") == _normalise("ещё")


def test_english_tokens_unchanged():
    assert _tokens("Don’t stop me now, twenty-one") == ["don't", "stop", "me", "now",
                                                        "twenty", "one"]


# --------------------------------------------------------------------------- #
# encodings
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "cp1251", "utf-16"])
def test_read_lyrics_file_encodings(tmp_path, encoding):
    p = tmp_path / "lyrics.txt"
    p.write_bytes(LYRICS_RU.replace("\n", "\r\n").encode(encoding))
    text = read_lyrics_file(p)
    assert text == LYRICS_RU
    assert "\ufeff" not in text


def test_read_lyrics_file_english_utf8_unchanged(tmp_path):
    p = tmp_path / "lyrics.txt"
    p.write_text("Coming out of my cage and I've been doing just fine", encoding="utf-8")
    assert read_lyrics_file(p).startswith("Coming out")


def test_read_lyrics_file_missing_and_trivial(tmp_path):
    assert read_lyrics_file(tmp_path / "nope.txt") is None
    (tmp_path / "short.txt").write_text("ля-ля", encoding="utf-8")
    assert read_lyrics_file(tmp_path / "short.txt") is None


# --------------------------------------------------------------------------- #
# syllables
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("word,expected", [
    ("молоко", ["мо", "ло", "ко"]),
    ("она", ["о", "на"]),
    ("ночь", ["ночь"]),
    ("улица", ["у", "ли", "ца"]),
    ("аптека", ["ап", "те", "ка"]),
])
def test_russian_syllables(word, expected):
    assert syllabify(word) == expected


@pytest.mark.parametrize("word", ["бессмысленный", "здравствуйте", "объявление",
                                  "четверть", "ещё", "мойка", "семья", "любовь"])
def test_russian_syllable_count_equals_vowels_and_rejoins(word):
    parts = syllabify(word)
    assert "".join(parts) == word
    assert len(parts) == sum(ch in "аеёиоуыэюя" for ch in word)


def test_russian_vowel_fallback_without_pyphen(monkeypatch):
    monkeypatch.setattr(L, "_ru_pyphen_cache", False)
    assert syllabify("молоко") == ["мо", "ло", "ко"]
    assert syllabify("песня") == ["пес", "ня"]
    assert syllabify("мойка") == ["мой", "ка"]


def test_cmudict_not_consulted_for_cyrillic(monkeypatch):
    def boom(_w):
        raise AssertionError("CMUdict must not be used for Russian")
    monkeypatch.setattr(L, "_cmudict_syllable_count", boom)
    assert syllabify("фонарь") == ["фо", "нарь"]


def test_english_syllables_regression():
    assert syllabify("twenty-one")[-1] == "one"
    assert "".join(syllabify("cage")) == "cage"


# --------------------------------------------------------------------------- #
# language selection
# --------------------------------------------------------------------------- #

def test_detect_script_language():
    assert detect_script_language(LYRICS_RU) == "ru"
    assert detect_script_language("Coming out of my cage") == "en"
    assert detect_script_language("123 ...") is None


@pytest.mark.parametrize("choice,text,expected", [
    ("ru", None, "ru"), ("en", LYRICS_RU, "en"),
    ("auto", LYRICS_RU, "ru"), ("auto", "I want to break free", "en"),
    ("auto", None, None), (None, None, None), ("AUTO", LYRICS_RU, "ru"),
])
def test_resolve_language(choice, text, expected):
    assert resolve_language(choice, text) == expected


def test_resolve_language_rejects_unknown():
    with pytest.raises(ValueError):
        resolve_language("de")


def test_prompt_from_lyrics_cut_on_word_boundary():
    p = prompt_from_lyrics(LYRICS_RU * 5, max_chars=60)
    assert len(p) <= 60 and "\n" not in p
    assert (LYRICS_RU * 5).replace("\n", " ").startswith(p)
    assert prompt_from_lyrics(None) is None


# --------------------------------------------------------------------------- #
# alignment to known Russian lyrics, with ASR errors
# --------------------------------------------------------------------------- #

def _asr(pairs):
    out, t = [], 1.0
    for text in pairs:
        out.append(Word(text, t, t + 0.4, 0.9))
        t += 0.5
    return out


def test_align_russian_lyrics_with_asr_errors():
    # Truth: ночь улица фонарь аптека бессмысленный и тусклый свет
    # ASR: wrong word (фонар), yo/e mismatch, dropped word (и), inserted junk.
    asr = _asr(["Ночь,", "улица", "фонар", "аптека", "бессмысленный",
                "тусклый", "свет", "э"])
    out = align_to_known_lyrics(asr, "Ночь, улица, фонарь, аптека,\n"
                                     "Бессмысленный и тусклый свет.")
    assert [w.text for w in out] == ["ночь", "улица", "фонарь", "аптека",
                                     "бессмысленный", "и", "тусклый", "свет"]
    # Exactly matched words keep ASR timing.
    by = {w.text: w for w in out}
    assert by["ночь"].start == pytest.approx(1.0)
    assert by["свет"].start == pytest.approx(asr[6].start)
    # Replaced word takes the replaced ASR word's slot.
    assert by["фонарь"].start == pytest.approx(asr[2].start)
    # Inserted word lands between its neighbours, times monotone.
    assert by["бессмысленный"].start <= by["и"].start <= by["тусклый"].start
    starts = [w.start for w in out]
    assert starts == sorted(starts)


def test_align_matches_yo_and_e():
    asr = _asr(["живи", "еще", "хоть", "четверть", "века"])
    out = align_to_known_lyrics(asr, "Живи ещё хоть четверть века")
    assert [w.text for w in out] == ["живи", "ещё", "хоть", "четверть", "века"]
    assert out[1].probability == pytest.approx(0.9)       # matched, not interpolated


def test_english_alignment_regression():
    asr = _asr(["coming", "out", "of", "my", "cage", "and", "ive", "been"])
    out = align_to_known_lyrics(asr, "Coming out of my cage and I've been")
    assert [w.text for w in out] == ["coming", "out", "of", "my", "cage", "and",
                                     "i've", "been"]


# --------------------------------------------------------------------------- #
# word <-> note mapping with Russian words
# --------------------------------------------------------------------------- #

def test_one_russian_word_spans_multiple_notes():
    # "молоко" sung over three notes, one syllable each.
    notes = [_N(1.0, 0.3), _N(1.3, 0.3), _N(1.6, 0.4)]
    assign_syllables_to_notes(notes, [Word("молоко", 1.0, 2.0)])
    assert [n.syllable for n in notes] == ["мо", "ло", "ко"]
    assert [n.syllableRole for n in notes] == ["wordStart", "continuation", "continuation"]


def test_note_longer_than_word():
    # Word "ночь" ends at 1.4 but the sustained note runs to 3.0.
    notes = [_N(1.0, 2.0)]
    assign_syllables_to_notes(notes, [Word("ночь", 1.0, 1.4)])
    assert notes[0].syllable == "ночь"
    assert notes[0].durationSeconds == 2.0          # note length is NOT tied to the word


def test_word_longer_than_note_melisma():
    # One-syllable "свет" held across a pitch move: second note is a melisma.
    notes = [_N(1.0, 0.4), _N(1.4, 0.6)]
    assign_syllables_to_notes(notes, [Word("свет", 1.0, 2.0)])
    assert notes[0].syllable == "свет"
    assert notes[1].syllable is None and notes[1].syllableRole == "melisma"


def test_words_keep_sung_order_with_few_notes():
    notes = [_N(1.0, 1.0)]
    assign_syllables_to_notes(notes, [Word("и", 1.0, 1.2), Word("свет", 1.3, 1.9)])
    assert notes[0].syllable == "и свет"


# --------------------------------------------------------------------------- #
# words.json / pipeline with a mocked Whisper
# --------------------------------------------------------------------------- #

def _song(tmp_path, lyrics_bytes=None):
    from vocalchart.paths import SongPaths
    root = tmp_path / "song"
    (root / "stems").mkdir(parents=True)
    (root / "cache").mkdir()
    (root / "stems" / "vocals.wav").write_bytes(b"RIFF0000WAVE")
    if lyrics_bytes is not None:
        (root / "lyrics.txt").write_bytes(lyrics_bytes)
    return SongPaths(root)


def _fake_transcribe(calls, words, detected="ru"):
    def fake(audio, model_size="medium", device="auto", language="en", progress=None,
             initial_prompt=None, info_out=None):
        calls.append({"language": language, "prompt": initial_prompt})
        if info_out is not None:
            info_out["language"] = language or detected
            info_out["language_probability"] = 0.97
        return list(words)
    return fake


def test_run_lyrics_russian_cp1251_prompt_and_words_json(tmp_path, monkeypatch):
    from vocalchart import cli
    song = _song(tmp_path, LYRICS_RU.encode("cp1251"))
    calls = []
    monkeypatch.setattr(L, "transcribe_words",
                        _fake_transcribe(calls, _asr(["ночь", "улица", "фанарь", "аптека"])))
    info = {}
    words, source = cli._run_lyrics(song, True, lambda _m: None, "small",
                                    language="auto", info=info)
    assert calls[0]["language"] == "ru"
    assert calls[0]["prompt"].startswith("Ночь, улица")
    assert source == "forced-alignment"
    assert words[2].text == "фонарь"
    assert info["language"] == "ru"
    payload = json.loads(song.words_json.read_text(encoding="utf-8"))
    assert payload["language"] == "ru" and payload["languageChoice"] == "auto"
    assert "фонарь" in song.words_json.read_text(encoding="utf-8")   # not \\u-escaped


def test_run_lyrics_english_unchanged(tmp_path, monkeypatch):
    from vocalchart import cli
    song = _song(tmp_path, b"Coming out of my cage and I've been doing just fine")
    calls = []
    monkeypatch.setattr(L, "transcribe_words",
                        _fake_transcribe(calls, _asr(["coming", "out"])))
    cli._run_lyrics(song, True, lambda _m: None, "small", language="auto")
    assert calls[0] == {"language": "en", "prompt": None}


def test_run_lyrics_auto_without_lyrics_lets_whisper_detect(tmp_path, monkeypatch):
    from vocalchart import cli
    song = _song(tmp_path)
    calls = []
    monkeypatch.setattr(L, "transcribe_words",
                        _fake_transcribe(calls, _asr(["привет"]), detected="ru"))
    info = {}
    _, source = cli._run_lyrics(song, True, lambda _m: None, "small",
                                language="auto", info=info)
    assert calls[0]["language"] is None and source == "asr"
    assert info["language"] == "ru"


def test_old_words_cache_without_language_is_english(tmp_path, monkeypatch):
    from vocalchart import cli
    from vocalchart.paths import file_fingerprint
    song = _song(tmp_path)
    old = {"lyricSource": "asr", "model": "small",
           "inputs": {"audio": file_fingerprint(song.analysis_input()),
                      "lyrics": file_fingerprint(song.lyrics_txt)},
           "words": [{"text": "hello", "start": 1.0, "end": 1.4, "probability": 0.9}]}
    song.words_json.write_text(json.dumps(old), encoding="utf-8")
    calls = []
    monkeypatch.setattr(L, "transcribe_words", _fake_transcribe(calls, []))
    # auto / en: old English cache is reused, no re-transcription.
    words, _ = cli._run_lyrics(song, False, lambda _m: None, "small", language="auto")
    assert [w.text for w in words] == ["hello"] and calls == []
    # ru explicitly: cache is in the wrong language -> re-transcribe.
    cli._run_lyrics(song, False, lambda _m: None, "small", language="ru")
    assert calls and calls[0]["language"] == "ru"


# --------------------------------------------------------------------------- #
# chart.json compatibility
# --------------------------------------------------------------------------- #

def test_old_chart_loads_and_round_trips_without_new_fields():
    path = REPO / "songs" / "synth-demo" / "chart.json"
    original = json.loads(path.read_text(encoding="utf-8"))
    chart = Chart.read_json(path)
    chart.validate()
    assert chart.generator.lyricLanguage is None
    again = json.loads(chart.to_json())
    assert "lyricLanguage" not in again["generator"]
    assert again["notes"] == original["notes"]


def test_lyric_language_is_optional_and_round_trips():
    path = REPO / "songs" / "synth-demo" / "chart.json"
    chart = Chart.read_json(path)
    chart.generator.lyricLanguage = "ru"
    d = json.loads(chart.to_json())
    assert d["generator"]["lyricLanguage"] == "ru"
    assert Chart.from_dict(d).generator.lyricLanguage == "ru"
