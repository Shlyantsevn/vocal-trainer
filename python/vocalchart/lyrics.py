"""Stage 3: word-level lyric timing, and syllable assignment onto notes.

DESIGN.md section 4 stage 3 specifies force-alignment via stable-ts as the primary
path.  This module takes a different route to the same place, deliberately:

    faster-whisper (CTranslate2) instead of stable-ts / whisperx.

Why: stable-ts and whisperx both pull in **torch**, ~2.5 GB, and torch is not an
M2 dependency.  faster-whisper produces word-level timestamps natively, runs on
the GPU through CTranslate2, and needs no torch at all.  Measured on this
machine: 9.7x realtime for the `small` model on a real vocal stem.

The accuracy problem the design doc warns about is real and is NOT solved by
switching models.  Whisper sits at 32-35% word error on SUNG lyrics.  So:

  * If ``lyrics.txt`` exists, the ASR output is ALIGNED to that known text.
    ASR supplies the timing (which it is good at); the file supplies the words
    (which it is bad at).  This turns a 30% word-error problem into a pure
    timing problem without needing a forced aligner - or torch.

  * If it does not, the words are marked ``lyricSource: "asr"`` and the UI
    renders them in italic with an "auto-transcribed" badge.  Never present ASR
    lyrics as authoritative; the user would practise the wrong words.

Syllabification reconciles two sources, because neither is sufficient alone:
CMUdict vowel-stress counting is phonetically authoritative but gives no split
POINTS; Pyphen gives orthographic split points but is typographic hyphenation,
not phonetics ("ev-ery" versus the sung "eve-ry").
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence

__all__ = [
    "Word",
    "transcribe_words",
    "align_to_known_lyrics",
    "syllabify",
    "assign_syllables_to_notes",
    "read_lyrics_file",
    "resolve_language",
    "detect_script_language",
    "prompt_from_lyrics",
    "LANGUAGE_CHOICES",
    "DEFAULT_MODEL",
]

# --language on the CLI. "auto" resolves from lyrics.txt's alphabet when known
# text exists, otherwise lets Whisper detect it (see resolve_language).
LANGUAGE_CHOICES = ("auto", "ru", "en")

DEFAULT_MODEL = "medium"

# Notes whose midpoint falls within this much of a word's span are candidates
# for carrying its syllables. DESIGN.md section 4 stage 4.
NOTE_WORD_TOLERANCE = 0.080


@dataclass
class Word:
    text: str
    start: float
    end: float
    probability: float = 1.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


# --------------------------------------------------------------------------- #
# transcription
# --------------------------------------------------------------------------- #

def transcribe_words(
    audio_path: str | Path,
    model_size: str = DEFAULT_MODEL,
    device: str = "auto",
    language: Optional[str] = "en",
    progress: Optional[Callable[[str], None]] = None,
    initial_prompt: Optional[str] = None,
    info_out: Optional[dict] = None,
) -> List[Word]:
    """Word-level ASR over an isolated vocal stem.

    `language=None` lets Whisper detect the language from the first 30 s.
    `initial_prompt` biases the FIRST decoding window towards the expected
    vocabulary (with condition_on_previous_text=False it is not carried
    further); pass the start of lyrics.txt for Russian, where Whisper's
    unprompted sung-Russian output drifts badly. `info_out`, if given, receives
    ``language`` and ``language_probability`` as Whisper reported them.

    `device="auto"` tries CUDA and falls back to CPU, because CTranslate2 needs
    cuDNN DLLs on Windows that are not guaranteed to be present.
    """
    import itertools

    from faster_whisper import WhisperModel

    say = progress or (lambda _m: None)
    attempts = [("cuda", "float16"), ("cpu", "int8")] if device == "auto" \
        else [(device, "float16" if device == "cuda" else "int8")]

    model = None
    segments = None
    last_error: Optional[Exception] = None

    for dev, compute in attempts:
        try:
            say(f"loading whisper '{model_size}' on {dev}")
            model = WhisperModel(model_size, device=dev, compute_type=compute)

            # NO VAD. WhisperX's speech-tuned VAD is documented to drop whole
            # sung lines (issues #828, #536) and cannot be disabled;
            # faster-whisper's is optional, so simply do not enable it. The
            # input is already an isolated vocal stem.
            gen, _info = model.transcribe(
                str(audio_path),
                language=language,
                initial_prompt=initial_prompt,
                word_timestamps=True,
                beam_size=5,
                condition_on_previous_text=False,   # stops one bad line poisoning the rest
                vad_filter=False,
            )

            # transcribe() is LAZY - real inference, and the cuDNN DLL load
            # that is the documented Windows failure mode (DESIGN.md:375),
            # happens on the FIRST pull of the generator, not at model
            # construction. Pull it here, inside the per-device attempt, so a
            # CUDA-only failure falls back to CPU instead of surfacing later
            # as "charting without lyrics".
            if info_out is not None:
                info_out["language"] = getattr(_info, "language", language)
                info_out["language_probability"] = float(
                    getattr(_info, "language_probability", 0.0) or 0.0)

            it = iter(gen)
            first = next(it, None)
            segments = itertools.chain([first], it) if first is not None else iter(())
            break
        except Exception as exc:                      # noqa: BLE001
            last_error = exc
            model = None
            say(f"  {dev} unavailable ({type(exc).__name__}); trying next")

    if model is None or segments is None:
        raise RuntimeError(f"could not start faster-whisper: {last_error}")

    words: List[Word] = []

    for seg in segments:
        for w in (seg.words or []):
            text = w.word.strip()

            if not text:
                continue

            start, end = float(w.start), float(w.end)

            # Whisper occasionally emits a zero-length first word. Give it a
            # nominal span rather than dropping it, so the lyric line is intact.
            if end <= start:
                end = start + 0.12

            words.append(Word(text=text, start=start, end=end,
                              probability=float(getattr(w, "probability", 1.0))))

    words = _drop_repetition_hallucinations(words, say)
    say(f"transcribed {len(words)} words")
    return words


def _drop_repetition_hallucinations(
    words: List[Word],
    say: Callable[[str], None],
    run_length: int = 4,
) -> List[Word]:
    """Remove Whisper's signature stuck-token output.

    Fed near-silence — an intro, an instrumental break, or the wrong stem
    entirely — Whisper emits the same short token over and over. Measured on an
    ad-lib stem that was 87% silence: the entire transcript was
    "you you you you you".

    A real lyric does repeat words, but rarely the same one-or-two-letter token
    four times consecutively with no other word between. Drop those runs; keep
    everything else.
    """
    if len(words) < run_length:
        return words

    keep = [True] * len(words)
    i = 0

    while i < len(words):
        j = i

        while (j + 1 < len(words)
               and _normalise(words[j + 1].text) == _normalise(words[i].text)):
            j += 1

        if (j - i + 1) >= run_length and len(_normalise(words[i].text)) <= 4:
            for k in range(i, j + 1):
                keep[k] = False

        i = j + 1

    dropped = keep.count(False)

    if dropped:
        say(f"dropped {dropped} hallucinated repeat tokens (near-silence artefact)")

    return [w for w, k in zip(words, keep) if k]


# --------------------------------------------------------------------------- #
# alignment to known lyrics
# --------------------------------------------------------------------------- #

# Latin + Cyrillic letters, with an INTERNAL apostrophe kept ("don't").
# Hyphens, dashes (- – —), quotes («»“”), digits and punctuation all split.
_TOKEN = re.compile(r"[a-zа-яё]+(?:'[a-zа-яё]+)*")
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'"})
_CYRILLIC = re.compile(r"[а-яё]")
_LATIN = re.compile(r"[a-z]")


def _tokens(text: str) -> List[str]:
    """ALL word tokens in `text`, lowercased. "twenty-one" -> ["twenty", "one"],
    "Кто-то, ёлка!" -> ["кто", "то", "ёлка"]. ё is KEPT here: these tokens are
    what the chart displays. Comparison goes through _normalise()."""
    return _TOKEN.findall(text.lower().translate(_APOSTROPHES))


def _match_key(tok: str) -> str:
    """Comparison form of one token: ё -> е (Whisper and lyric sites use both)."""
    return tok.replace("ё", "е")


def _normalise(tok: str) -> str:
    """First word token only, in comparison form (ё -> е) — for comparing two
    single words. Use _tokens() when splitting free text, or a hyphenated
    compound loses its tail."""
    found = _tokens(tok)
    return _match_key(found[0]) if found else ""


def detect_script_language(text: str) -> Optional[str]:
    """"ru" / "en" from the alphabet of `text`, or None if it has no letters or
    is genuinely mixed (neither script reaches 70% of the letters)."""
    low = (text or "").lower()
    cyr = len(_CYRILLIC.findall(low))
    lat = len(_LATIN.findall(low))
    total = cyr + lat
    if total == 0:
        return None
    if cyr / total >= 0.7:
        return "ru"
    if lat / total >= 0.7:
        return "en"
    return None


def resolve_language(choice: Optional[str], lyrics_text: Optional[str] = None) -> Optional[str]:
    """Map the --language choice to what Whisper is given.

    "ru"/"en" are returned as-is. "auto" (or None) uses the alphabet of the
    known lyrics when there are any - deterministic, and an English lyrics.txt
    resolves to "en" exactly as before - and otherwise returns None so Whisper
    detects the language itself.
    """
    c = (choice or "auto").strip().lower()
    if c in ("ru", "en"):
        return c
    if c != "auto":
        raise ValueError(f"language must be one of {LANGUAGE_CHOICES}, got {choice!r}")
    return detect_script_language(lyrics_text) if lyrics_text else None


def prompt_from_lyrics(lyrics_text: Optional[str], max_chars: int = 200) -> Optional[str]:
    """First lines of the known lyrics, cut at a word boundary, as a Whisper
    initial_prompt. Whisper prompts are capped at 224 tokens and Cyrillic costs
    ~1 token per 2-3 characters, so 200 characters stays well inside it."""
    if not lyrics_text:
        return None
    flat = " ".join(lyrics_text.split())
    if len(flat) <= max_chars:
        return flat or None
    cut = flat[:max_chars]
    space = cut.rfind(" ")
    return (cut[:space] if space > 0 else cut) or None


def align_to_known_lyrics(
    asr_words: Sequence[Word],
    lyrics_text: str,
    progress: Optional[Callable[[str], None]] = None,
) -> List[Word]:
    """Replace ASR text with the TRUE words, keeping ASR timing.

    ASR is good at *when* and bad at *what* on sung audio. The known lyrics are
    the opposite. A sequence alignment between the two gets both, with no
    forced-aligner dependency.

    Unmatched true words are interpolated into the gap they belong in, so a line
    Whisper mangled still appears with plausible timing rather than vanishing.
    """
    say = progress or (lambda _m: None)

    # _tokens, not _normalise-per-whitespace-token: "twenty-one" must yield BOTH
    # words into the true stream, or the tail vanishes from every chart.
    true_tokens = _tokens(lyrics_text)

    if not true_tokens or not asr_words:
        return list(asr_words)

    asr_tokens = [_normalise(w.text) for w in asr_words]
    true_keys = [_match_key(t) for t in true_tokens]

    matcher = difflib.SequenceMatcher(a=asr_tokens, b=true_keys, autojunk=False)
    out: List[Word] = []
    matched = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                w = asr_words[i1 + k]
                out.append(Word(true_tokens[j1 + k], w.start, w.end, w.probability))
            matched += i2 - i1
            continue

        # For replace/insert/delete, spread the TRUE words evenly across the
        # time span the ASR words occupied. Timing is approximate here by
        # definition — but the words are right, which matters more.
        n_true = j2 - j1

        if n_true == 0:
            continue

        if i2 > i1:
            span_start, span_end = asr_words[i1].start, asr_words[i2 - 1].end
        elif out:
            span_start = span_end = out[-1].end
        else:
            span_start = span_end = asr_words[0].start if asr_words else 0.0

        span = max(span_end - span_start, 0.12 * n_true)
        step = span / n_true

        for k in range(n_true):
            s = span_start + k * step
            out.append(Word(true_tokens[j1 + k], s, s + step * 0.9, 0.5))

    out.sort(key=lambda w: w.start)
    say(f"aligned to lyrics.txt: {matched}/{len(asr_words)} ASR words matched exactly, "
        f"{len(out)} words out")
    return out


# --------------------------------------------------------------------------- #
# syllabification
# --------------------------------------------------------------------------- #

_cmudict_cache: Optional[dict] = None
_pyphen_cache = None


def _cmudict_syllable_count(word: str) -> Optional[int]:
    """Authoritative syllable COUNT from CMUdict vowel-stress digits."""
    global _cmudict_cache

    if _cmudict_cache is None:
        try:
            import cmudict

            _cmudict_cache = cmudict.dict()
        except Exception:                             # noqa: BLE001
            _cmudict_cache = {}

    prons = _cmudict_cache.get(word.lower())

    if not prons:
        return None

    return sum(1 for ph in prons[0] if ph[-1].isdigit())


_RU_VOWELS = set("аеёиоуыэюя")
_RU_SIGNS = set("ьъ")
_ru_pyphen_cache = None


def _ru_vowel_split(word: str) -> List[str]:
    """Russian syllables by vowel nuclei. One syllable per vowel; between two
    vowels a single consonant opens the next syllable ("мо-ло-ко"), a cluster
    leaves its first consonant behind ("пес-ня", "сер-дце"); й, ь and ъ always
    stay with the preceding syllable ("мой-ка", "объ-яв-ле-ни-е")."""
    nuclei = [i for i, ch in enumerate(word) if ch in _RU_VOWELS]
    if len(nuclei) <= 1:
        return [word]
    cuts = []
    for a, b in zip(nuclei, nuclei[1:]):
        between = word[a + 1:b]
        cut = a + 1
        if between:
            # Keep й/ь/ъ (and the consonant before a sign) on the left.
            lead = 0
            while lead < len(between) and between[lead] == "й":
                lead += 1
            rest = between[lead:]
            last_sign = max((k for k, ch in enumerate(rest) if ch in _RU_SIGNS), default=-1)
            if last_sign >= 0:
                lead += last_sign + 1
            elif len(rest) >= 2:
                lead += 1
            cut = a + 1 + lead
        cuts.append(cut)
    parts, prev = [], 0
    for c in cuts:
        parts.append(word[prev:c])
        prev = c
    parts.append(word[prev:])
    return [p for p in parts if p]


def _syllabify_ru(clean: str) -> List[str]:
    """Count = vowel count (authoritative, like CMUdict for English). Split
    points from Pyphen ru_RU when it agrees with that count; otherwise (Pyphen
    never leaves a one-letter syllable, so "о-на" comes back as "она") the
    vowel-nucleus rule."""
    global _ru_pyphen_cache
    target = sum(1 for ch in clean if ch in _RU_VOWELS)
    if target <= 1:
        return [clean]
    if _ru_pyphen_cache is None:
        try:
            import pyphen

            _ru_pyphen_cache = pyphen.Pyphen(lang="ru_RU")
        except Exception:                             # noqa: BLE001
            _ru_pyphen_cache = False
    if _ru_pyphen_cache:
        split = _ru_pyphen_cache.inserted(clean, "-").split("-")
        if all(split) and len(split) == target and "".join(split) == clean:
            return split
    return _ru_vowel_split(clean)


def syllabify(word: str) -> List[str]:
    """Split a word into sung syllables.

    Cyrillic words go through _syllabify_ru (vowel count + Pyphen ru_RU);
    CMUdict is consulted for Latin-script words only.

    Pyphen supplies the split POINTS, CMUdict the authoritative COUNT; when they
    disagree the Pyphen fragments are merged until the count matches. Pyphen
    alone is typographic hyphenation, which is not the same thing as phonetic
    syllabification.
    """
    global _pyphen_cache

    found = _tokens(word)

    if not found:
        return [word]

    # A hyphenated compound ("twenty-one") syllabifies per part, so the tail is
    # kept rather than silently dropped with the first token.
    if len(found) > 1:
        return [s for part in found for s in syllabify(part)]

    clean = found[0]

    if _CYRILLIC.search(clean):
        return _syllabify_ru(clean)

    if _pyphen_cache is None:
        try:
            import pyphen

            _pyphen_cache = pyphen.Pyphen(lang="en_US")
        except Exception:                             # noqa: BLE001
            _pyphen_cache = False

    parts = [clean]

    if _pyphen_cache:
        split = _pyphen_cache.inserted(clean, "-").split("-")

        if all(split):
            parts = split

    target = _cmudict_syllable_count(clean)

    if target is None or target <= 0 or target == len(parts):
        return parts

    if target > len(parts):
        # Trust CMUdict's count but we have nowhere to split; keep what we have
        # rather than inventing boundaries.
        return parts

    # Too many fragments: merge the shortest neighbours until the count matches.
    while len(parts) > target:
        i = min(range(len(parts) - 1), key=lambda k: len(parts[k]) + len(parts[k + 1]))
        parts[i:i + 2] = [parts[i] + parts[i + 1]]

    return parts


# --------------------------------------------------------------------------- #
# note assignment
# --------------------------------------------------------------------------- #

def assign_syllables_to_notes(
    notes: Sequence,
    words: Sequence[Word],
    tolerance: float = NOTE_WORD_TOLERANCE,
    progress: Optional[Callable[[str], None]] = None,
) -> int:
    """Attach syllables to note events IN PLACE. Returns how many notes got text.

    ONE TIME-ORDERED PASS. Every word is bucketed onto its nearest note and the
    buckets are emitted in time order, which guarantees two things that a
    greedy first-come pass does not:

      * no word is ever lost, and
      * the words READ IN THE ORDER THEY ARE SUNG.

    That second one is not cosmetic. An earlier greedy version appended
    leftover words to whichever note was nearest, producing
    "out Coming of cage my and been I've doing fine just" for a line that is
    actually "Coming out of my cage and I've been doing just fine". Unreadable,
    and worse than no lyrics at all.

    Multi-note words spread across the notes they occupy (DESIGN.md section 4
    stage 4); a syllable held across a pitch move leaves the later notes with
    syllable=None and syllableRole="melisma", so the renderer draws a tie
    instead of repeating the word. Notes with no word keep syllable=None, which
    is LEGAL and common — Whisper deletes non-lexical vocables, and those are
    exactly the sustained notes the highway most wants to show.
    """
    say = progress or (lambda _m: None)

    for n in notes:
        n.syllable = None
        n.syllableRole = "none"

    if not words or not notes:
        return 0

    order = sorted(range(len(notes)), key=lambda i: notes[i].startSeconds)
    mids = [notes[i].startSeconds + notes[i].durationSeconds * 0.5 for i in order]
    spans = [(notes[i].startSeconds,
              notes[i].startSeconds + notes[i].durationSeconds) for i in order]

    def _gap(i: int, w: Word) -> float:
        """Time between the word's span and note i's span. 0 == they overlap."""
        s, e = spans[i]
        return max(s - w.end, w.start - e, 0.0)

    def _overlap(i: int, w: Word) -> float:
        s, e = spans[i]
        return max(0.0, min(e, w.end) - max(s, w.start))

    # --- bucket every word onto its anchor note, in time order ---------------
    # Anchor = the overlapping note with the LARGEST overlap, earliest on a tie
    # — a word held across several notes must anchor where it BEGINS, not where
    # its midpoint happens to fall. Words overlapping nothing take the nearest
    # note by gap — CAPPED at `tolerance`. Without the cap, a hallucinated word
    # or one whose timestamp drifted into an instrumental break got slapped
    # onto a note seconds away, in the wrong phrase. Unassigned is better.
    buckets: dict = {}
    dropped_far = 0

    for w in sorted(words, key=lambda x: (x.start, x.end)):
        # Overlap quantised to 1 ms: two notes covering equal halves of a held
        # word must TIE (and the earlier one win), not be split by float dust.
        k = min(range(len(spans)),
                key=lambda i: (_gap(i, w), -round(_overlap(i, w), 3), spans[i][0]))

        # A word sung across several notes must anchor where it BEGINS (see
        # above). The largest-overlap rule alone anchored "мо-ло-ко" sung over
        # 0.3 + 0.3 + 0.4 s notes on the LAST note and left the first two
        # empty. So: the earliest overlapping note whose overlap is at least
        # half the best one - "at least half" keeps a word that merely grazes
        # the previous note's tail (ASR jitter) off that note.
        best = round(_overlap(k, w), 3)
        if best > 0.0:
            for i in range(k):
                if round(_overlap(i, w), 3) >= 0.5 * best:
                    k = i
                    break

        if _gap(k, w) > tolerance:
            dropped_far += 1
            continue

        buckets.setdefault(k, []).append(w)

    if dropped_far:
        say(f"left {dropped_far} words unassigned "
            f"(no note within {int(tolerance * 1000)} ms of them)")

    assigned = 0
    crowded = 0

    for k in sorted(buckets):
        bucket = buckets[k]

        # Syllables for this note's word(s), in sung order. Multi-syllable words
        # keep their internal hyphens so "gotta" reads as "got-ta".
        pieces = ["-".join(syllabify(w.text)) for w in bucket]

        if len(bucket) > 1:
            crowded += len(bucket) - 1

        # A single multi-syllable word may spread onto following EMPTY notes,
        # which is what actually happens when a word is sung across a pitch move.
        if len(bucket) == 1:
            w = bucket[0]
            syls = syllabify(w.text)
            free = [k]

            # Following bucket-free notes join this word while either (a) a
            # syllable still needs a note, or (b) the note starts under the
            # word's own span — (b) is a syllable HELD across a pitch move,
            # i.e. a melisma, which is what makes the else-branch below
            # reachable at all. (It used to be dead code: `free` was capped at
            # the syllable count, so the pipeline never emitted a melisma.)
            j = k + 1
            while j < len(mids) and j not in buckets:
                needs_syllable = len(free) < len(syls)
                under_word = spans[j][0] < w.end + tolerance

                if not (needs_syllable or under_word):
                    break

                free.append(j)
                j += 1

            if len(free) > 1:
                for pos, note_idx in enumerate(free):
                    target = notes[order[note_idx]]

                    if pos < len(syls):
                        target.syllable = syls[pos]
                        target.syllableRole = "wordStart" if pos == 0 else "continuation"
                        assigned += 1
                    else:
                        target.syllable = None
                        target.syllableRole = "melisma"
                continue

        target = notes[order[k]]
        target.syllable = " ".join(pieces)
        target.syllableRole = "wordStart"
        assigned += 1

    if crowded:
        say(f"{crowded} words shared a note with a neighbour (the segmenter found "
            f"fewer notes than sung syllables; words are kept in sung order)")

    say(f"assigned syllables to {assigned}/{len(notes)} notes")
    return assigned


def decode_lyrics_bytes(raw: bytes) -> str:
    """Decode a user-supplied lyrics file.

    Order matters: UTF-16 only with a BOM; then strict UTF-8 (with or without
    BOM — Notepad writes both); then CP1251, which is what a Russian Windows
    "ANSI" save produces. CP1251 decodes almost any byte string, so it must
    come after the strict UTF-8 attempt, never before.
    """
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    try:
        return raw.decode("cp1251")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")


def read_lyrics_file(path: Path) -> Optional[str]:
    """lyrics.txt if present and non-trivial, else None."""
    path = Path(path)
    if not path.exists():
        return None

    text = decode_lyrics_bytes(path.read_bytes()).replace("\r\n", "\n").strip()
    return text if len(text) > 20 else None
