"""No note blocks where the singer is silent (step 2b VAD + step 6b confidence).

SwiftF0 confidently tracks pitched reverb tails and separation bleed in pauses.
Step 11 only removes whole notes and only with >= 20 candidates, so short songs
and notes whose tail runs into the pause slipped through. These synthetic
fixtures pin the frame-level gate that fixes both.
"""

from __future__ import annotations

import numpy as np

from vocalchart.f0 import F0Result
from vocalchart.segment import SegmentConfig, build_dataclass, get_field, segment_notes

HOP = 0.016


def hz_of(midi):
    return 440.0 * 2.0 ** ((np.asarray(midi, float) - 69.0) / 12.0)


def f0_from(midi, conf):
    midi = np.asarray(midi, float)
    times = np.arange(midi.size) * HOP
    return build_dataclass(F0Result, times=times, hz=hz_of(midi),
                           confidence=np.asarray(conf, float),
                           hop_seconds=HOP, tuning_hz=440.0)


def frames(seconds):
    return int(round(seconds / HOP))


def spans(result):
    return [(float(get_field(n, "start_seconds")),
             float(get_field(n, "start_seconds")) + float(get_field(n, "duration_seconds")))
            for n in result.notes]


def phrase_pause_phrase(pause_level_db=-75.0, pause_conf=0.8):
    """1 s sung A4 at -12 dBFS, 1 s pitched 'pause' (reverb/bleed), 1 s sung C5."""
    a, p, c = frames(1.0), frames(1.0), frames(1.0)
    midi = np.concatenate([np.full(a, 69.0), np.full(p, 69.0), np.full(c, 72.0)])
    conf = np.concatenate([np.full(a, 0.95), np.full(p, pause_conf), np.full(c, 0.95)])
    level = np.concatenate([np.full(a, -12.0), np.full(p, pause_level_db), np.full(c, -12.0)])
    return f0_from(midi, conf), level


def test_silent_pause_produces_no_notes():
    f0, level = phrase_pause_phrase()
    res = segment_notes(f0, level_db=level)
    assert len(res.notes) == 2
    for start, end in spans(res):
        assert not (1.05 < start < 1.95) and not (1.05 < end < 1.95), spans(res)
    assert res.vad_gated_frames >= frames(1.0) - 2


def test_quiet_pitched_bleed_relative_to_song_is_silence():
    # -45 dBFS is above the absolute floor but 33 dB below the song's singing.
    f0, level = phrase_pause_phrase(pause_level_db=-47.0)
    res = segment_notes(f0, level_db=level)
    assert len(res.notes) == 2


def test_isolated_pitched_tail_in_short_song_is_removed():
    # Fewer than reject_level_min_notes notes: step 11 is skipped entirely,
    # which is exactly where the old code left the spurious block in.
    f0, level = phrase_pause_phrase()
    old = segment_notes(f0, SegmentConfig(vad_relative_db=None, vad_absolute_floor_db=None),
                        level_db=level)
    new = segment_notes(f0, level_db=level)
    assert old.rejected_silent == 0
    assert spans(old)[0][1] > 1.9           # old: A4 note swallowed the pause
    assert spans(new)[0][1] < 1.05          # new: A4 ends where singing ends


def test_voiced_segment_keeps_its_notes():
    n = frames(2.0) // 2 * 2
    midi = np.concatenate([np.full(n // 2, 64.0), np.full(n // 2, 67.0)])
    level = np.full(n, -18.0)
    level[n // 2:] = -30.0                  # softer second note, still singing
    res = segment_notes(f0_from(midi, np.full(n, 0.9)), level_db=level)
    assert [int(get_field(x, "midi")) for x in res.notes] == [64, 67]
    assert res.vad_gated_frames == 0


def test_note_edges_are_trimmed_to_the_sung_region():
    lead, sung, tail = frames(0.3), frames(1.0), frames(0.5)
    n = lead + sung + tail
    midi = np.full(n, 69.0)
    level = np.concatenate([np.full(lead, -70.0), np.full(sung, -15.0),
                            np.linspace(-55.0, -80.0, tail)])
    res = segment_notes(f0_from(midi, np.full(n, 0.9)), level_db=level)
    assert len(res.notes) == 1
    start, end = spans(res)[0]
    assert abs(start - 0.3) <= 2 * HOP
    assert abs(end - 1.3) <= 3 * HOP


def test_low_mean_confidence_note_is_rejected():
    n = frames(0.6)
    conf = np.full(n, 0.55)                 # passes 0.50 per frame, mean < 0.60
    res = segment_notes(f0_from(np.full(n, 60.0), conf))
    assert res.notes == []
    assert res.rejected_low_confidence == 1


def test_without_level_behaviour_is_unchanged():
    f0, _ = phrase_pause_phrase()
    res = segment_notes(f0)
    assert len(res.notes) >= 1
    assert res.vad_gated_frames == 0
    assert np.isnan(res.vad_threshold_db)


def test_gain_invariance_of_relative_gate():
    f0, level = phrase_pause_phrase(pause_level_db=-47.0)
    a = spans(segment_notes(f0, level_db=level))
    b = spans(segment_notes(f0, level_db=level - 10.0))
    assert a == b
