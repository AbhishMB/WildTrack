import dataclasses

import pytest

from wildtrack.lock import NEVER_LOCK, LockManager, lockable_classes
from wildtrack.models import Detection


def det(tid, cls="elephant", conf=0.8, t=0.0, box=(0, 0, 10, 10)):
    return Detection(t, tid, cls, conf, box)


def feed(lm, frames):
    for k, dets in enumerate(frames):
        lm.observe(dets, k * 0.1)


def test_candidate_after_five_consecutive_frames(cfg):
    lm = LockManager(cfg.lock, cfg.classes)
    feed(lm, [[det(1)]] * 4)
    assert not lm.is_candidate(det(1))
    feed(lm, [[det(1)]])
    assert lm.is_candidate(det(1))


def test_streak_resets_on_low_confidence_or_missing_frame(cfg):
    lm = LockManager(cfg.lock, cfg.classes)
    feed(lm, [[det(1)]] * 4 + [[det(1, conf=0.3)]] + [[det(1)]] * 4)
    assert not lm.is_candidate(det(1))
    lm2 = LockManager(cfg.lock, cfg.classes)
    feed(lm2, [[det(1)]] * 4 + [[]] + [[det(1)]] * 4)
    assert not lm2.is_candidate(det(1))


def test_confusers_never_become_candidates(cfg):
    lm = LockManager(cfg.lock, cfg.classes)
    feed(lm, [[det(2, "cattle", 0.99)]] * 30)
    assert not lm.is_candidate(det(2, "cattle", 0.99))
    assert lm.best_candidate([det(2, "cattle", 0.99)]) is None


def test_person_is_never_lockable_even_in_test_mode(cfg):
    c = dataclasses.replace(cfg.classes, test_mode=True, test_classes=("cattle", "tarp", "person"))
    allowed = lockable_classes(c)
    assert "person" not in allowed and {"cattle", "tarp"} <= allowed
    lm = LockManager(cfg.lock, c)
    with pytest.raises(ValueError):
        lm.lock(det(1, "person"))
    assert NEVER_LOCK == {"person"}


def test_test_classes_ignored_outside_test_mode(cfg):
    c = dataclasses.replace(cfg.classes, test_mode=False, test_classes=("tarp",))
    assert "tarp" not in lockable_classes(c)
    assert "cattle" not in lockable_classes(cfg.classes)


def test_best_candidate_is_highest_confidence(cfg):
    lm = LockManager(cfg.lock, cfg.classes)
    frame = [det(1, conf=0.6), det(2, "leopard", conf=0.9), det(3, "cattle", conf=0.95)]
    feed(lm, [frame] * 5)
    assert lm.best_candidate(frame).track_id == 2


def test_auto_lock_needs_fifteen_frames_at_point_seven(cfg):
    lm = LockManager(cfg.lock, cfg.classes)
    feed(lm, [[det(1, conf=0.75)]] * 14)
    assert lm.auto_lock_ready([det(1, conf=0.75)]) is None
    feed(lm, [[det(1, conf=0.75)]])
    assert lm.auto_lock_ready([det(1, conf=0.75)]).track_id == 1
    lm2 = LockManager(cfg.lock, cfg.classes)
    feed(lm2, [[det(1, conf=0.65)]] * 30)  # candidate, but below auto_lock_conf
    assert lm2.is_candidate(det(1, conf=0.65)) and lm2.auto_lock_ready([det(1, conf=0.65)]) is None


def test_find_and_reacquire(cfg):
    lm = LockManager(cfg.lock, cfg.classes)
    lm.lock(det(5))
    assert lm.find_target([det(4), det(5)]).track_id == 5
    gate = {7: 5.0, 8: 1.2, 9: 0.4}  # mahalanobis distance by track id
    dets = [det(7), det(8), det(9, "leopard")]  # 9 is a different class
    got = lm.reacquire(dets, lambda d: gate[d.track_id])
    assert got.track_id == 8 and lm.locked_id == 8
    assert lm.reacquire([det(10)], lambda d: 3.5) is None  # outside the gate: a different animal
    assert lm.reacquire([det(11)], lambda d: None) is None  # can't be geolocated
