from concurrent.futures import ThreadPoolExecutor
from threading import Event, Thread

import pytest

from demo_safety import (
    MAX_PROCESS_REQUESTS_PER_HOUR,
    DemoSafetyGuard,
    demo_capabilities,
    question_within_limit,
)
from upload_lifecycle import read_upload_bytes


class FakeClock:
    def __init__(self, value=100.0):
        self.value = value

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def test_question_length_limit_includes_boundary_only():
    assert question_within_limit("x" * 1_000, 1_000)
    assert not question_within_limit("x" * 1_001, 1_000)


def test_public_upload_and_indexing_remain_available_when_gemini_is_disabled():
    can_upload, can_generate = demo_capabilities(
        public_demo_mode=True,
        gemini_demo_enabled=False,
        index_available=False,
    )

    assert can_upload is True
    assert can_generate is False


def test_public_generation_requires_enabled_gemini_and_an_index():
    assert demo_capabilities(True, False, True) == (True, False)
    assert demo_capabilities(True, True, False) == (True, False)
    assert demo_capabilities(True, True, True) == (True, True)


def test_disabled_generation_keeps_pdf_size_and_indexing_guards():
    can_upload, can_generate = demo_capabilities(True, False, False)
    assert can_upload is True
    assert can_generate is False

    class OversizedUpload:
        size = 5 * 1024 * 1024 + 1

        def getvalue(self):
            raise AssertionError("oversized upload must not be copied")

    with pytest.raises(ValueError, match="upload size limit"):
        read_upload_bytes(OversizedUpload(), max_bytes=5 * 1024 * 1024)

    guard = DemoSafetyGuard()
    assert guard.try_acquire_indexing()
    assert guard.try_acquire_indexing() is False
    guard.release_indexing()


def test_session_question_limit_and_cooldown_use_fake_clock():
    clock = FakeClock()
    guard = DemoSafetyGuard(clock)
    reason, now = guard.admit_gemini_request(0, None)
    assert reason is None and now == 100.0

    reason, _ = guard.admit_gemini_request(1, now)
    assert reason == "cooldown"

    clock.advance(20)
    reason, next_time = guard.admit_gemini_request(1, now)
    assert reason is None and next_time == 120.0

    reason, _ = guard.admit_gemini_request(10, None)
    assert reason == "session_limit"


def test_process_hour_budget_is_rolling_and_counts_failed_attempt_reservations():
    clock = FakeClock()
    guard = DemoSafetyGuard(clock)
    for count in range(MAX_PROCESS_REQUESTS_PER_HOUR):
        reason, _ = guard.admit_gemini_request(0, None)
        assert reason is None
        clock.advance(20)

    reason, _ = guard.admit_gemini_request(0, None)
    assert reason == "process_limit"

    clock.advance(3601)
    reason, _ = guard.admit_gemini_request(0, None)
    assert reason is None


def test_process_budget_admission_is_atomic_under_concurrency():
    clock = FakeClock()
    guard = DemoSafetyGuard(clock)
    start = Event()

    def admit(_):
        start.wait(2)
        return guard.admit_gemini_request(0, None)[0]

    with ThreadPoolExecutor(max_workers=MAX_PROCESS_REQUESTS_PER_HOUR * 2) as pool:
        futures = [pool.submit(admit, i) for i in range(MAX_PROCESS_REQUESTS_PER_HOUR * 2)]
        start.set()
        reasons = [future.result(timeout=2) for future in futures]

    assert reasons.count(None) == MAX_PROCESS_REQUESTS_PER_HOUR
    assert reasons.count("process_limit") == MAX_PROCESS_REQUESTS_PER_HOUR


def test_indexing_and_generation_permits_fail_fast_and_release():
    guard = DemoSafetyGuard()
    assert guard.try_acquire_indexing()
    assert guard.try_acquire_indexing() is False
    guard.release_indexing()
    assert guard.try_acquire_indexing()
    guard.release_indexing()

    assert guard.try_acquire_generation()
    assert guard.try_acquire_generation() is False
    guard.release_generation()
    assert guard.try_acquire_generation()
    guard.release_generation()


def test_generation_permit_is_shared_between_threads():
    guard = DemoSafetyGuard()
    started = Event()
    finish = Event()
    active = 0
    maximum = 0

    def hold_permit():
        nonlocal active, maximum
        assert guard.try_acquire_generation()
        active += 1
        maximum = max(maximum, active)
        started.set()
        finish.wait(2)
        active -= 1
        guard.release_generation()

    thread = Thread(target=hold_permit)
    thread.start()
    assert started.wait(1)
    assert guard.try_acquire_generation() is False
    finish.set()
    thread.join(2)
    assert not thread.is_alive()
    assert maximum == 1


def test_indexing_permit_is_released_after_indexing_exception():
    guard = DemoSafetyGuard()
    assert guard.try_acquire_indexing()
    try:
        with pytest.raises(RuntimeError, match="indexing failed"):
            raise RuntimeError("indexing failed")
    finally:
        guard.release_indexing()
    assert guard.try_acquire_indexing()
    guard.release_indexing()


@pytest.mark.parametrize("failure", ["retrieval failed", "provider setup failed"])
def test_generation_permit_is_released_after_pre_generation_failure(failure):
    guard = DemoSafetyGuard()
    assert guard.try_acquire_generation()
    try:
        with pytest.raises(RuntimeError, match=failure):
            raise RuntimeError(failure)
    finally:
        guard.release_generation()
    assert guard.try_acquire_generation()
    guard.release_generation()
