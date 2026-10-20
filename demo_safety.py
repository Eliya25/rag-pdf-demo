"""Best-effort process-local admission and concurrency controls for the demo."""

from collections import deque
from threading import BoundedSemaphore, Lock
from time import monotonic
from typing import Callable


MAX_QUESTIONS_PER_SESSION = 10
MIN_REQUEST_INTERVAL_SECONDS = 20.0
MAX_PROCESS_REQUESTS_PER_HOUR = 20
REQUEST_WINDOW_SECONDS = 60 * 60


def question_within_limit(question: str, max_chars: int) -> bool:
    return len(question) <= max_chars


class DemoSafetyGuard:
    """Non-blocking semaphores and an atomic rolling request budget.

    State is intentionally in memory: it resets on process restart and is not
    shared across worker processes or replicas.
    """

    def __init__(self, clock: Callable[[], float] = monotonic) -> None:
        self._clock = clock
        self._budget_lock = Lock()
        self._indexing = BoundedSemaphore(1)
        self._generation = BoundedSemaphore(1)
        self._request_times: deque[float] = deque()

    def try_acquire_indexing(self) -> bool:
        return self._indexing.acquire(blocking=False)

    def release_indexing(self) -> None:
        self._indexing.release()

    def try_acquire_generation(self) -> bool:
        return self._generation.acquire(blocking=False)

    def release_generation(self) -> None:
        self._generation.release()

    def admit_gemini_request(
        self,
        session_question_count: int,
        last_session_request_at: float | None,
    ) -> tuple[str | None, float]:
        """Reserve one request or return a safe reason code.

        A reservation counts even if provider construction or generation later
        fails. `now` is returned for storing the per-session cooldown state.
        """
        with self._budget_lock:
            now = self._clock()
            if session_question_count >= MAX_QUESTIONS_PER_SESSION:
                return "session_limit", now
            if (
                last_session_request_at is not None
                and now - last_session_request_at < MIN_REQUEST_INTERVAL_SECONDS
            ):
                return "cooldown", now

            cutoff = now - REQUEST_WINDOW_SECONDS
            while self._request_times and self._request_times[0] <= cutoff:
                self._request_times.popleft()
            if len(self._request_times) >= MAX_PROCESS_REQUESTS_PER_HOUR:
                return "process_limit", now

            self._request_times.append(now)
            return None, now
