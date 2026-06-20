import time
import functools

from groq import RateLimitError


def with_retry(max_retries: int = 3, base_delay_seconds: float = 8.0):
    """Retries a function on Groq rate limit errors with exponential backoff.

    Free-tier token-per-minute limits are easy to hit when running several
    LLM calls back to back (e.g. the eval harness, or even just a single
    /query request that fires rewrite + synthesis + guardrail calls in
    quick succession). Rather than crashing the whole request, wait and
    retry - the limit resets every minute, so a short wait usually clears it.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except RateLimitError as e:
                    last_error = e
                    delay = base_delay_seconds * (attempt + 1)
                    print(
                        f"  Rate limit hit, waiting {delay:.0f}s before retry "
                        f"({attempt + 1}/{max_retries})..."
                    )
                    time.sleep(delay)
            raise last_error

        return wrapper

    return decorator
