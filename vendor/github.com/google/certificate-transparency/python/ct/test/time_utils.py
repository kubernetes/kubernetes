"""Utilities for testing time-sensitive code."""

import contextlib
import os
import time


def get_timezone_environ():
    return os.environ.get("TZ", '')

def set_timezone_environ(new_timezone):
    os.environ["TZ"] = new_timezone
    time.tzset()

@contextlib.contextmanager
def timezone(temp_tz):
    """Sets the timezone, yields, then resets the timezone.

    Args:
        temp_tz: See https://docs.python.org/2/library/time.html#time.tzset
    """
    original_tz = get_timezone_environ()
    set_timezone_environ(temp_tz)
    try:
        yield
    finally:
        set_timezone_environ(original_tz)
