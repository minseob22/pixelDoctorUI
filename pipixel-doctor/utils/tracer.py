# utils/tracer.py

import inspect
from functools import wraps


def trace(func):
    @wraps(func)
    def wrapper(*args, **kwargs):

        file_path = inspect.getfile(func)

        print(
            f"\n[TRACE] {file_path}"
        )

        print(
            f"[FUNC ] {func.__name__}() 시작"
        )

        result = func(*args, **kwargs)

        print(
            f"[DONE ] {func.__name__}() 완료"
        )

        return result

    return wrapper