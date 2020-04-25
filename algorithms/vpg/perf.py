from functools import wraps
from time import time


def timeit(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        ret = f(*args, **kwargs)
        end = time()
        print(f"Func {f.__name__} took {end - start:.5f} sec")
        return ret

    return wrapper
