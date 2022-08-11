
def run_async_process(func):
    from multiprocessing import Process
    from functools import wraps

    @wraps(func)
    def async_func(*args, **kwargs):
        func_hl = Process(target = func, args = args, kwargs = kwargs)
        func_hl.daemon = True
        func_hl.start()
        return func_hl
    return async_func


def run_async_thread(func):
    from threading import Thread
    from functools import wraps

    @wraps(func)
    def async_func(*args, **kwargs):
        func_hl = Thread(target = func, args = args, kwargs = kwargs)
        func_hl.daemon = True
        func_hl.start()
        return func_hl
    return async_func


