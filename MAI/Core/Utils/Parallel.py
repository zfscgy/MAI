import threading


def parallel(functions: list, arguments: list):
    res = [None for _ in functions]

    def call_save(idx):
        res[idx] = functions[idx](*arguments[idx])

    call_threads = []
    for i, func in enumerate(functions):
        call_threads.append(threading.Thread(
            target=call_save, args=(i,)
        ))
        call_threads[-1].start()

    for thread in call_threads:
        thread.join()

    return res
