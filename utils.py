import collections
import time


def groupby(seq, keyfunc, mapfunc = None):
    d = collections.defaultdict(lambda:[])
    if mapfunc == None:
        for x in seq:
            d[keyfunc(x)].append(x)
    else:
        for x in seq:
            d[keyfunc(x)].append(mapfunc(x))
    return d


class Timer(object):
    def __init__(self, text = None):
        if text != None:
            print('{}:'.format(text)), 
        sys.stdout.flush()
        self.start_clock = time.clock()
        self.start_time = time.time()
    def stop(self):
        print(f'Wall time: {time.time() - self.start_time:.2f} seconds.  CPU time: {time.clock() - self.start_clock:.2f} seconds.')
    def __enter__(self):
        return self
    def __exit__(self, type, value, tb):
        self.stop()

