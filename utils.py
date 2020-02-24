import sys
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


class Profiler(object):
    def __init__(self):
        import cProfile
        self.pr = cProfile.Profile()
        self.pr.enable()
    def stop(self):
        self.pr.disable()
        import pstats
        ps = pstats.Stats(self.pr)
        ps.sort_stats('cumtime').print_stats(50)
    def __enter__(self):
        return self
    def __exit__(self, type, value, tb):
        self.stop()


def beep():
    import sys
    if sys.platform == 'darwin':
        import os
        os.system('say beep!')
        os.system('say bop!')
        os.system('say boop!')
    else:
        print('\a'*10)
