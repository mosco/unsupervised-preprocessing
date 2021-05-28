"""
Helper routines for dumping and loading data into gzip files.

Author:
    Amit Moscovich
    amit@moscovich.org
"""
import os
import pickle
import gzip
import errno
import inspect
import time


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PICKLES_PATH = os.path.join(CURRENT_DIR, 'pickles')


def mkdir_recursively(dirname):
    try:
        os.makedirs(dirname)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(dirname):
            pass
        else:
            raise


def _pickled_name(name):
    return os.path.join(PICKLES_PATH, name) + '.pickle.gz'


def dump(filename, **kwargs):
    mkdir_recursively(PICKLES_PATH)
    filename = _pickled_name(filename)

    #frame = inspect.getouterframes(inspect.currentframe())[1]
    #metadata = {'caller': '{}() in {}:{}'.format(frame.function, frame.filename, frame.lineno), 'caller_cnotext': frame.code_context, 'date': time.ctime()}
    metadata = {'date': time.ctime()}

    print('Saving to', filename)
    print("Saved fields: ", ', '.join(sorted(kwargs.keys())))

    with gzip.GzipFile(filename, 'wb') as f:
        pickle.dump({'metadata': metadata, 'data': kwargs}, f, 2)


class StructFromDict(object):
    def __init__(self, d): 
        self.__dict__.update(d)
    def __repr__(self):
        return repr(self.__dict__)


def load(name):
    filename = _pickled_name(name)
    print('Loading', filename)
    with gzip.GzipFile(filename, 'rb') as f:
        d = pickle.load(f)
    print('Creation time:', d['metadata']['date'])
    #print('Caller:', d['metadata']['caller'])
    return StructFromDict(d['data'])
