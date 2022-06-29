import os
import pickle

import numpy as np


# ====================
def is_running_from_ipython():
    try:
        from IPython import get_ipython
        return True
    except ModuleNotFoundError:
        return False


# ====================
def load_file(fp: str):

    _, fext = os.path.splitext(fp)
    if fext == '.pickle':
        return load_pickle(fp)
    elif fext == '.npy':
        return load_npy(fp)
    else:
        raise RuntimeError('Invalid file ext!')


# ====================
def load_pickle(fp: str):

    with open(fp, 'rb') as f:
        unpickled = pickle.load(f)
    return unpickled


# ====================
def load_npy(fp: str):

    with open(fp, 'rb') as f:
        opened = np.load(f)
    return opened


# ====================
def save_file(data, fp: str):

    _, fext = os.path.splitext(fp)
    if fext == '.pickle':
        save_pickle(data, fp)
    elif fext == '.npy':
        save_npy(data, fp)
    else:
        raise RuntimeError('Invalid file ext!')


# ====================
def save_pickle(data, fp: str):

    with open(fp, 'wb') as f:
        pickle.dump(data, f)


# ====================
def save_npy(data, fp: str):

    with open(fp, 'wb') as f:
        np.save(f, data)
