import os
import pickle

import numpy as np
from tqdm import tqdm as non_notebook_tqdm
from tqdm.notebook import tqdm as notebook_tqdm
from typing import Any


# ====================
def mk_dir_if_does_not_exist(path):

    if not os.path.exists(path):
        os.makedirs(path)


# ====================
def get_tqdm() -> type:
    """Return tqdm.notebook.tqdm if code is being run from a notebook,
    or tqdm.tqdm otherwise"""

    if is_running_from_ipython():
        tqdm_ = notebook_tqdm
    else:
        tqdm_ = non_notebook_tqdm
    return tqdm_


# ====================
def is_running_from_ipython():
    """Determine whether or not the current script is being run from
    a notebook"""

    try:
        # Notebooks have IPython module installed
        from IPython import get_ipython
        return True
    except ModuleNotFoundError:
        return False


# ====================
def display_or_print(obj: Any):

    if is_running_from_ipython():
        display(obj)
    else:
        print(obj)


# ====================
def load_file(fp: str, mmap: bool = False):
    """Load a .pickle file, or .npy file with mmap_mode either True or False"""

    _, fext = os.path.splitext(fp)
    if fext == '.pickle':
        return load_pickle(fp)
    elif fext == '.npy' and mmap is False:
        return load_npy(fp)
    elif fext == '.npy' and mmap is True:
        return np.load(fp, mmap_mode='r')
    else:
        raise RuntimeError('Invalid file ext!')


# ====================
def load_pickle(fp: str) -> Any:
    """Load a .pickle file and return the data"""

    with open(fp, 'rb') as f:
        unpickled = pickle.load(f)
    return unpickled


# ====================
def load_npy(fp: str) -> Any:
    """Load a .npy file and return the data"""

    with open(fp, 'rb') as f:
        opened = np.load(f)
    return opened


# ====================
def save_file(data: Any, fp: str):
    """Save data to a .pickle or .npy file"""

    _, fext = os.path.splitext(fp)
    if fext == '.pickle':
        save_pickle(data, fp)
    elif fext == '.npy':
        save_npy(data, fp)
    else:
        raise RuntimeError('Invalid file ext!')


# ====================
def save_pickle(data: Any, fp: str):
    """Save data to a .pickle file"""

    with open(fp, 'wb') as f:
        pickle.dump(data, f)


# ====================
def save_npy(data: Any, fp: str):
    """Save data to a .npy file"""

    with open(fp, 'wb') as f:
        np.save(f, data)
