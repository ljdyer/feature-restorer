import os
import pickle

import pandas as pd
import numpy as np
from tqdm import tqdm as non_notebook_tqdm
from tqdm.notebook import tqdm as notebook_tqdm
from typing import Any, Union

Str_or_List = Union[str, list]
Int_or_Tuple = Union[int, tuple]
Str_or_List_or_Series = Union[str, list, pd.Series]
List_or_Tuple = Union[list, tuple]

STR_OR_LIST_OR_SERIES_TYPE_ERROR = \
    "Must have type list, str, or pandas.Series."
STR_OR_LIST_TYPE_ERROR = "Must have type list, str, or pandas.Series."


# ====================
def only_or_all(input_: List_or_Tuple) -> Any:
    """If the list or tuple contains only a single element, return that element.

    Otherwise return the original list or tuple."""

    if len(input_) == 1:
        return input_[0]
    else:
        return input_


# ====================
def str_or_list_or_series_to_list(
     input_: Str_or_List_or_Series) -> list:

    if isinstance(input_, str):
        return [input_]
    elif isinstance(input_, pd.Series):
        return input_.to_list()
    elif isinstance(input_, list):
        return input_
    else:
        raise TypeError(STR_OR_LIST_OR_SERIES_TYPE_ERROR)


# ====================
def str_or_list_to_list(
     input_: Str_or_List) -> list:

    if isinstance(input_, str):
        return [input_]
    elif isinstance(input_, list):
        return input_
    else:
        raise TypeError(STR_OR_LIST_TYPE_ERROR)


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
