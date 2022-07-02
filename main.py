"""
Assumptions:

- Spaces come after other punctuation characters
(i.e. "eats, shoots, and leaves" NOT "eats ,shoots ,and leaves"
"""

import json
import math
import os
from random import shuffle

import numpy as np
import pandas as pd
import psutil
from keras.layers import LSTM, Bidirectional, Dense, TimeDistributed
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm as non_notebook_tqdm
from tqdm.notebook import tqdm as notebook_tqdm
from typing import List, Union

from helper import is_running_from_ipython, load_file, save_file

CLASS_ATTRS_FNAME = 'class_attrs.pickle'

ASSETS = {
    'CLASS_ATTRS': CLASS_ATTRS_FNAME,
    'TRAIN_DATA': 'train_data.npy',
    'X_TOKENIZER': 'X_tokenizer.pickle',
    'Y_TOKENIZER': 'y_tokenizer.pickle',
    'X_TRAIN_RAW': 'X_train.pickle',
    'Y_TRAIN_RAW': 'y_train.pickle',
    'X_TRAIN_TOKENIZED': 'X_train_tokenized.pickle',
    'Y_TRAIN_TOKENIZED': 'y_train_tokenized.pickle',
    'X_TRAIN_NPY': 'X_train.npy',
    'Y_TRAIN_NPY': 'y_train.npy',
    'X_TRAIN': 'X_train.npy',
    'X_VAL': 'X_val.npy',
    'Y_TRAIN': 'y_train.npy',
    'Y_VAL': 'y_val.npy'
}

# ====================
if is_running_from_ipython():
    my_tqdm = notebook_tqdm
else:
    my_tqdm = non_notebook_tqdm


# ====================
class FeatureRestorer:
    """
    Required attributes for new instance:

    root_folder: str,
    capitalisation: bool,
    spaces: bool,
    other_features: list,
    seq_length: int,
    one_of_each: bool = True,
    """

    # ====================
    def __init__(self, attrs: dict = None, load_folder: str = None):

        if attrs is not None:
            self.__dict__.update(attrs)
            if self.spaces:
                self.feature_chars = self.other_features + [' ']
            else:
                self.feature_chars = self.other_features
            self.assets = ASSETS
            self.num_tokenizer_categories = {}
            if not os.path.exists(self.root_folder):
                os.makedirs(self.root_folder)
            self.save()
        elif load_folder is not None:
            class_attrs_path = os.path.join(load_folder, CLASS_ATTRS_FNAME)
            attrs = load_file(class_attrs_path)
            self.__dict__.update(attrs)
        else:
            raise ValueError('You must specify either attrs or load_path.')

    # ====================
    def get_file_path(self, fname: str):

        return os.path.join(self.root_folder, fname)

    # ====================
    def save_tmp_file(self, data, fname: str):

        fpath = self.get_file_path(fname)
        save_file(data, fpath)

    # ====================
    def load_tmp_file(self, fname: str):

        fpath = self.get_file_path(fname)
        return load_file(fpath)

    # ====================
    def asset_path(self, asset_name: str):

        fname = self.assets[asset_name]
        return self.get_file_path(fname)

    # ====================
    def get_asset(self, asset_name: str, mmap: bool = False):

        asset_path = self.asset_path(asset_name)
        return load_file(asset_path, mmap=mmap)

    # ====================
    def save_asset(self, data, asset_name: str):

        asset_path = self.asset_path(asset_name)
        save_file(data, asset_path)

    # ====================
    def save(self):

        class_attrs = self.__dict__.copy()
        self.save_asset(class_attrs, 'CLASS_ATTRS')

    # ====================
    def do_assets_exist(self):

        output = []
        asset_list = self.assets.copy()
        for asset_name, asset_fname in asset_list.items():
            fpath = self.asset_path(asset_name)
            output.append({
                'Asset name': asset_name,
                'Asset path': asset_fname,
                'Exists': True if os.path.exists(fpath) else False}
            )
        df = pd.DataFrame(output)
        print(df)

    # ====================
    def load_train_data(self, data: list):

        X_train = []
        y_train = []
        pbar = my_tqdm(range(len(data)))
        for i in pbar:
            pbar.set_postfix({
                'ram_usage': f"{psutil.virtual_memory().percent}%",
                'num_samples': len(X_train),
                'estimated_num_samples':
                    len(X_train) * (len(pbar) / (pbar.n + 1))
            })
            Xy = self.datapoint_to_Xy(data[i])
            if Xy is not None:
                X, y = Xy
                X_train.extend(X)
                y_train.extend(y)
        self.save_asset(X_train, 'X_TRAIN_RAW')
        self.save_asset(y_train, 'Y_TRAIN_RAW')

    # ====================
    def pickle_to_numpy(self, pickle_asset_name: str, numpy_asset_name: str):

        data_pickle = self.get_asset(pickle_asset_name)
        data_np = np.array(data_pickle)
        self.save_asset(data_np, numpy_asset_name)

    # ====================
    def tokenize(self, tokenizer_name: str, raw_asset_name: str,
                 tokenized_asset_name: str, char_level: bool):

        data = self.get_asset(raw_asset_name)
        tokenizer = Tokenizer(char_level=char_level)
        tokenizer.fit_on_texts(data)
        tokenized = tokenizer.texts_to_sequences(data)
        self.num_tokenizer_categories[tokenizer_name] = \
            len(tokenizer.word_index)
        self.save_asset(tokenized, tokenized_asset_name)
        print(f'Saved numpy array with shape {tokenized.shape}')
        self.save_asset(tokenizer, tokenizer_name)

    # ====================
    def get_num_categories(self, tokenizers: Union[List[str], str]):

        if isinstance(tokenizers, str):
            tokenizer = self.get_asset(tokenizers)
            num_categories = len(tokenizer.word_index) + 1
        elif isinstance(tokenizers, list):
            tokenizers = [self.get_asset(t) for t in tokenizers]
            num_categories = tuple([len(t.word_index) + 1 for t in tokenizers])
        return num_categories

    # ====================
    def Xy_to_output(self, X: list, y: list) -> str:

        X_tokenizer = self.get_asset('X_TOKENIZER')
        X_decoded = X_tokenizer.sequences_to_texts([X])[0].replace(' ', '')
        y_tokenizer = self.get_asset('Y_TOKENIZER')
        y_decoded = self.decode_class_list(y_tokenizer, y)
        output_parts = [self.char_and_class_to_output_str(X_, y_)
                        for X_, y_ in zip(X_decoded, y_decoded)]
        output = ''.join(output_parts)
        return output

    # ====================
    def preview_Xy(self, X: list, y: list):

        X_tokenizer = self.get_asset('X_TOKENIZER')
        X = X_tokenizer.sequences_to_texts([X])[0].replace(' ', '')
        y_tokenizer = self.get_asset('Y_TOKENIZER')
        y = self.decode_class_list(y_tokenizer, y)
        X_preview = ''
        y_preview = ''
        assert len(X) > 10
        for i in range(len(X)):
            reqd_length = max(len(X[i]), len(y[i])) + 1
            X_preview = X_preview + (str(X[i])).ljust(reqd_length)
            y_preview = y_preview + (str(y[i])).ljust(reqd_length)
        print(X_preview)
        print(y_preview)

    # ====================
    @staticmethod
    def decode_class_list(tokenizer, encoded: list) -> list:

        index_word = json.loads(tokenizer.get_config()['index_word'])
        decoded = [index_word[str(x)] for x in encoded]
        return decoded

    # ====================
    @staticmethod
    def char_and_class_to_output_str(X_: str, y_: str) -> str:

        if len(y_) > 0 and y_[0] == 'u':
            X_ = X_.upper()
            y_ = y_[1:]
        return X_ + y_

    # ====================
    @staticmethod
    def show_ram_used():

        print(f"RAM used: {psutil.virtual_memory().percent}%")

    # ====================
    def datapoint_to_Xy(self, datapoint: str) -> list:

        if self.spaces:
            X = []
            y = []
            words = datapoint.split()
            substrs = [' '.join(words[i:]) for i in range(len(words))]
            for substr in substrs:
                Xy = self.substr_to_Xy(substr)
                if Xy is not None:
                    X_, y_ = Xy
                    X.append(X_)
                    y.append(y_)
        else:
            raise ValueError('Not implemented yet when spaces=False.')
        return X, y

    # ====================
    def substr_to_Xy(self, substr: str) -> tuple:

        X = []
        y = []
        chars = list(substr)
        if chars[0] in self.feature_chars:
            # Substring can't begin with a feature char
            return None
        for _ in range(self.seq_length):
            this_class = ''
            feature_chars_encountered = []
            # Get the next letter
            try:
                this_char = chars.pop(0)
            except IndexError:
                # Substr not long enough
                return None
            # Check for capitalisation
            if self.capitalisation and this_char.isupper():
                X.append(this_char.lower())
                this_class = 'U' + this_class
            else:
                X.append(this_char)
            # Check for other features
            while chars and chars[0] in self.feature_chars:
                this_feature_char = chars.pop(0)
                feature_chars_encountered.append(this_feature_char)
            if self.one_of_each:
                this_class = ''.join(
                        [this_class] +
                        [f for f in self.feature_chars
                         if f in feature_chars_encountered]
                    )
            else:
                raise ValueError('Not implemented yet when one_of_each=False.')
            y.append(this_class)
        assert len(X) == self.seq_length
        assert len(y) == self.seq_length
        return ''.join(X), y

    # ====================
    def create_model(self, units: int, dropout: float, recur_dropout: float):

        num_X_categories, num_y_categories = \
            self.get_num_categories(['X_TOKENIZER', 'Y_TOKENIZER'])
        model = Sequential()
        model.add(Bidirectional(
                    LSTM(
                        units,
                        return_sequences=True,
                        dropout=dropout,
                        recurrent_dropout=recur_dropout
                    ),
                    input_shape=(self.seq_length, num_X_categories)
                ))
        model.add(TimeDistributed(Dense(num_y_categories,
                                        activation='softmax')))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    # ====================
    def data_loader(self, train_or_val: str, batch_size: int):
        """Iterator function to create batches"""

        while True:
            X = self.get_asset('X_TRAIN_NPY', mmap=True)
            y = self.get_asset('Y_TRAIN_NPY', mmap=True)
            idxs = self.train_or_val_idxs(train_or_val)
            shuffle(idxs)
            num_iters = len(idxs) // batch_size
            num_X_categories, num_y_categories = \
                self.get_num_categories(['X_TOKENIZER', 'Y_TOKENIZER'])
            for i in range(num_iters):
                X_encoded = to_categorical(
                    X[idxs[(i*batch_size):((i+1)*batch_size)]], num_X_categories)
                y_encoded = to_categorical(
                    y[idxs[(i*batch_size):((i+1)*batch_size)]], num_y_categories)
                yield (X_encoded, y_encoded)

    # ====================
    def train_or_val_idxs(self, train_or_val: str):

        if train_or_val == 'TRAIN':
            return self.train_idxs
        elif train_or_val == 'VAL':
            return self.val_idxs
        else:
            raise RuntimeError('train_or_val must be "TRAIN" or "VAL".')

    # ====================
    def train_val_split(self, test_size=0.2):

        X = self.get_asset('X_TRAIN_NPY')
        all_idxs = range(len(X))
        self.train_idxs, self.test_idxs = \
            train_test_split(all_idxs, test_size=test_size)
