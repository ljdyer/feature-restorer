"""
Assumptions:

- Spaces come after other punctuation characters
(i.e. "eats, shoots, and leaves" NOT "eats ,shoots ,and leaves"
"""

import json
import os
from random import shuffle
from typing import Any, List, Union

import numpy as np
import pandas as pd
import psutil
import keras
from keras.layers import LSTM, Bidirectional, Dense, TimeDistributed
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from helper import get_tqdm, load_file, mk_dir_if_does_not_exist, save_file

CLASS_ATTRS_FNAME = 'CLASS_ATTRS.pickle'
MODEL_ATTRS_FNAME = 'MODEL_ATTRS.pickle'

ASSETS = {
    'CLASS_ATTRS': CLASS_ATTRS_FNAME,
    'X_TOKENIZER': 'X_TOKENIZER.pickle',
    'Y_TOKENIZER': 'y_TOKENIZER.pickle',
    'X_RAW': 'X_RAW.pickle',
    'Y_RAW': 'Y_RAW.pickle',
    'X_TOKENIZED': 'X_TOKENIZED.pickle',
    'Y_TOKENIZED': 'Y_TOKENIZED.pickle',
    'X': 'X.npy',
    'Y': 'Y.npy',
}

REQUIRED_ATTRS = {
    'root_folder': str,
    'capitalisation': bool,
    'spaces': bool,
    'other_features': list,
    'seq_length': int,
    'one_of_each': bool
}

tqdm_ = get_tqdm()


# ====================
class FeatureRestorer:

    # ====================
    def __init__(self, attrs: dict = None, load_folder: str = None):
        """Initialize a new instance.

        Exactly one of attrs or load_folder must be specified.

        If attrs is specified, a new instance is created with the attributes
        provided. If load_folder is specified, attributes are loaded from the
        load folder."""

        if attrs is not None and load_folder is not None:
            raise ValueError('You cannot specify both attrs and load_folder.')
        elif attrs is not None:
            self.init_from_attrs(attrs)
        elif load_folder is not None:
            self.init_from_load_folder(load_folder)
        else:
            raise ValueError('You must specify either attrs or load_folder.')

    # ====================
    def init_from_attrs(self, attrs: dict):
        """Initialize a new instance from a dictionary of attributes"""

        for reqd_attr, reqd_type in REQUIRED_ATTRS.items():
            try:
                val = attrs[reqd_attr]
                if not isinstance(val, reqd_type):
                    raise ValueError(
                        f'{reqd_attr} should have type {str(reqd_attr)}')
            except KeyError:
                raise ValueError(f'attrs must have key {reqd_attr}.')
        self.__dict__.update(attrs)
        mk_dir_if_does_not_exist(self.root_folder)
        self.assets = ASSETS
        self.set_model_path()
        self.set_feature_chars()
        self.save_class_attrs()

    # ====================
    def save_class_attrs(self):
        """Save the class attributes"""

        class_attrs = self.__dict__.copy()
        self.save_asset(class_attrs, 'CLASS_ATTRS')

    # ====================
    def init_from_load_folder(self, load_folder: str):
        """Initialize a new instance from a load folder"""

        class_attrs_path = os.path.join(load_folder, CLASS_ATTRS_FNAME)
        attrs = load_file(class_attrs_path)
        self.__dict__.update(attrs)

    # ====================
    def set_feature_chars(self):
        """Set feature_chars attribute based on other_features and spaces
        attributes"""

        if self.spaces:
            self.feature_chars = self.other_features + [' ']
        else:
            self.feature_chars = self.other_features

    # ====================
    def set_model_path(self):
        """Set feature_chars attribute based on other_features and spaces
        attributes"""

        self.model_path = os.path.join(self.load_folder, 'models')
        mk_dir_if_does_not_exist(self.model_path)

    # ====================
    def get_file_path(self, fname: str) -> str:
        """Get a file path from a file name by appending the root folder of
        the current instance."""

        return os.path.join(self.root_folder, fname)

    # ====================
    def asset_path(self, asset_name: str) -> str:
        """Get the path to an asset"""

        fname = self.assets[asset_name]
        return self.get_file_path(fname)

    # ====================
    def get_asset(self, asset_name: str, mmap: bool = False) -> Any:
        """Get an asset based on the asset name

        Use numpy memmap if mmap=True"""

        asset_path = self.asset_path(asset_name)
        return load_file(asset_path, mmap=mmap)

    # ====================
    def save_asset(self, data: Any, asset_name: str):
        """Save data to the asset file for the named asset"""

        asset_path = self.asset_path(asset_name)
        save_file(data, asset_path)

    # ====================
    def do_assets_exist(self):
        """Print a table of asset names, file names, and whether each file
        exists in the root folder."""

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
    def Xy_from_data(self, data: List[str]):
        """Get raw X and y lists from a list of strings

        Save in 'X_RAW' and 'Y_RAW' assets"""

        X_train = []
        y_train = []
        pbar = tqdm_(range(len(data)))
        for _ in pbar:
            pbar.set_postfix({
                'ram_usage': f"{psutil.virtual_memory().percent}%",
                'num_samples': len(X_train),
                'estimated_num_samples':
                    len(X_train) * (len(pbar) / (pbar.n + 1))
            })
            Xy = self.datapoint_to_Xy(data.pop(0))
            if Xy is not None:
                X, y = Xy
                X_train.extend(X)
                y_train.extend(y)
        self.save_asset(X_train, 'X_RAW')
        self.save_asset(y_train, 'Y_RAW')

    # ====================
    def tokenize(self, tokenizer_name: str, raw_asset_name: str,
                 tokenized_asset_name: str, char_level: bool):
        """Open an asset, create and fit a Keras tokenizer, tokenize the
        asset, and save both the tokenizer and the tokenized data"""

        data = self.get_asset(raw_asset_name)
        tokenizer = Tokenizer(char_level=char_level)
        tokenizer.fit_on_texts(data)
        tokenized = tokenizer.texts_to_sequences(data)
        self.save_asset(tokenized, tokenized_asset_name)
        print(f'Saved numpy array with shape {tokenized.shape}')
        self.save_asset(tokenizer, tokenizer_name)

    # ====================
    def pickle_to_numpy(self, pickle_asset_name: str, numpy_asset_name: str):
        """Open a .pickle asset, convert to a numpy array, and save as a .npy
        asset"""

        data_pickle = self.get_asset(pickle_asset_name)
        data_np = np.array(data_pickle)
        self.save_asset(data_np, numpy_asset_name)

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
    def add_model(self, model_name: str, model_attrs: dict):

        self.__dict__.update(model_attrs)
        model = self.new_model()
        model_root_path = self.get_model_root_path(model_name)
        if os.path.exists(model_root_path):
            raise ValueError('A model with this name already exists!')
        model_save_folder = os.path.join(model_root_path, 'model')
        model.save(model_save_folder)
        model_checkpoints_folder = os.path.join(model_root_path, 'checkpoints')
        model_log_file = os.path.join(model_root_path, 'log.csv')
        model_attrs_file = os.path.join(model_root_path, MODEL_ATTRS_FNAME)
        model_attrs.update({
            'model_name': model_name,
            'model_last_epoch': 0,
            'model_root_path': model_root_path,
            'model_save_folder': model_save_folder,
            'model_checkpoints_folder': model_checkpoints_folder,
            'model_log_file': model_log_file,
            'model_attrs_file': model_attrs_file
        })
        save_file(model_attrs, model_attrs_file)

    # ====================
    def load_model(self, model_name: str):

        self.model = keras.models.load_model
        model_root_path = self.get_model_root_path(model_name)
        model_attrs = load_file(os.path.join(model_root_path,
                                             MODEL_ATTRS_FNAME))
        self.__dict__.update(model_attrs)
        log_df = pd.read_csv(self.model_log_file)
        last_epoch = max([int(e) for e in log_df['epoch'].to_list])
        self.model_last_epoch = last_epoch

    # ====================
    def get_model_root_path(self, model_name: str):

        return os.path.join(self.model_path, model_name)

    # ====================
    def new_model(self, units: int, dropout: float, recur_dropout: float):

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
            X = self.get_asset('X', mmap=True)
            y = self.get_asset('Y', mmap=True)
            idxs = self.train_or_val_idxs(train_or_val)
            shuffle(idxs)
            num_iters = len(idxs) // batch_size
            num_X_categories, num_y_categories = \
                self.get_num_categories(['X_TOKENIZER', 'Y_TOKENIZER'])
            for i in range(num_iters):
                X_encoded = to_categorical(
                    X[idxs[(i*batch_size):((i+1)*batch_size)]],
                    num_X_categories
                )
                y_encoded = to_categorical(
                    y[idxs[(i*batch_size):((i+1)*batch_size)]],
                    num_y_categories
                )
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
    def train_val_split(self, keep_size: float = None, val_size: float = 0.2):

        X = self.get_asset('X', mmap=True)
        all_idxs = range(len(X))
        if keep_size is not None:
            keep_idxs, _ = \
                train_test_split(all_idxs, test_size=(1.0-keep_size))
        else:
            keep_idxs = all_idxs
        self.train_idxs, self.val_idxs = \
            train_test_split(keep_idxs, test_size=val_size)
        self.save_class_attrs()
