"""
Assumptions:

- Spaces come after other punctuation characters
(i.e. "eats, shoots, and leaves" NOT "eats ,shoots ,and leaves"
"""

import json
import logging
import os
from random import shuffle
from typing import Any, List, Union

import keras
import numpy as np
import pandas as pd
import psutil
from keras.layers import LSTM, Bidirectional, Dense, TimeDistributed
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from helper import (display_or_print, get_tqdm, load_file,
                    mk_dir_if_does_not_exist, save_file)

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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
# tensorflow.get_logger().setLevel('ERROR')
# tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)


# ====================
class FeatureRestorer:

    # === CLASS INSTANCE ADMIN ===

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
        self.set_models_path()
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
    def set_models_path(self):
        """Set feature_chars attribute based on other_features and spaces
        attributes"""

        self.models_path = os.path.join(self.root_folder, 'models')
        mk_dir_if_does_not_exist(self.models_path)

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
        display_or_print(df)

    # === DATA GENERATION ===

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
        print(f"Saved {len(X_train)} samples in 'X_RAW' and 'Y_RAW'")

    # ====================
    def tokenize(self, tokenizer_name: str, raw_asset_name: str,
                 tokenized_asset_name: str, char_level: bool):
        """Open an asset, create and fit a Keras tokenizer, tokenize the
        asset, and save both the tokenizer and the tokenized data"""

        data = self.get_asset(raw_asset_name)
        tokenizer = Tokenizer(
            oov_token='OOV', filters='', char_level=char_level)
        tokenizer.fit_on_texts(data)
        tokenized = tokenizer.texts_to_sequences(data)
        self.save_asset(tokenized, tokenized_asset_name)
        print(f'Saved {len(tokenized)} tokenized samples to',
              f'{tokenized_asset_name}.')
        self.save_asset(tokenizer, tokenizer_name)
        print(f'Saved tokenizer with {self.get_num_categories(tokenizer_name)}',
              f'categories to {tokenizer_name}.')

    # ====================
    def pickle_to_numpy(self, pickle_asset_name: str, numpy_asset_name: str):
        """Open a .pickle asset, convert to a numpy array, and save as a .npy
        asset"""

        data_pickle = self.get_asset(pickle_asset_name)
        data_np = np.array(data_pickle)
        self.save_asset(data_np, numpy_asset_name)
        print(f'Saved numpy array with shape {str(data_np.shape)} to',
              f'{numpy_asset_name}.')

    # === PREDICTION ===

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
    def X_tokenize_input_str(self, input_str):

        tokenizer = self.get_asset('X_TOKENIZER')
        tokenized = tokenizer.texts_to_sequences([input_str])
        return tokenized

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
    def add_model(self, model_attrs: dict):

        model_name = model_attrs['model_name']
        if os.path.exists(self.get_model_root_path(model_name)):
            raise ValueError('A model with this name already exists!')
        self.__dict__.update(model_attrs)
        self.model = self.new_model()
        self.model_root_path = self.get_model_root_path(self.model_name)
        self.model_latest_path = os.path.join(self.model_root_path, 'latest')
        self.model.save(self.model_latest_path)
        self.model_checkpoints_folder = \
            os.path.join(self.model_root_path, 'checkpoints')
        self.model_log_file = os.path.join(self.model_root_path, 'log.csv')
        self.model_attrs_file = \
            os.path.join(self.model_root_path, MODEL_ATTRS_FNAME)
        self.train_val_split()
        self.save_model_attrs()

    # ====================
    def load_model(self, model_name: str):

        model_attrs = self.get_model_attrs_from_file(model_name)
        self.__dict__.update(model_attrs)
        self.model = keras.models.load_model(self.model_latest_path)
        try:
            log_df = pd.read_csv(self.model_log_file)
            last_epoch = max([int(e) for e in log_df['epoch'].to_list()])
        except FileNotFoundError:
            last_epoch = 0
        self.model_last_epoch = last_epoch

    # ====================
    def save_model_attrs(self):

        model_attrs = self.get_model_attrs()
        model_attrs_path = self.model_attrs_file
        save_file(model_attrs, model_attrs_path)
        print(
            f"Saved {len(model_attrs.keys())} model attributes to",
            model_attrs_path
        )

    # ====================
    def get_model_attrs(self):

        model_attrs = {
            attr: value for attr, value in self.__dict__.items()
            if attr.startswith('model_')
        }
        return model_attrs

    # ====================
    def get_class_attrs(self):

        class_attrs = {
            attr: value for attr, value in self.__dict__.items()
            if not attr.startswith('model_')
        }
        return class_attrs

    # ====================
    def train_model(self, epochs: int):

        num_train = len(self.train_or_val_idxs('TRAIN'))
        num_val = len(self.train_or_val_idxs('VAL'))
        batch_size = self.model_batch_size
        save_each_checkpoint = ModelCheckpoint(
            filepath=os.path.join(self.model_root_path, 'cp-{epoch:02d}'),
            save_freq='epoch'
        )
        save_latest_checkpoint = ModelCheckpoint(
            filepath=self.model_latest_path,
            save_freq='epoch'
        )
        csv_logger = CSVLogger(self.model_log_file, append=True)
        self.model.fit(
            self.data_loader('TRAIN', batch_size=batch_size),
            steps_per_epoch=(num_train // batch_size),
            validation_data=self.data_loader('VAL', batch_size=batch_size),
            validation_steps=(num_val // batch_size),
            callbacks=[
                save_each_checkpoint, save_latest_checkpoint, csv_logger],
            initial_epoch=self.model_last_epoch,
            epochs=(self.model_last_epoch + epochs),
        )
        self.model_last_epoch += epochs

    # ====================
    def get_model_attrs_from_file(self, model_name: str):

        model_root_path = self.get_model_root_path(model_name)
        model_attrs_path = os.path.join(model_root_path, MODEL_ATTRS_FNAME)
        return load_file(model_attrs_path)

    # ====================
    def show_model_attrs(self):

        model_attrs = self.get_model_attrs()
        model_attrs_df = pd.DataFrame.from_dict(model_attrs, orient='index')
        display_or_print(model_attrs_df)

    # ====================
    def show_class_attrs(self):

        class_attrs = self.get_class_attrs()
        class_attrs_df = pd.DataFrame.from_dict(class_attrs, orient='index')
        display_or_print(class_attrs_df)

    # ====================
    def show_model_log_file(self):

        log_file_df = pd.read_csv(self.model_log_file)
        display_or_print(log_file_df)

    # ====================
    def get_model_root_path(self, model_name: str):

        return os.path.join(self.models_path, model_name)

    # ====================
    def predict(self, raw_str: str):

        input_str = self.raw_str_to_input_str(raw_str)
        if len(input_str) < self.seq_length:
            # ⳨ chosen to trigger OOV. Change if restoring features for Coptic
            # language.
            print("Warning: length of input string is less than model sequence", 
                  "length")
            input_str = \
                input_str + ['⳨' for _ in range(self.seq_length - len(input_str)]
        tokenized = self.input_str_to_tokenized(input_str)
        num_X_categories = self.get_num_categories('X_TOKENIZER')
        X_encoded = to_categorical(tokenized, num_X_categories)
        predicted = self.model.predict(X_encoded)
        y = np.argmax(predicted, axis=2)[0]
        y_tokenizer = self.get_asset('Y_TOKENIZER')
        y_decoded = self.decode_class_list(y_tokenizer, y)
        output_parts = [self.char_and_class_to_output_str(X_, y_)
                        for X_, y_ in zip(input_str, y_decoded)]
        output = ''.join(output_parts)
        return output

    # ====================
    def predict_doc(self, raw_str: str) -> list:

        text = self.raw_str_to_input_str(raw_str)
        all_words = []
        prefix = ''
        while text:
            restore_until = self.seq_length - len(prefix)
            text_to_restore = prefix + text[:restore_until]
            text = text[restore_until:]
            chunk_restored = self.predict(text_to_restore).split()
            prefix = ''.join(chunk_restored[-5:])
            all_words.extend(chunk_restored[:-5])
        output = ' '.join(all_words)
        # Add any text remaining in 'prefix'
        if prefix:
            output = output + ' ' + self.predict(prefix).strip()
        return output

    # ====================
    def new_model(self):

        units = self.model_units
        dropout = self.model_dropout
        recur_dropout = self.model_recur_dropout
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
            return self.model_train_idxs
        elif train_or_val == 'VAL':
            return self.model_val_idxs
        else:
            raise RuntimeError('train_or_val must be "TRAIN" or "VAL".')

    # ====================
    def train_val_split(self):

        keep_size = self.model_keep_size
        val_size = self.model_val_size
        X = self.get_asset('X', mmap=True)
        all_idxs = range(len(X))
        if keep_size is not None:
            keep_idxs, _ = \
                train_test_split(all_idxs, test_size=(1.0-keep_size))
        else:
            keep_idxs = all_idxs
        self.model_train_idxs, self.model_val_idxs = \
            train_test_split(keep_idxs, test_size=val_size)
        self.save_class_attrs()

    # ====================
    def input_str_to_tokenized(self, input_str):

        if len(input_str) > self.seq_length:
            error_msg = 'The sequence length for this feature restorer is ' +\
                        f"{self.seq_length} and this input string has " +\
                        f"{len(input_str)} non-feature characters."
            raise ValueError(error_msg)
        return self.X_tokenize_input_str(input_str)

    # ====================
    def raw_str_to_input_str(self, raw_str):

        if self.capitalisation is True:
            input_str = raw_str.lower()
        else:
            input_str = raw_str
        for fc in self.feature_chars:
            input_str = input_str.replace(fc, '')
        return input_str

    # === STATIC METHODS ===

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
