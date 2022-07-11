"""
Assumptions:

- Spaces come after other punctuation characters
(i.e. "eats, shoots, and leaves" NOT "eats ,shoots ,and leaves"
"""

import json
import logging
import os
from random import sample, shuffle
from typing import Any, List

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

from helper import (Int_or_Tuple, Str_or_List, Str_or_List_or_Series,
                    display_or_print, get_tqdm, load_file,
                    mk_dir_if_does_not_exist, only_or_all, save_file,
                    str_or_list_or_series_to_list, str_or_list_to_list)

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
REQUIRED_ATTRS_IF_SPACES_FALSE = {
    'char_shift': int
}

# General messages
SAVED_RAW_SAMPLES = "Saved {num_samples} samples in 'X_RAW' and 'Y_RAW'"
SAVED_TOKENIZER = """Saved tokenizer with {num_categories} \
categories to {tokenizer_name}."""
SAVED_NUMPY_ARRAY = """Saved numpy array with shape {shape} to \
{numpy_asset_name}."""
SAVED_TOKENIZED_SAMPLES = """Saved {num_samples} tokenized samples to \
{tokenized_asset_name}."""
SAVED_MODEL_ATTRS = "Saved {num_attrs} model attributes to {model_attrs_path}"
MESSAGE_RAM_IN_USE = "RAM currently in use: {ram_in_use}%"
MESSAGE_GENERATING_RAW_SAMPLES = "Generating raw samples from data provided..."
MESSAGE_TOKENIZING_INPUTS = "Tokenizing model inputs (X)..."
MESSAGE_TOKENIZING_OUTPUTS = "Tokenizing model outputs (y)..."
MESSAGE_CONVERTING_INPUTS_TO_NUMPY = """Converting model inputs (X) to numpy \
format..."""
MESSAGE_CONVERTING_OUTPUTS_TO_NUMPY = """Converting model outputs (y) to numpy \
format..."""

# Warning messages
WARNING_INPUT_STR_TOO_SHORT = """Warning: length of input string is less than model \
sequence length."""

# Error messages
ERROR_INPUT_STR_TOO_LONG = """The sequence length for this feature restorer is \
{seq_length} and this input string has {len_input} non-feature characters."""
ERROR_TRAIN_OR_VAL = 'Parameter train_or_val must be "TRAIN" or "VAL".'
ERROR_ATTRS_AND_LOAD = 'You cannot specify both attrs and load_folder.'
ERROR_MISSING_ATTR = 'attrs must have key {reqd_attr}.'
ERROR_REQD_ATTR_TYPE = "{reqd_attr} should have type {reqd_type}."
ERROR_SPECIFY_ATTRS_OR_LOAD_FOLDER = """You must specify either attrs or \
load_folder."""
ERROR_SPACES_FALSE_NOT_IMPLEMENTED = 'Not implemented yet when spaces=False.'
ERROR_ONE_OF_EACH_FALSE_NOT_IMPLEMENTED = """Not implemented yet when \
one_of_each=False."""

CHUNKER_NUM_PREFIX_WORDS = 5
CHUNKER_NUM_PREFIX_CHARS = 10

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
            raise ValueError(ERROR_ATTRS_AND_LOAD)
        elif attrs is not None:
            self.init_from_attrs(attrs)
        elif load_folder is not None:
            self.init_from_load_folder(load_folder)
        else:
            raise ValueError(ERROR_SPECIFY_ATTRS_OR_LOAD_FOLDER)

    # ====================
    def init_from_attrs(self, attrs: dict):
        """Initialize a new instance from a dictionary of attributes"""

        self.check_reqd_attrs_provided(attrs)
        self.__dict__.update(attrs)
        mk_dir_if_does_not_exist(self.root_folder)
        self.assets = ASSETS
        self.set_models_path()
        self.set_feature_chars()
        self.save_class_attrs()

    # ====================
    def check_reqd_attrs_provided(self, attrs: dict):
        """Check presence and type of required attributes in dictionary
        of attributes"""

        for reqd_attr, reqd_type in REQUIRED_ATTRS.items():
            try:
                val = attrs[reqd_attr]
                if not isinstance(val, reqd_type):
                    raise ValueError(ERROR_REQD_ATTR_TYPE.format(
                        reqd_attr=reqd_attr,
                        reqd_type=str(reqd_type)
                    ))
            except KeyError:
                raise ValueError(ERROR_MISSING_ATTR.format(
                    reqd_attr=reqd_attr
                ))
        if attrs['spaces'] is False:
            for reqd_attr, reqd_type in \
             REQUIRED_ATTRS_IF_SPACES_FALSE.items():
                try:
                    val = attrs[reqd_attr]
                    if not isinstance(val, reqd_type):
                        raise ValueError(ERROR_REQD_ATTR_TYPE.format(
                            reqd_attr=reqd_attr,
                            reqd_type=str(reqd_type)
                        ))
                except KeyError:
                    raise ValueError(ERROR_MISSING_ATTR.format(
                        reqd_attr=reqd_attr
                    ))

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
        """Set models_path attribute

        Path for models is a directory named 'models' below the FeatureRestorer
        root folder"""

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
        """Output a table of asset names, file names, and whether each file
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

    # ====================
    def show_class_attrs(self):
        """Output a table of all the attributes of the class instance that do
        not refer to the model that is currently loaded"""

        class_attrs = self.get_class_attrs()
        class_attrs_df = pd.DataFrame.from_dict(class_attrs, orient='index')
        display_or_print(class_attrs_df)

    # ====================
    def get_class_attrs(self):
        """Get the attributes of the class instance that do not refer to the
        model that is currently loaded.

        Names of model-specific attributes begin with 'model_', so get all
        attributes whose names do not begin with 'model_'"""

        class_attrs = {
            attr: value for attr, value in self.__dict__.items()
            if not attr.startswith('model_')
        }
        return class_attrs

    # ====================
    def show_model_attrs(self):
        """Output a table of all the attributes of the class instance that refer
        to the model that is currently loaded"""

        model_attrs = self.get_model_attrs()
        model_attrs_df = pd.DataFrame.from_dict(model_attrs, orient='index')
        display_or_print(model_attrs_df)

    # ====================
    def get_model_attrs(self):
        """Get the attributes of the class instance that refer to the model that
        is currently loaded.

        Names of these attributes begin with 'model_'"""

        model_attrs = {
            attr: value for attr, value in self.__dict__.items()
            if attr.startswith('model_')
        }
        return model_attrs

    # === DATA GENERATION ===

    # ====================
    def load_data(self, data: List[str]):
        """Convert provided data into form required for model training"""

        self.generate_raw(data)
        self.show_ram_used()
        print()
        self.tokenize_inputs()
        self.show_ram_used()
        print()
        self.tokenize_outputs()
        self.show_ram_used()
        print()
        self.convert_inputs_to_numpy()
        self.show_ram_used()
        print()
        self.convert_outputs_to_numpy()
        self.show_ram_used()
        print()

    # ====================
    def generate_raw(self, data: List[str]):
        """Generate lists of raw X and y values to use as samples from datapoints
        (documents) in the data provided.

        Save in 'X_RAW' and 'Y_RAW' assets"""

        print(MESSAGE_GENERATING_RAW_SAMPLES)
        X = []
        y = []
        pbar = tqdm_(range(len(data)))
        for _ in pbar:
            pbar.set_postfix({
                'ram_usage': f"{psutil.virtual_memory().percent}%",
                'num_samples': len(X),
                'estimated_num_samples':
                    len(X) * (len(pbar) / (pbar.n + 1))
            })
            Xy = self.datapoint_to_Xy(data.pop(0))
            if Xy is not None:
                X_, y_ = Xy
                X.extend(X_)
                y.extend(y_)
        self.save_asset(X, 'X_RAW')
        self.save_asset(y, 'Y_RAW')
        print(SAVED_RAW_SAMPLES.format(num_samples=len(X)))

    # ====================
    def datapoint_to_Xy(self, datapoint: str) -> list:
        """Given a datapoint (i.e. a document in the data provided), generate a
        lists of X and y values for training."""

        if self.spaces is True:
            X, y = self.datapoint_to_Xy_spaces_true(datapoint)
        else:
            X, y = self.datapoint_to_Xy_spaces_false(datapoint)
        return X, y

    # ====================
    def datapoint_to_Xy_spaces_true(self, datapoint: str) -> list:
        """Generate X and y values from a datapoint (document) in the case that
        self.spaces=True.

        Generate a sample beginning at the start of each word (after each
        space)."""

        X = []
        y = []
        words = datapoint.split(' ')
        substrs = [' '.join(words[i:]) for i in range(len(words))]
        for substr in substrs:
            Xy = self.substr_to_Xy(substr)
            if Xy is not None:
                X_, y_ = Xy
                X.append(X_)
                y.append(y_)
        return X, y

    # ====================
    def datapoint_to_Xy_spaces_false(self, datapoint: str) -> list:
        """Generate X and y values from a datapoint (document) in the case that
        self.spaces=False.

        Use a sliding window beginning from the first character and shifting
        self.char_shift characters at each step."""

        X = []
        y = []
        start_char = 0
        char_shift = self.char_shift
        while start_char < len(datapoint):
            substr = datapoint[start_char:]
            Xy = self.substr_to_Xy(substr)
            if Xy is not None:
                X_, y_ = Xy
                X.append(X_)
                y.append(y_)
            start_char += char_shift
        return X, y

    # ====================
    def substr_to_Xy(self, substr: str) -> tuple:
        """Generate X and y values from a substring of a datapoint
        (document)"""

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
                raise ValueError(ERROR_ONE_OF_EACH_FALSE_NOT_IMPLEMENTED)
            y.append(this_class)
        assert len(X) == self.seq_length
        assert len(y) == self.seq_length
        return ''.join(X), y

    # ====================
    def tokenize_inputs(self):
        """Tokenize inputs (X)"""

        print(MESSAGE_TOKENIZING_INPUTS)
        self.tokenize('X_TOKENIZER', 'X_RAW', 'X_TOKENIZED', char_level=True)

    # ====================
    def tokenize_outputs(self):
        """Tokenize outputs (y)"""

        print(MESSAGE_TOKENIZING_OUTPUTS)
        self.tokenize('Y_TOKENIZER', 'Y_RAW', 'Y_TOKENIZED', char_level=False)

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
        print(SAVED_TOKENIZED_SAMPLES.format(
            num_samples=len(tokenized),
            tokenized_asset_name=tokenized_asset_name
        ))
        self.save_asset(tokenizer, tokenizer_name)
        print(SAVED_TOKENIZER.format(
            num_categories=self.get_num_categories(tokenizer_name),
            tokenizer_name=tokenizer_name
        ))

    # ====================
    def convert_inputs_to_numpy(self):
        """Convert inputs (X) to numpy format"""

        print(MESSAGE_CONVERTING_INPUTS_TO_NUMPY)
        self.pickle_to_numpy('X_TOKENIZED', 'X')

    # ====================
    def convert_outputs_to_numpy(self):
        """Convert outputs (y) to numpy format"""

        print(MESSAGE_CONVERTING_OUTPUTS_TO_NUMPY)
        self.pickle_to_numpy('Y_TOKENIZED', 'Y')

    # ====================
    def pickle_to_numpy(self, pickle_asset_name: str, numpy_asset_name: str):
        """Open a .pickle asset, convert to a numpy array, and save as a .npy
        asset"""

        data_pickle = self.get_asset(pickle_asset_name)
        data_np = np.array(data_pickle)
        self.save_asset(data_np, numpy_asset_name)
        print(SAVED_NUMPY_ARRAY.format(
            shape=str(data_np.shape),
            numpy_asset_name=numpy_asset_name
        ))

    # ====================
    def preview_samples(self, k: int = 10):

        X = self.get_asset('X', mmap=True)
        y = self.get_asset('Y', mmap=True)
        all_idxs = range(len(X))
        idxs = sample(all_idxs, k)
        outputs = []
        for idx in idxs:
            restored = self.Xy_to_output(X[idx], y[idx])
            outputs.append((f"{idx:,}", restored))
        outputs_df = pd.DataFrame(outputs, columns=['Index', 'Output'])
        prev_colwidth = pd.options.display.max_colwidth
        pd.set_option('display.max_colwidth', None)
        display_or_print(outputs_df)
        pd.set_option('display.max_colwidth', prev_colwidth)

    # === PREPROCESSING ===

    # ====================
    def preprocess_raw_str(self, raw_str):
        """Preprocess a raw string for input to a model"""

        if self.capitalisation is True:
            input_str = raw_str.lower()
        else:
            input_str = raw_str
        for fc in self.feature_chars:
            input_str = input_str.replace(fc, '')
        return input_str

    # ====================
    def input_str_to_model_input(self, input_str):
        """Prepare a raw string for input to a model"""

        tokenized = self.tokenize_input_str(input_str)
        encoded = self.encode_tokenized_str(tokenized)
        return encoded

    # ====================
    def tokenize_input_str(self, input_str):
        """Tokenize an input string"""

        input_str = self.impose_seq_length(input_str)
        tokenizer = self.get_asset('X_TOKENIZER')
        tokenized = tokenizer.texts_to_sequences([input_str])
        return tokenized

    # ====================
    def impose_seq_length(self, input_str):
        """Check that the length of an input string is less than or equal
        to the model sequence length."""

        if len(input_str) > self.seq_length:
            error_msg = ERROR_INPUT_STR_TOO_LONG.format(
                seq_len=self.seq_length,
                len_input_len=len(input_str)
            )
            raise ValueError(error_msg)
        if len(input_str) < self.seq_length:
            if len(input_str) < self.seq_length:
                # ⳨ chosen to trigger OOV. Change if restoring features
                # for Coptic language!
                # TODO: Concat zeros instead of using OOV
                # print(WARNING_INPUT_STR_TOO_SHORT)
                input_str = input_str + \
                    ('⳨' * (self.seq_length - len(input_str)))
        return input_str

    # ====================
    def encode_tokenized_str(self, tokenized: str):

        num_X_categories = self.get_num_categories('X_TOKENIZER')
        encoded = to_categorical(tokenized, num_X_categories)
        return encoded

    # === PREDICTION & PREVIEW

    # ====================
    def predict(self, raw_str: str):
        """Get the predicted output for a string of length less than or
        equal to the model sequence length"""

        input_str = self.preprocess_raw_str(raw_str)
        X_encoded = self.input_str_to_model_input(raw_str)
        predicted = self.model.predict(X_encoded)
        y = np.argmax(predicted, axis=2)[0]
        y_tokenizer = self.get_asset('Y_TOKENIZER')
        y_decoded = self.decode_class_list(y_tokenizer, y)
        output_parts = [self.char_and_class_to_output_str(X_, y_)
                        for X_, y_ in zip(input_str, y_decoded)]
        output = ''.join(output_parts)
        return output

    # ====================
    def predict_docs(self, docs: Str_or_List_or_Series) -> Str_or_List:
        """Get the predicted output for a single doc, or a list
        or pandas Series of docs."""

        docs = str_or_list_or_series_to_list(docs)
        outputs = []
        pbar = tqdm_(range(len(docs)))
        for i in pbar:
            pbar.set_postfix(
                {'ram_usage': f"{psutil.virtual_memory().percent}%"})
            outputs.append(self.predict_single_doc(docs[i]))
        return only_or_all(outputs)

    # ====================
    def predict_single_doc(self, raw_str: str) -> str:
        """Get the predicted output for a document (any length)."""

        input_str = self.preprocess_raw_str(raw_str)
        if self.spaces is True:
            output = self.predict_doc_spaces_true(input_str)
        else:
            output = self.predict_doc_spaces_false(input_str)
        return output

    # ====================
    def predict_doc_spaces_true(self, input_str: str) -> str:

        all_output = []
        prefix = ''
        while input_str:
            restore_until = self.seq_length - len(prefix)
            text_to_restore = prefix + input_str[:restore_until]
            input_str = input_str[restore_until:]
            chunk_restored = self.predict(text_to_restore).split(' ')
            prefix = ''.join(chunk_restored[-CHUNKER_NUM_PREFIX_WORDS:])
            all_output.extend(chunk_restored[:-CHUNKER_NUM_PREFIX_WORDS])
        output = ' '.join(all_output)
        # Add any text remaining in 'prefix'
        if prefix:
            output = output + ' ' + self.predict(prefix).strip()
        return output

    # ====================
    def predict_doc_spaces_false(self, input_str: str) -> str:

        all_output = []
        prefix = ''
        while input_str:
            restore_until = self.seq_length - len(prefix)
            text_to_restore = prefix + input_str[:restore_until]
            input_str = input_str[restore_until:]
            chunk_restored = self.predict(text_to_restore)
            prefix = ''.join(chunk_restored[-CHUNKER_NUM_PREFIX_CHARS:])
            all_output.extend(chunk_restored[:-CHUNKER_NUM_PREFIX_CHARS])
        output = ''.join(all_output)
        # Add any text remaining in 'prefix'
        if prefix:
            output = output + self.predict(prefix).strip()
        return output

    # ====================
    def Xy_to_output(self, X: list, y: list) -> str:
        """Generate a raw string (text with features) from model input (X)
        and output (y)"""

        X_tokenizer = self.get_asset('X_TOKENIZER')
        X_decoded = X_tokenizer.sequences_to_texts([X])[0].replace(' ', '')
        y_tokenizer = self.get_asset('Y_TOKENIZER')
        y_decoded = self.decode_class_list(y_tokenizer, y)
        output_parts = [self.char_and_class_to_output_str(X_, y_)
                        for X_, y_ in zip(X_decoded, y_decoded)]
        output = ''.join(output_parts)
        return output

    # === MODEL ADMIN & TRAINING ===

    # ====================
    def get_num_categories(self, tokenizers: Str_or_List) -> Int_or_Tuple:
        """Get the number of categories in one or more tokenizers.

        If a single tokenizer name is passed, the return value is an integer.
        If a list of tokenizer names is passed, the return value is a tuple of
        integers."""

        tokenizers = str_or_list_to_list(tokenizers)
        tokenizers = [self.get_asset(t) for t in tokenizers]
        num_categories = tuple([len(t.word_index) + 1 for t in tokenizers])
        return only_or_all(num_categories)

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
        self.train_val_split(' ')
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
        print(SAVED_MODEL_ATTRS.format(
            num_attrs=len(model_attrs.keys()),
            model_attrs_path=model_attrs_path
        ))

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
    def show_model_log_file(self):

        log_file_df = pd.read_csv(self.model_log_file)
        display_or_print(log_file_df)

    # ====================
    def get_model_root_path(self, model_name: str):

        return os.path.join(self.models_path, model_name)

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
            raise RuntimeError(ERROR_TRAIN_OR_VAL)

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

        print(MESSAGE_RAM_IN_USE.format(
            ram_in_use=psutil.virtual_memory().percent))
