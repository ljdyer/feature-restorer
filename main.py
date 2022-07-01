"""
Assumptions:

- Spaces come after other punctuation characters
(i.e. "eats, shoots, and leaves" NOT "eats ,shoots ,and leaves"
"""

import json
import os

import numpy as np
import psutil
from keras.layers import LSTM, Bidirectional, Dense, TimeDistributed
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm as non_notebook_tqdm
from tqdm.notebook import tqdm as notebook_tqdm
from helper import save_pickle, load_pickle

from helper import is_running_from_ipython, load_file, save_file

CLASS_ATTRS_FNAME = 'class_attrs.pickle'

ASSETS = {
    'CLASS_ATTRS': CLASS_ATTRS_FNAME,
    'TRAIN_DATA': 'train_data.npy',
    'X_TOKENIZER': 'X_tokenizer.pickle',
    'Y_TOKENIZER': 'y_tokenizer.pickle',
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
    def get_asset(self, asset_name: str):

        asset_path = self.asset_path(asset_name)
        return load_file(asset_path)

    # ====================
    def save_asset(self, data, asset_name: str):

        asset_path = self.asset_path(asset_name)
        save_file(data, asset_path)

    # ====================
    def save(self):

        class_attrs = self.__dict__.copy()
        self.save_asset(class_attrs, 'CLASS_ATTRS')

    # ====================
    def load_train_data(self, data: list, verbose: bool = False):

        self.verbose = verbose
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

        self.print_if_verbose("Got X_train and y_train")
        self.print_if_verbose(f"RAM used: {psutil.virtual_memory().percent}%")
        assert len(X_train) == len(y_train)
        num_samples = len(X_train)
        self.save_tmp_file(y_train, 'y_train_tmp.pickle')
        del y_train
        self.print_if_verbose("Saved and deleted y_train")
        self.print_if_verbose(f"RAM used: {psutil.virtual_memory().percent}%")
        X_tokenized = self.tokenize('X_TOKENIZER', X_train, char_level=True)
        self.save_tmp_file(X_tokenized, 'X_tok_tmp.pickle')
        del X_tokenized
        self.print_if_verbose("Saved and deleted X_tok")
        self.print_if_verbose(f"RAM used: {psutil.virtual_memory().percent}%")
        y_train = self.load_tmp_file('y_train_tmp.pickle')
        y_tokenized = self.tokenize('Y_TOKENIZER', y_train, char_level=False)
        del y_train
        X_tokenized = self.load_tmp_file('X_tok_tmp.pickle')
        self.print_if_verbose("Loaded x_tok and y_tok")
        self.print_if_verbose(f"RAM used: {psutil.virtual_memory().percent}%")
        all_train_data = []
        num_samples_at_start = len(X_tokenized)
        pbar = my_tqdm(range(num_samples_at_start))
        for i in pbar:
            pbar.set_postfix({
                'ram_usage': f"{psutil.virtual_memory().percent}%",
                'num_samples_remaining': len(X_tokenized),
                'total_samples': num_samples_at_start
            })
            all_train_data.append([X_tokenized.pop(0),
                                   y_tokenized.pop(0)])
        assert len(all_train_data) == num_samples_at_start
        assert len(X_tokenized) == 0
        del X_tokenized
        del y_tokenized
        all_train_data = np.array(all_train_data)
        assert len(all_train_data) == num_samples
        self.print_if_verbose(f"RAM used: {psutil.virtual_memory().percent}%")
        self.save_asset(all_train_data, 'TRAIN_DATA')
        del all_train_data
        self.print_if_verbose("Deleted all_train_data.")
        self.print_if_verbose(f"RAM used: {psutil.virtual_memory().percent}%")
        self.save()

    # ====================
    def print_if_verbose(self, msg: str):

        if self.verbose == True:
            print(msg)

    # ====================
    def tokenize(self, tokenizer_name: str, data: list, char_level: bool):

        tokenizer = Tokenizer(char_level=char_level)
        tokenizer.fit_on_texts(data)
        tokenized = tokenizer.texts_to_sequences(data)
        self.num_tokenizer_categories[tokenizer_name] = \
            len(tokenizer.word_index)
        self.save_asset(tokenizer, tokenizer_name)
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
    def onehot_encode_batch(self, sample_batch: np.ndarray) -> np.ndarray:
        """Convert a batch of samples to a 3d numpy array"""

        num_x_categories = self.num_tokenizer_categories['X_TOKENIZER']
        num_y_categories = self.num_tokenizer_categories['Y_TOKENIZER']
        X_seqs = []
        y_seqs = []
        for X, y in sample_batch:
            X = to_categorical(X, num_x_categories)
            y = to_categorical(y, num_y_categories)
            X_seqs.append(X)
            y_seqs.append(y)
        return (np.concatenate(X_seqs), np.concatenate(y_seqs))

    # ====================
    def data_loader(self, samples: np.ndarray, batch_size: int):
        """Iterator function to create batches"""

        while True:
            for idx in range(0, len(samples) - batch_size, batch_size):
                sample_batch = samples[idx: idx + batch_size]
                X, y = self.onehot_encode_batch(sample_batch)
                yield(X, y)

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
        # Encode input and output
        assert len(X) == self.seq_length
        assert len(y) == self.seq_length
        return ''.join(X), y

    # ====================
    def create_model(self, units: int, dropout: float, recur_dropout: float):

        num_x_categories = self.num_tokenizer_categories['X_TOKENIZER']
        model = Sequential()
        model.add(Bidirectional(
                    LSTM(
                        units,
                        return_sequences=True,
                        dropout=dropout,
                        recurrent_dropout=recur_dropout
                    ),
                    input_shape=(self.seq_length, num_x_categories)
                ))
        model.add(TimeDistributed(Dense(num_x_categories,
                                        activation='softmax')))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model
