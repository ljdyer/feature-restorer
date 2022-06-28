"""
Assumptions:

- Spaces come after other punctuation characters
(i.e. "eats, shoots, and leaves" NOT "eats ,shoots ,and leaves"
"""

CLASS_ATTRS_FNAME = 'class_attrs.pickle'

ASSETS = {
    'CLASS_ATTRS': CLASS_ATTRS_FNAME,
    'X_TRAIN':  'X_train.pickle',
    'Y_TRAIN': 'y_train.pickle',
    'X_TOKENIZER': 'X_tokenizer.pickle',
    'Y_TOKENIZER': 'y_tokenizer.pickle',
}


import json
import os
import pickle

import numpy as np
import psutil
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm as non_notebook_tqdm
from tqdm.notebook import tqdm as notebook_tqdm


# ====================
def is_running_from_ipython():
    try:
        from IPython import get_ipython
        return True
    except ModuleNotFoundError:
        return False
    

# ====================
if is_running_from_ipython():
    my_tqdm = notebook_tqdm
else:
    my_tqdm = non_notebook_tqdm


# ====================
def load_pickle(fp: str):

    with open(fp, 'rb') as f:
        unpickled = pickle.load(f)
    return unpickled


# ====================
def save_pickle(data, fp: str):

    with open(fp, 'wb') as f:
        pickle.dump(data, f)


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
            if not os.path.exists(self.root_folder):
                os.makedirs(self.root_folder)
            self.save()
        elif load_folder is not None:
            class_attrs_path = os.path.join(load_folder, CLASS_ATTRS_FNAME)
            attrs = load_pickle(class_attrs_path)
            self.__dict__.update(attrs)
        else:
            raise ValueError('You must specify either attrs or load_path.')

    # ====================
    def asset_path(self, asset_name: str):

        return os.path.join(self.root_folder, self.assets[asset_name])

    # ====================
    def get_asset(self, asset_name: str):

        asset_path = self.asset_path(asset_name)
        return load_pickle(asset_path)

    # ====================
    def load_asset(self, asset_name: str):

        self.loaded_assets[asset_name] = self.get_asset(asset_name)
        return self.loaded_assets[asset_name]

    # ====================
    def save_asset(self, data, asset_name: str):

        asset_path = self.asset_path(asset_name)
        save_pickle(data, asset_path)

    # ====================
    def save(self):

        class_attrs = self.__dict__.copy()
        self.save_asset(class_attrs, 'CLASS_ATTRS')

    # ====================
    def load_train_data(self, data: list):

        X_train = []
        y_train = []
        pbar = my_tqdm(range(len(data)))
        for i in pbar:
            pbar.set_postfix({
                'ram_usage': psutil.virtual_memory().percent,
                'num_samples': len(X_train),
                'estimated_num_samples': len(X_train) * (len(pbar) / (pbar.n + 1))
            })
            Xy = self.datapoint_to_Xy(data[i])
            if Xy is not None:
                X, y = Xy
                X_train.extend(X)
                y_train.extend(y)
        
        X_tokenizer = Tokenizer(char_level=True)
        X_tokenizer.fit_on_texts(X)
        X_train_tokenized = X_tokenizer.texts_to_sequences(X)
        self.save_asset(X_tokenizer, 'X_TOKENIZER')
        self.save_asset(X_train_tokenized, 'X_TRAIN')
        y_tokenizer = Tokenizer()
        y_tokenizer.fit_on_texts(y)
        y_train_tokenized = y_tokenizer.texts_to_sequences(y)
        self.save_asset(y_tokenizer, 'Y_TOKENIZER')
        self.save_asset(y_train_tokenized, 'Y_TRAIN')
        self.save()

    # ====================
    def Xy_to_output(self, X: list, y: list) -> str:

        X_tokenizer = self.get_asset('X_TOKENIZER')
        X_decoded = X_tokenizer.sequences_to_texts([X])[0].replace(' ', '')
        y_tokenizer = self.get_asset('Y_TOKENIZER')
        y_decoded = self.decode_class_list(y_tokenizer, y)
        output_parts = [self.char_and_class_to_output_str(X_, y_) for X_, y_ in zip(X_decoded, y_decoded)]
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
    def decode_class_list(self, tokenizer, encoded: list) -> list:

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
                this_class = ''.join([this_class] + [f for f in self.feature_chars if f in feature_chars_encountered])
            else:
                raise ValueError('Not implemented yet when one_of_each=False.')
            y.append(this_class)
        # Encode input and output
        assert len(X) == self.seq_length
        assert len(y) == self.seq_length
        return X, y
