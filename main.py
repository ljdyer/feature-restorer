"""
Assumptions:

- Spaces come after other punctuation characters
(i.e. "eats, shoots, and leaves" NOT "eats ,shoots ,and leaves"
"""

CLASS_ATTRS_FNAME = 'class_attrs.pickle'
X_TRAIN_FNAME = 'X_train.pickle'
Y_TRAIN_FNAME = 'y_train.pickle'
X_TOKENIZER_FNAME = 'X_tokenizer.pickle'
INPUT_TOKENIZER_FNAME = 'input_tokenizer.pickle'
OUTPUT_TOKENIZER_FNAME = ''


import pandas as pd
import numpy as np
from tqdm.notebook import tqdm as notebook_tqdm
from tqdm import tqdm as non_notebook_tqdm
import psutil
import pickle
import os
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical


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
class SampleMaker:
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
    def __init__(self, attrs: dict = None, name_to_load: str = None):

        if attrs is not None:
            self.__dict__.update(attrs)
            if self.spaces:
                self.feature_chars = self.other_features + [' ']
            else:
                self.feature_chars = self.other_features
            self.class_attrs_path = os.path.join(self.root_folder, CLASS_ATTRS_FNAME)
            self.X_train_path = os.path.join(self.root_folder, X_TRAIN_FNAME)
            self.y_train_path = os.path.join(self.root_folder, Y_TRAIN_FNAME)
            self.X_tokenizer_path = os.path.join(self.root_folder, X_TOKENIZER_FNAME)
            if not os.path.exists(self.root_folder):
                os.makedirs(self.root_folder)
            self.save()
        elif name_to_load is not None:
            class_attrs_path = os.path.join(name_to_load, CLASS_ATTRS_FNAME)
            attrs = load_pickle(class_attrs_path)
            self.__dict__.update(attrs)
        else:
            raise ValueError('You must specify either attrs or load_path.')

    # ====================
    def save(self):

        class_attrs = self.__dict__.copy()
        save_pickle(class_attrs, self.class_attrs_path)

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
        X_train_tokenized = X_tokenizer.fit_on_texts(X)
        save_pickle(X_tokenizer, self.X_tokenizer_path)
        save_pickle(X_train_tokenized, self.X_train_path)
        save_pickle(y_train, self.y_train_path)
        self.save()

    # ====================
    def sample_to_output(self, sample: np.ndarray) -> str:

        char_encoding = sample[0]
        class_list = sample[1]
        output_list = []
        for idx, (char_encoded, class_encoded) in enumerate(list(zip(char_encoding, class_list))):
            char = self.int_to_char[char_encoded]
            class_ = self.int_to_class[class_encoded]
            if self.one_of_each:
                if class_ % 2 == 0:
                    char = char.upper()
                for prime in sorted(self.prime_to_feature_char.keys()):
                    if class_ % prime == 0:
                        char = char + self.prime_to_feature_char[prime]
                output_list.append(char)
            else:
                raise ValueError('Not implemented yet when one_of_each=False.')
        return ''.join(output_list)

    # # ====================
    # def get_train_sample(self, n: int):

    #     train_samples = np.load(self.train_samples_path)
    #     x = train_samples[n]
    #     del train_samples
    #     return x    

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

    # ====================
    def encode_classes(self, classes: list):

        # Add any new classes to the list of classes
        if 'output_classes' not in self.__dict__:
            self.output_classes = []
        new_classes = set(list([c for c in classes if c not in self.output_classes]))
        self.output_classes.extend(new_classes)
        # Regenerate encoder dictionaries
        self.class_to_int = {c: i for c, i in zip(self.output_classes, range(len(self.output_classes)))}
        self.int_to_class = {i: c for c, i in self.class_to_int.items()}
        # Encode
        return [self.class_to_int[c] for c in classes]

# ====================
if __name__ == "__main__":

    data_path = 'TED_TRAIN.csv'
    data = pd.read_csv(data_path)['all_cleaned'].to_list()
    attrs = {
        'name': 'TED_TALKS_2',
        'capitalisation': True,
        'spaces': True,
        'other_features': list('.,'),
        'seq_length': 200,
        'one_of_each': True
    }
    sample_maker = SampleMaker(attrs)
    sample_maker.generate_char_maps(data)
    sample_maker.load_train_data(data)
    print(sample_maker.sample_to_output(sample_maker.get_train_sample(3)))