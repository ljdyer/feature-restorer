"""
Assumptions:

- Spaces come after other punctuation characters
(i.e. "eats, shoots, and leaves" NOT "eats ,shoots ,and leaves"
"""


# TODO: save samples as numpy array
# TODO: use memmap?

CLASS_ATTRS_FNAME = 'class_attrs.pickle'
TRAIN_SAMPLES_FNAME = 'train_samples.npy'


import sympy
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm as notebook_tqdm
from tqdm import tqdm as non_notebook_tqdm
import psutil
import pickle
import os

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
class SampleMaker:
    """
    Required attributes for new instance:

    name: str,
    capitalisation: bool,
    spaces: bool,
    other_features: list,
    seq_length: int,
    one_of_each: bool = True
    """

    # ====================
    def __init__(self, attrs: dict = None, name_to_load: str = None):

        if attrs is not None:
            self.__dict__.update(attrs)
            if self.spaces:
                self.feature_chars = self.other_features + [' ']
            else:
                self.feature_chars = self.other_features
            self.feature_char_to_prime = {char: sympy.prime(idx+2) for idx, char in enumerate(self.feature_chars)}
            self.prime_to_feature_char = {p: c for c, p in self.feature_char_to_prime.items()}
            self.class_attrs_path = os.path.join(self.name, CLASS_ATTRS_FNAME)
            self.train_samples_path = os.path.join(self.name, TRAIN_SAMPLES_FNAME)
            if not os.path.exists(self.name):
                os.makedirs(self.name)
            self.save()
        elif name_to_load is not None:
            class_attrs_path = os.path.join(name_to_load, CLASS_ATTRS_FNAME)
            with open(class_attrs_path, 'rb') as f:
                attrs = pickle.load(f)
            self.__dict__.update(attrs)
        else:
            raise ValueError('You must specify either attrs or load_path.')

    # ====================
    def save(self):

        class_attrs = self.__dict__.copy()
        with open(self.class_attrs_path, 'wb') as f:
            pickle.dump(class_attrs, f)

    # ====================
    def generate_char_maps(self, data):

        all_chars = set()
        for d in data:
            chars = set(d)
            all_chars = all_chars | chars
        if self.capitalisation:
            all_chars = set([c.lower() for c in list(all_chars)])
        if self.spaces:
            all_chars = all_chars - {' '}
        if self.other_features:
            all_chars = all_chars - set(self.other_features)
        all_chars = list(sorted(all_chars))
        all_chars.append('<UNK>')
        self.char_to_int = {c: i for c, i in zip(all_chars, range(len(all_chars)))}
        self.int_to_char = {i: c for c, i in self.char_to_int.items()}
        self.save()

    # ====================
    def generate_train_samples(self, data: list):

        all_samples = []
        pbar = my_tqdm(range(len(data)))
        for i in pbar:
            pbar.set_postfix({
                'ram_usage': psutil.virtual_memory().percent,
                'num_samples': len(all_samples),
                'estimated_num_samples': len(all_samples) * (len(pbar) / (pbar.n + 1))
            })
            samples = self.datapoint_to_samples(data[i])
            if samples:
                all_samples.extend(samples)
        all_samples_array = np.array(all_samples)
        np.save(self.train_samples_path, all_samples_array)
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

    # ====================
    def get_train_sample(self, n: int):

        train_samples = np.load(self.train_samples_path)
        x = train_samples[n]
        del train_samples
        return x    

    # ====================
    def datapoint_to_samples(self, datapoint: str) -> list:

        if self.spaces:
            words = datapoint.split()
            substrs = [' '.join(words[i:]) for i in range(len(words))]
            samples = [self.substr_to_sample(substr) for substr in substrs]
            samples = [s for s in samples if s is not None]
        else:
            raise ValueError('Not implemented yet when spaces=False.')
        return samples

    # ====================
    def substr_to_sample(self, substr: str) -> tuple:

        output_chars = []
        class_list = []
        chars = list(substr)
        if chars[0] in self.feature_chars:
            # Substring can't begin with a feature char
            return None
        for _ in range(self.seq_length):
            this_class = [1]
            # Get the next letter
            try:
                this_char = chars.pop(0)
            except IndexError:
                # Substr not long enough
                return None
            # Check for capitalisation
            if self.capitalisation and this_char.isupper():
                output_chars.append(this_char.lower())
                this_class.append(2)
            else:
                output_chars.append(this_char)
            # Check for other features
            while chars and chars[0] in self.feature_chars:
                this_feature_char = chars.pop(0)
                this_class.append(self.feature_char_to_prime[this_feature_char])
            if self.one_of_each:
                this_class = list(set(this_class))
            class_list.append(np.product(this_class))
        # Encode input and output
        output_ints = self.encode_chars(output_chars)
        output_classes = self.encode_classes(class_list)
        assert len(output_ints) == self.seq_length
        assert len(output_classes) == self.seq_length
        return (np.array([output_ints, output_classes]))

    # ====================
    def get_int_from_char(self, char: str) -> int:

        if char in self.char_to_int:
            return self.char_to_int[char]
        else:
            return self.char_to_int['<UNK>']

    # ====================
    def encode_chars(self, chars: list) -> list:

        return [self.char_to_int[c] for c in chars]

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
    sample_maker.generate_train_samples(data)
    print(sample_maker.sample_to_output(sample_maker.get_train_sample(3)))