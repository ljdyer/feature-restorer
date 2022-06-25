"""
Assumptions:

- Spaces come after other punctuation characters
(i.e. "eats, shoots, and leaves" NOT "eats ,shoots ,and leaves"
"""


# TODO: save samples as numpy array
# TODO: use memmap?


import re
import sympy
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm as notebook_tqdm
from tqdm import tqdm as non_notebook_tqdm
import psutil
import pickle

# ====================
def is_running_from_ipython():
    try:
        from IPython import get_ipython
        return True
    except ModuleNotFoundError:
        return False
    

if is_running_from_ipython():
    my_tqdm = notebook_tqdm
else:
    my_tqdm = non_notebook_tqdm



class SampleMaker:

    # ====================
    def __init__(self,
                 data: list,
                 capitalisation: bool,
                 spaces: bool,
                 other_features: list,
                 seq_length: int,
                 pickle_path: str,
                 train_sample_npy_path: str):

        self.data = data
        self.capitalisation = capitalisation
        self.spaces = spaces
        self.other_features = other_features
        self.seq_length = seq_length
        self.pickle_path = pickle_path
        self.train_sample_npy_path = train_sample_npy_path
        if self.spaces:
            self.feature_chars = other_features + [' ']
        else:
            self.feature_chars = other_features
        self.feature_char_to_prime = {char: sympy.prime(idx+2) for idx, char in enumerate(self.feature_chars)}
        self.prime_to_feature_char = {p: c for c, p in self.feature_char_to_prime.items()}
        self.output_classes = set()
        self.generate_char_maps()
        self.save()
        self.generate_samples()

    # ====================
    def generate_char_maps(self):

        all_chars = set()
        for d in self.data:
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

    # ====================
    def generate_samples(self):

        all_samples = []
        pbar = my_tqdm(range(len(data)))
        for i in pbar:
            pbar.set_postfix({'ram_usage': psutil.virtual_memory().percent})
            samples = self.datapoint_to_samples(i)
            if samples:
                all_samples.extend(samples)
        all_samples_array = np.array(all_samples)
        np.save(self.train_sample_npy_path, all_samples_array)
        self.save()

    # ====================
    def save(self):

        member_variables = self.__dict__.copy()
        member_variables['data'] = None
        with open(self.pickle_path, 'wb') as f:
            pickle.dump(member_variables, f)


    
    # ====================
    def datapoint_to_samples(self, datapoint_index: int) -> list:

        datapoint = self.data[datapoint_index]
        if self.spaces:
            words = datapoint.split()
            substrs = [' '.join(words[i:]) for i in range(len(words))]
            samples = [self.substr_to_sample(substr) for substr in substrs]
            samples = [s for s in samples if s is not None]
        else:
            raise ValueError('Not implemented yet when spaces=False')

    # ====================
    def substr_to_sample(self, substr: str) -> tuple:

        output_chars = []
        class_list = []
        chars = list(substr)
        if chars[0] in self.feature_chars:
            # Substring can't begin with a feature char
            return None
        for _ in range(self.seq_length):
            try:
                this_char = chars.pop(0)
            except IndexError:
                # Substr not long enough
                return None
            if self.capitalisation:
                output_chars.append(self.get_int_from_char(this_char.lower()))
                class_list.append(1 if this_char.islower() else 2)
            else:
                output_chars.append(self.get_int_from_char(this_char))
                class_list.append(1)
            while chars and chars[0] in self.feature_chars:
                this_feature_char = chars.pop(0)
                class_list[-1] *= self.feature_char_to_prime[this_feature_char]
        assert len(output_chars) == self.seq_length
        assert len(class_list) == self.seq_length
        self.add_output_classes(set(class_list))
        return (np.array([output_chars, class_list]))

    # ====================
    def get_int_from_char(self, char: str) -> int:

        if char in self.char_to_int:
            return self.char_to_int[char]
        else:
            return self.char_to_int['<UNK>']

    # ====================
    def add_output_classes(self, new_classes: set):

        self.output_classes = self.output_classes | new_classes


# ====================
if __name__ == "__main__":

    data_path = 'TED_TRAIN.csv'
    data = pd.read_csv(data_path)['all_cleaned'].to_list()
    sample_maker = SampleMaker(data,
                               capitalisation=True,
                               spaces=True, 
                               other_features=list('.,'),
                               seq_length=100,
                               pickle_path='sample_maker.pickle',
                               train_sample_npy_path='train_samples.npy')
    print(sample_maker.datapoint_to_samples(0))