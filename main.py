"""
Assumptions:

- Spaces come after other punctuation characters
(i.e. "eats, shoots, and leaves" NOT "eats ,shoots ,and leaves"
"""


# TODO: save samples as numpy array
# TODO: use memmap?


import re
import sympy

class SampleMaker:

    def __init__(self,
                 data: list,
                 capitalisation: bool,
                 spaces: bool,
                 other_features: list,
                 seq_length: int):

        self.data = data
        self.capitalisation = capitalisation
        self.spaces = spaces
        self.other_features = other_features
        self.seq_length = seq_length
        if self.spaces:
            self.feature_chars = other_features + [' ']
        else:
            self.feature_chars = other_features
        self.feature_char_to_prime = {char: sympy.prime(idx+2) for idx, char in enumerate(self.feature_chars)}
        self.prime_to_feature_char = {p: c for c, p in self.feature_char_to_prime.items()}
        self.generate_char_maps()

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

    def datapoint_to_samples(self, datapoint: str) -> list:

        print(datapoint)
        return datapoint
        # if self.spaces:
        #     print(datapoint)
        #     print(type(datapoint))
        #     words = datapoint.split()
        #     try:
        #         substrs = []
        #     except:
        #         print('Huh?')
        #         quit()
        #     for i in range(len(words)):
        #         try:
        #             substrs.append(' '.join(words[i:]))
        #         except:
        #             print(words[i])
        #             quit()
        #     # substrs = [ for i in range(len(words))]
        #     substrs = [s for s in substrs
        #                if len(re.sub(rf"[{''.join(self.feature_chars)}]", '', s)) > self.seq_length]
        #     samples = []
        #     for substr in substrs:
        #         this_sample = self.substr_to_sample(substr)
        #         if this_sample:
        #             samples.append(this_sample)
        #     return samples
        # else:
        #     raise ValueError('Not implemented yet when spaces=False')

    def substr_to_sample(self, substr: str) -> tuple:

        output_chars = []
        class_list = []
        chars = list(substr)
        # Substring can't begin with a feature char
        if chars[0] in self.feature_chars:
            return None
        for _ in self.seq_length:
            this_char = chars.pop(0)
            if self.capitalisation:
                output_chars.append(self.get_int_from_char(this_char.lower()))
                class_list.append(1 if this_char.islower() else 2)
            else:
                output_chars.append(self.get_int_from_char(this_char))
                class_list.append(1)
            while chars[0] in self.feature_chars:
                this_feature_char = chars.pop(0)
                class_list[-1] *= self.feature_char_to_prime[this_feature_char]
        assert len(output_chars) == self.seq_length
        assert len(class_list) == self.seq_length
        return (output_chars, class_list)

    def get_int_from_char(self, char: str) -> int:

        if char in self.char_to_int:
            return self.char_to_int[char]
        else:
            return self.char_to_int['<UNK>']