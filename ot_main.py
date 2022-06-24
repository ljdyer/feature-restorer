import string

SEQ_LENGTH = 200
WORD_SHIFT = 1
END_BUFFER = 100

vocab_chars = string.ascii_lowercase + '*'
NUM_VOCAB_CHARS = len(vocab_chars)
char_to_int = {c: i for c, i in zip(vocab_chars, range(len(vocab_chars)))}
int_to_char = {i: c for c, i in zip(vocab_chars, range(len(vocab_chars)))}

def assign_prime_labels(chars: list) -> dict:

    char_to_label = {char: sympy.prime(idx+1)
                     for idx, char in enumerate(chars)}
    return char_to_label


char_to_label = assign_prime_labels(CHARS_TO_RESTORE)
print(char_to_label)

def get_input_and_labels(output_str: str) -> list:
    """Convert an output (Gold standard) string to a list of tuples
    (input_char, class)"""

    input_chars = []
    class_list = []

    for index, c in enumerate(list(output_str)):
        if c in char_to_label:
            if len(class_list) > 1:
                class_list[-1] *= char_to_label[c]
        else:
            input_chars.append(c)
            class_list.append(1)
    
    assert len(input_chars) == len(class_list)
    return list(zip(input_chars, class_list))