def all_chars_in_dataset(dataset: list) -> list:

    all_chars = set()
    for d in dataset:
        chars = set(dataset)
        all_chars = all_chars | chars
    return list(sorted(all_chars))
