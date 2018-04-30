"""
A utility file for word processing functions
"""
import multiprocessing
import itertools

from joblib import Parallel, delayed
from nltk.tokenize import word_tokenize


def extract_character_vocab(data):
    """
    A function to extract vocabs in character level
    :param data:
    :return:
    """
    special_symbols = ['<PAD>', '<UNK>', '<GO>', '<EOS>']
    set_symbols = set([character for line in data for character in line])
    all_symbols = special_symbols + list(set_symbols)
    int_to_symbol = {word_i: word for word_i, word in enumerate(all_symbols)}
    symbol_to_int = {word: word_i for word_i, word in int_to_symbol.items()}
    return int_to_symbol, symbol_to_int


def extract_word_vocab(data, n_jobs=4):
    """
    A function to extract vocabs in word level
    :param data:
    :return:
    """
    special_symbols = ['<PAD>', '<UNK>', '<GO>', '<EOS>']

    set_symbols = set(itertools.chain.from_iterable(
        (Parallel(n_jobs=4)(delayed(word_tokenize)(line) for line in data))))
    all_symbols = special_symbols + list(set_symbols)
    int_to_symbol = {word_i: word for word_i, word in enumerate(all_symbols)}
    symbol_to_int = {word: word_i for word_i, word in int_to_symbol.items()}
    return all_symbols, int_to_symbol, symbol_to_int
