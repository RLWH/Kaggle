"""
A utility file for word processing functions
"""
import multiprocessing
import itertools
import pandas as pd
import csv

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
        (Parallel(n_jobs=n_jobs)(delayed(word_tokenize)(line) for line in data))))
    all_symbols = special_symbols + list(set_symbols)
    int_to_symbol = {word_i: word for word_i, word in enumerate(all_symbols)}
    symbol_to_int = {word: word_i for word_i, word in int_to_symbol.items()}
    return all_symbols, int_to_symbol, symbol_to_int


def build_vocab(source_path, export_path):
    # Create a vocabulary csv from the source file
    df = pd.read_csv(source_path)
    all_symbols, _, _ = extract_word_vocab(df['comment_text'].values)
    all_symbols = [x.encode('UTF8') for x in all_symbols]

    # print(type(all_symbols))
    print("%s vocabs found" % len(all_symbols))

    with open(export_path, 'w') as myfile:
        wr = csv.writer(myfile)
        for symbol in all_symbols:
            wr.writerow([symbol])


def read_vocab(source_path):
    # Create a vocabulary csv from the source file
    with open(source_path, 'r', newline='') as myfile:
        reader = csv.reader(myfile)

        all_symbols = []
        for symbol in reader:
            all_symbols.extend(symbol)

        print("%s vocabs found" % len(all_symbols))

    return all_symbols


if __name__ == "__main__":
    build_vocab(source_path='../data/train.csv', export_path='../data/vocab.csv')
    all_symbols = read_vocab(source_path='../data/vocab.csv')
    print(len(all_symbols))