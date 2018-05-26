import torch


def save_checkpoint():
    """
    A function that help to save and load the model during training (i.e. checkpoint-ing the model)
    This implementation is heavily inspired by
    https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/util/checkpoint.py

    To make a checkpoint, call this function and pass in the model, optimizer, epoch, step

    :return:
    """