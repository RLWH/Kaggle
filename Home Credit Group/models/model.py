"""Abstract Model definition"""

import os

from abc import ABCMeta, abstractclassmethod

OUTPUT_DIR = 'output'

class Model(object):

    __metaclass__ = ABCMeta

    def __init__(self, hparams=None, pretrained_model_path=None, trial=0):

        self.pretrained_model_path = pretrained_model_path

        # Set model output path
        self.output_path = os.path.join(os.getcwd(), OUTPUT_DIR)
        print("Model output path %s" % self.output_path)

        # Set model hparams
        self.hparams = hparams
        self.trial = trial
        print("Trial %s - Model built with hparams %s" % (self.trial, self.hparams))

    @abstractclassmethod
    def train(self, train_dataset, eval_dataset=None, num_round=None, verbose=False):
        print("Start training model")
        raise NotImplementedError()

    @abstractclassmethod
    def val(self, eval_dataset, y_true, transformation=None, threshold=0.5):
        raise NotImplementedError()

    @abstractclassmethod
    def infer(self, test_dataset, threshold=0.5, output_as='logits', to_csv=False):
        raise NotImplementedError()