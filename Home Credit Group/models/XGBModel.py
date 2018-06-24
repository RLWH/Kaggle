import os
import xgboost as xgb
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix
from models.model import Model


def generate_params_set(num_trials=2):

    trials = []

    # Generate n trials
    for i in range(num_trials):

        XGB_HPARAMS = {
            'max_depth': np.random.randint(1, 20),
            'min_child_weight': np.random.randint(5, 12),
            'gamma': np.power(2, np.random.randint(4)),
            'eta': np.round(np.random.uniform(low=0.5), decimals=2),
            'objective': 'binary:logistic',
            'nthread': 4,
            'eval_metric': 'auc',
            'silent': 1
        }

        trials.append(XGB_HPARAMS)

    return trials


class XGBModel(Model):
    """
    XGBoostModel
    """

    def __init__(self, hparams=None, pretrained_model_path=None, trial=0):
        super().__init__(hparams=hparams, pretrained_model_path=pretrained_model_path, trial=trial)
        if self.pretrained_model_path is not None:
            self.model_path = self.pretrained_model_path
            self.model = xgb.Booster(params=self.hparams, model_file=self.model_path)
        else:
            self.model = None

    def train(self, train_dataset, eval_dataset=None, num_round=None, verbose=False):
        """

        :param train_X:
        :param train_Y:
        :return:
        """

        eval_dict = {}

        if eval_dataset:
            bst = xgb.train(self.hparams, train_dataset, num_boost_round=num_round, verbose_eval=verbose,
                            early_stopping_rounds=20, evals=eval_dataset, evals_result=eval_dict)
        else:
            bst = xgb.train(self.hparams, train_dataset, num_boost_round=num_round, evals_result=eval_dict, verbose_eval=verbose)

        bst.save_model(os.path.join(self.output_path, "xgb_%s.model" % self.trial))

        self.model = bst

        # Output the best store and the respective model
        scores = np.asarray(eval_dict['eval']['auc'])
        best_score = np.max(scores)
        best_model = np.argmax(scores)

        return bst, {'best_score': best_score, 'best_model': best_model}

    def val(self, eval_dataset, y_true, threshold=0.6):

        y_pred = (self.model.predict(eval_dataset) > threshold).astype(int)
        y_true = y_true.as_matrix()

        print(accuracy_score(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))

    def infer(self, test_dataset, index=None, threshold=0.5, output_as='proba', to_csv=False):

        if output_as == 'proba':

            logits = self.model.predict(test_dataset)

            d = {'logits': logits}
            result = pd.DataFrame(d)

            if to_csv:
                result.to_csv('logits.csv')

            return result

        elif output_as == 'pred':

            logits = self.model.predict(test_dataset)
            y_pred = (logits > threshold).astype('int')

            d = {'logits': logits, 'y_pred': y_pred}
            result = pd.DataFrame(d)

            if to_csv:
                result.to_csv('result.csv')
            return result

        else:
            """ Return logits by default """
            print("No selected output type. Return nothing. ")
