import os
import numpy as np
import pandas as pd

from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import  Ridge

from models.model import Model


def generate_params_set(num_trials=1):

    trials = []

    # Generate n trials
    for i in range(num_trials):

        HPARAMS = {
            'fit_intercept': True,
            'normalize': False,
            'alpha': np.random.uniform(0, 5),
            'tol': np.random.uniform(0, 1)
        }

        trials.append(HPARAMS)

    return trials


class RidgeLRModel(Model):
    """
    Linear Regression Model
    """

    def __init__(self, hparams=None, pretrained_model_path=None, trial=0):
        super().__init__(hparams=hparams, pretrained_model_path=pretrained_model_path, trial=trial)
        if self.pretrained_model_path is not None:
            self.model_path = self.pretrained_model_path
        else:
            self.model = None

    def train(self, train_dataset, eval_dataset=None, num_round=None, verbose=False):

        features, target = train_dataset

        model = Ridge(**self.hparams)
        model.fit(X=features, y=target)

        self.model = model

        joblib.dump(model, os.path.join(self.output_path, "Ridge_%s.pkl" % self.trial))

        return model

    def val(self, eval_dataset, y_true, transformation=None, threshold=0.6):

        features, y_true = eval_dataset

        if self.model is not None:
            y_pred = self.model.predict(features)

            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)

            r2 = r2_score(y_true, y_pred)

            print("RMSE: %s" % rmse)
            print("R2 score %s" % r2)

            if transformation is not None:

                # Transformation of y
                y_true_transformed = np.expm1(y_true)
                y_pred_transformed = np.expm1(y_pred)

                transformed_mse = mean_squared_error(y_true_transformed, y_pred_transformed)
                transformed_rmse = np.sqrt(transformed_mse)

                print("Transformed RMSE: %s" % transformed_rmse)

                return {'RMSE': rmse, 'transformed_rmse': transformed_rmse, 'r2': r2}

            return {'RMSE': rmse, 'r2': r2}

    def infer(self, test_dataset, index=None, output_as='number', to_csv=False):
        pass
