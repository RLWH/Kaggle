import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

from functools import reduce;
from abc import ABCMeta, abstractclassmethod
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# TRAIN_FEATURE_COLUMNS = [
#     'SK_ID_CURR',
#     'NAME_CONTRACT_TYPE',
#     'CODE_GENDER',
#     'CNT_CHILDREN',
#     'NAME_INCOME_TYPE',
#     'NAME_EDUCATION_TYPE',
#     'NAME_HOUSING_TYPE',
#     'DAYS_EMPLOYED',
#     'DAYS_REGISTRATION',
#     'DAYS_ID_PUBLISH',
#     'OWN_CAR_AGE',
#     'FLAG_MOBIL',
#     'FLAG_EMP_PHONE',
#     'FLAG_WORK_PHONE',
#     'OCCUPATION_TYPE',
#     'CNT_FAM_MEMBERS',
#     'REGION_RATING_CLIENT',
#     'ORGANIZATION_TYPE',
#     'EXT_SOURCE_1',
#     'EXT_SOURCE_2',
#     'EXT_SOURCE_3',
#     'APARTMENTS_AVG',
#     'DAYS_LAST_PHONE_CHANGE',
#     'FLAG_DOCUMENT_2',
#     'FLAG_DOCUMENT_6',
#     'FLAG_DOCUMENT_7',
#     'FLAG_DOCUMENT_14',
#     'FLAG_DOCUMENT_15',
#     'FLAG_DOCUMENT_18',
#     'FLAG_DOCUMENT_21',
#     'AMT_REQ_CREDIT_BUREAU_YEAR'
# ]
#
# BUREAU_FEATURE_COLUMNS = [
#     'DAYS_ENDDATE_FACT',
#     'AMT_CREDIT_MAX_OVERDUE'
# ]

FEATURE_FILE_LIST = [
    {'file': 'data/application_train.csv',
     'target': 'TARGET',
     'features': [
         {'column_name': 'SK_ID_CURR', 'dtype': 'int'},
         {'column_name': 'NAME_CONTRACT_TYPE', 'dtype': 'category'},
         {'column_name': 'CODE_GENDER', 'dtype': 'category'},
         {'column_name': 'CNT_CHILDREN', 'dtype': 'int'},
         {'column_name': 'NAME_INCOME_TYPE', 'dtype': 'category'},
         {'column_name': 'NAME_EDUCATION_TYPE', 'dtype': 'category'},
         {'column_name': 'NAME_HOUSING_TYPE', 'dtype': 'category'},
         {'column_name': 'DAYS_EMPLOYED', 'dtype': 'int'},
         {'column_name': 'DAYS_REGISTRATION', 'dtype': 'int'},
         {'column_name': 'DAYS_ID_PUBLISH', 'dtype': 'int'},
         {'column_name': 'OWN_CAR_AGE', 'dtype': 'int'},
         {'column_name': 'FLAG_MOBIL', 'dtype': 'category'},
         {'column_name': 'FLAG_EMP_PHONE', 'dtype': 'category'},
         {'column_name': 'FLAG_WORK_PHONE', 'dtype': 'category'},
         {'column_name': 'OCCUPATION_TYPE', 'dtype': 'category'},
         {'column_name': 'CNT_FAM_MEMBERS', 'dtype': 'int'},
         {'column_name': 'REGION_RATING_CLIENT', 'dtype': 'int'},
         {'column_name': 'ORGANIZATION_TYPE', 'dtype': 'category'},
         {'column_name': 'EXT_SOURCE_1', 'dtype': 'float'},
         {'column_name': 'EXT_SOURCE_2', 'dtype': 'float'},
         {'column_name': 'EXT_SOURCE_3', 'dtype': None},
         {'column_name': 'APARTMENTS_AVG', 'dtype': None},
         {'column_name': 'DAYS_LAST_PHONE_CHANGE', 'dtype': None},
         {'column_name': 'FLAG_DOCUMENT_2', 'dtype': 'category'},
         {'column_name': 'FLAG_DOCUMENT_6', 'dtype': 'category'},
         {'column_name': 'FLAG_DOCUMENT_7', 'dtype': 'category'},
         {'column_name': 'FLAG_DOCUMENT_14', 'dtype': 'category'},
         {'column_name': 'FLAG_DOCUMENT_15', 'dtype': 'category'},
         {'column_name': 'FLAG_DOCUMENT_18', 'dtype': 'category'},
         {'column_name': 'FLAG_DOCUMENT_21', 'dtype': 'category'},
         {'column_name': 'AMT_REQ_CREDIT_BUREAU_YEAR', 'dtype': 'category'}],
     'transformation': [
         {'column_name': 'DAYS_EMPLOYED',
          'action': 'replace',
          'parameters': {'to_replace': 365243},
          'assign': 'DAYS_EMPLOYED'},
         {'column_name': 'DAYS_REGISTRATION',
          'action': 'apply',
          'parameters': {'func': eval('lambda x: np.log1p(np.abs(x))')},
          'assign': 'LOG_DAYS_REGISTRATION'},
         {'column_name': 'DAYS_ID_PUBLISH',
          'action': 'apply',
          'parameters': {'func': eval('lambda x: np.log1p(np.abs(x))')},
          'assign': 'LOG_DAYS_ID_PUBLISH'}],
     'aggregation': None
     },

    {'file': 'data/bureau.csv',
     'target': None,
     'features': [
         {'column_name': 'SK_ID_CURR', 'dtype': 'int'},
         {'column_name': 'DAYS_ENDDATE_FACT', 'dtype': 'int'},
         {'column_name': 'AMT_CREDIT_MAX_OVERDUE', 'dtype': 'int'}],
     'transformation': None,
     'aggregation': 
     }

]

DTYPES_TRANSFORM_COLS = {
    'SK_ID_CURR': 'int',
    'NAME_CONTRACT_TYPE': 'category',
    'CODE_GENDER': 'category',
    'NAME_INCOME_TYPE': 'category',
    'NAME_EDUCATION_TYPE': 'category',
    'NAME_HOUSING_TYPE': 'category',
    'FLAG_MOBIL': 'category',
    'FLAG_EMP_PHONE': 'category',
    'FLAG_WORK_PHONE': 'category',
    'OCCUPATION_TYPE': 'category',
    'REGION_RATING_CLIENT': 'category',
    'ORGANIZATION_TYPE': 'category',
    'FLAG_DOCUMENT_2': 'category',
    'FLAG_DOCUMENT_6': 'category',
    'FLAG_DOCUMENT_7': 'category',
    'FLAG_DOCUMENT_14': 'category',
    'FLAG_DOCUMENT_15': 'category',
    'FLAG_DOCUMENT_18': 'category',
    'FLAG_DOCUMENT_21': 'category',
    'AMT_REQ_CREDIT_BUREAU_YEAR': 'category'
}

SERIES_TRANSFORMATION_COLS = {
    'DAYS_EMPLOYED': {'action': 'replace', 'parameters': {'to_replace': 365243}, 'assign': 'DAYS_EMPLOYED'},
    'DAYS_REGISTRATION': {'action': 'apply',
                          'parameters': {'func': eval('lambda x: np.log1p(np.abs(x))')},
                          'assign': 'LOG_DAYS_REGISTRATION'},
    'DAYS_ID_PUBLISH': {'action': 'apply',
                              'parameters': {'func': eval('lambda x: np.log1p(np.abs(x))')},
                              'assign': 'LOG_DAYS_ID_PUBLISH'},
}

DROP_COLS_BEFORE_TRAINING = ['SK_ID_CURR', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH']

LABEL_COLUMN = 'TARGET'


class Model(object):

    __metaclass__ = ABCMeta

    def __init__(self, hparams, pretrained_model=None, trial=0):
        self.hparams = hparams
        self.trial = trial
        print("Model built with hparams %s" % self.hparams)
        if pretrained_model:
            self.model = pretrained_model
        else:
            self.model = None

    @abstractclassmethod
    def train(self, dtrain, evallist=None, num_round=None, verbose=False):
        print("Start training model")
        raise NotImplementedError()

    @abstractclassmethod
    def val(self, dtest, y_true):
        raise NotImplementedError()

    @abstractclassmethod
    def infer(self, model, export=False):
        raise NotImplementedError()


class XGBModel(Model):
    """
    XGBoostModel
    """

    def train(self, dtrain, evallist=None, num_round=None, verbose=False):
        """

        :param train_X:
        :param train_Y:
        :return:
        """

        if evallist:
            bst = xgb.train(self.hparams, dtrain, num_boost_round=num_round, verbose_eval=verbose,
                            early_stopping_rounds=10, evals=evallist)
        else:
            bst = xgb.train(self.hparams, dtrain, num_boost_round=num_round, verbose_eval=verbose)

        bst.save_model("xgb.model_%s" % self.trial)

        self.model = bst

        return bst

    def val(self, dtest, y_true, threshold=0.5):

        y_pred = (self.model.predict(dtest, ntree_limit=self.model.best_iteration) > threshold).astype(int)
        y_true = y_true.as_matrix()

        print(accuracy_score(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))


def load_data(feature_dict, set_index=None, how_join='left', verbose=False):
    """

    :param feature_dict: List of Dictionary of features.
                         In the format of [{'filename': {'features': [], 'target': []}]
    :param set_index:
    :param verbose:
    :return:
    """

    # Test the feature dict
    if verbose:
        print("Length of feature dictionary: %s" % len(feature_dict))

    features_list = []
    target = None

    for i, file in enumerate(feature_dict):

        if verbose:
            print("The filename of the first file: %s" % file['file'])
            print("The features to be extracted of the first file: %s" % file['features'])
            print("The target to be extracted of the first file: %s" % file['target'])

        if file['target'] is not None:

            if verbose:
                print("Extract with target")

            file_features, target = load_individual_file(file['file'],
                                                         feature_columns=file['features'],
                                                         label_column=file['target'])
        else:
            if verbose:
                print("Extract without target")
            file_features = load_individual_file(file['file'],
                                                 feature_columns=file['features'])

        file_features = file_features.set_index(set_index)
        features_list.append(file_features)



    return features_list, target

    # if len(features_list) > 1:
    #     features = reduce(lambda left, right: pd.merge(left, right, how=how_join), features_list)
    # else:
    #     features = features_list[0]
    #
    # print(features)


    # if label_column:
    #     target = dataframe[label_column]
    #     if label_column:
    #         print(target.value_counts())
    #     return features, target
    # else:
    #     return features


def load_individual_file(file_path, feature_columns, label_column=None):
    dataframe = pd.read_csv(file_path)
    features = dataframe.loc[:, feature_columns]

    if label_column:
        target = dataframe[label_column]
        return features, target
    else:
        return features


def split_data(features, target=None, test_size=0.1, random_state=42, verbose=False):

    if verbose:
        print("Splitting data...")

    # Split the dataframe to features and target
    X_train, X_val, y_train, y_val = train_test_split(features,
                                                      target,
                                                      test_size=test_size,
                                                      random_state=random_state,
                                                      stratify=target)

    return X_train, X_val, y_train, y_val


def generate_params_set(model='xgb', num_trials=10):

    if model == 'xgb':

        trials = []

        # Generate n trials
        for i in range(num_trials):

            XGB_HPARAMS = {
                'max_depth': np.power(2, np.random.randint(8)),
                'min_child_weight': np.random.randint(15),
                'gamma': np.power(2, np.random.randint(4)),
                'eta': np.round(np.random.uniform(low=0.5), decimals=2),
                'objective': 'binary:logistic',
                'nthread': 4,
                'eval_metric': 'auc',
                'silent': 1
            }

            trials.append(XGB_HPARAMS)

        return trials


def transform(dataframe, verbose=False):
    """
    Return a transformed dataframe
    :param dataframe:
    :return:
    """

    # 1. Coerce the dataset into the correct type
    for _, (field_name, target_type) in enumerate(DTYPES_TRANSFORM_COLS.items()):
        if verbose:
            print("Coercing dtype of %s to %s" % (field_name, target_type))
        dataframe[field_name] = dataframe[field_name].astype(target_type)

        if target_type == "category":
            dataframe[field_name] = dataframe[field_name].cat.codes

    # 2. ADHOC TRANSFORMATION
    for _, (field_name, action) in enumerate(SERIES_TRANSFORMATION_COLS.items()):
        print("Transforming %s with function %s and parameters %s" %
              (field_name, action['action'], action['parameters']))
        series = getattr(dataframe[field_name], action['action'])(**action['parameters'])
        dataframe = dataframe.assign(**{action['assign']: series})

    # 3. Drop unused columns
    dataframe = dataframe.drop(columns=DROP_COLS_BEFORE_TRAINING, axis=1)

    return dataframe


def train():
    # Load the file
    features_list, target = load_data(FEATURE_FILE_LIST, set_index='SK_ID_CURR')

    print(features_list[0])
    print(features_list[1])

    # transformed_features = transform(features, verbose=True)
    # print(transformed_features.info(verbose=True))
    #
    # # Split dataset
    # X_train, X_val, y_train, y_val = split_data(features, target)
    #
    # # Transform the dataset for xgb
    # dtrain = xgb.DMatrix(X_train, label=y_train)
    # deval = xgb.DMatrix(X_val, label=y_val)
    #
    # evallist = [(deval, 'eval'), (dtrain, 'train')]
    #
    # # Generate hyperparameters
    # hparams_set = generate_params_set('xgb', num_trials=10)
    #
    # best_scores = []
    # best_round = []
    #
    # # Build model for each hparams set
    # for i, hparam in enumerate(hparams_set):
    #
    #     model = XGBModel(hparam, trial=i)
    #     bst = model.train(dtrain=dtrain, evallist=evallist, num_round=30, verbose=True)
    #
    #     best_scores.append(bst.best_score)
    #     best_round.append(bst.best_iteration)
    #
    #     model.val(dtest=deval, y_true=y_val)
    #
    # # bst = model.train(dtrain, evallist, num_round=10)
    # #
    # # # Plot the analysis
    # # xgb.plot_importance(bst)
    # # plt.show()
    # #
    # # # Plot the confusion matrix
    #
    # print(best_scores)
    # print(best_round)


if __name__ == "__main__":
    train()