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
         {'column_name': 'OWN_CAR_AGE', 'dtype': None},
         {'column_name': 'FLAG_MOBIL', 'dtype': 'category'},
         {'column_name': 'FLAG_EMP_PHONE', 'dtype': 'category'},
         {'column_name': 'FLAG_WORK_PHONE', 'dtype': 'category'},
         {'column_name': 'OCCUPATION_TYPE', 'dtype': 'category'},
         {'column_name': 'CNT_FAM_MEMBERS', 'dtype': None},
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
         {'type': 'series',
          'column_name': 'DAYS_EMPLOYED',
          'action': 'replace',
          'parameters': {'to_replace': 365243},
          'assign': 'DAYS_EMPLOYED'},
         {'type': 'series',
          'column_name': 'DAYS_REGISTRATION',
          'action': 'apply',
          'parameters': {'func': eval('lambda x: np.log1p(np.abs(x))')},
          'assign': 'LOG_DAYS_REGISTRATION'},
         {'type': 'series',
          'column_name': 'DAYS_ID_PUBLISH',
          'action': 'apply',
          'parameters': {'func': eval('lambda x: np.log1p(np.abs(x))')},
          'assign': 'LOG_DAYS_ID_PUBLISH'}],
     'aggregation': None
     },

    {'file': 'data/bureau.csv',
     'target': None,
     'features': [
         {'column_name': 'SK_ID_CURR', 'dtype': 'int'},
         {'column_name': 'DAYS_ENDDATE_FACT', 'dtype': None},
         {'column_name': 'AMT_CREDIT_SUM_DEBT', 'dtype': None},
         {'column_name': 'AMT_CREDIT_SUM', 'dtype': None},
         {'column_name': 'AMT_CREDIT_MAX_OVERDUE', 'dtype': None}],
     'transformation': [
         {'type': 'cross-series',
          'column_name': 'AMT_CREDIT_SUM_DEBT',
          'action': 'div',
          'other': 'AMT_CREDIT_SUM',
          'parameters': {'fill_value': 0},
          'assign': 'DEBT_TO_CREDIT'}
     ],
     'aggregation': {
          'groupby': ['SK_ID_CURR'],
          'agg_params': {
               'CREDIT_DAY_OVERDUE': 'mean',
               'SK_ID_CURR': 'count',
               'AMT_CREDIT_MAX_OVERDUE': 'mean',
               'CNT_CREDIT_PROLONG': 'mean',
               'AMT_CREDIT_SUM': 'mean',
               'DEBT_TO_CREDIT': 'mean'},
          'rename': {
              'CREDIT_DAY_OVERDUE': 'AVG_CREDIT_DAY_OVERDUE',
              'SK_ID_CURR': 'CB_RECORD_COUNT',
              'AMT_CREDIT_MAX_OVERDUE': 'AVG_AMT_CREDIT_MAX_OVERDUE',
              'CNT_CREDIT_PROLONG': 'AVG_CNT_CREDIT_PROLONG',
              'AMT_CREDIT_SUM': 'AVG_AMT_CREDIT_SUM',
              'DEBT_TO_CREDIT': 'AVG_DEBT_TO_CREDIT'},
          'post_transformation': [
              {'type': 'series',
               'column_name': 'AVG_AMT_CREDIT_MAX_OVERDUE',
               'action': 'fillna',
               'parameters': {'value': 0}}
          ]}
     }

]

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

        features_column_names = [feature_column['column_name'] for feature_column in file['features']]

        if file['target'] is not None:

            if verbose:
                print("Extract with target")
            file_features, target = load_individual_file(file['file'],
                                                         feature_columns=features_column_names,
                                                         label_column=file['target'])
        else:
            if verbose:
                print("Extract without target")
            file_features = load_individual_file(file['file'],
                                                 feature_columns=features_column_names)

        # Transform the dataset
        transformed_features = transform(file_features, feature_dict[i], verbose=True)

        transformed_features = transformed_features.set_index(set_index)
        features_list.append(transformed_features)

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

def transform(dataframe, feature_param, verbose=False):
    """

    :param dataframe:
    :param transform_params: Dictionary. Contains, feature dtypes, transformation and aggregation.
    :param verbose:
    :return:
    """

    # 1. Coerce the dataset into the correct type
    for _, x in enumerate(feature_param['features']):

        target_type = x['dtype']

        if target_type is not None:

            if verbose:
                print("Coercing dtype of %s to %s" % (x['column_name'], target_type))

            dataframe[x['column_name']] = dataframe[x['column_name']].astype(target_type)

            if target_type == "category":
                dataframe[x['column_name']] = dataframe[x['column_name']].cat.codes

        else:
            if verbose:
                print("No target type specified for column %s" % x['column_name'])

    # # 2. ADHOC TRANSFORMATION
    for _, x in enumerate(feature_param['transformation']):

        transform_type = x['type']
        transform_column = x['column_name']
        action = x['action']
        params = x['parameters']
        assign = x['assign']

        if transform_type == 'series':
            print("Transforming %s with function %s and parameters %s. Assign to %s" %
                  (transform_column, action, params, assign))
            series = getattr(dataframe[transform_column], action)(**params)
            dataframe = dataframe.assign(**{assign: series})

        elif transform_type == 'cross-series':
            other = dataframe[x['other']]
            assert type(other) == pd.Series
            print("Transforming %s with function %s and parameters %s, cross column %s. Assign to %s" %
                  (transform_column, action, params, other.name, assign))
            params = {'other': other, **params}

            series = getattr(dataframe[transform_column], action)(**params)
            dataframe = dataframe.assign(**{assign: series})

    # 3. Aggregation
    for _, x in enumerate(feature_param['aggregation']):
        pass


    #
    # # 3. Drop unused columns
    # dataframe = dataframe.drop(columns=DROP_COLS_BEFORE_TRAINING, axis=1)

    return dataframe


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


def train():
    # Load the file
    features_list, target = load_data(FEATURE_FILE_LIST, set_index='SK_ID_CURR')

    print(features_list[0].info())
    print(features_list[1].info())

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