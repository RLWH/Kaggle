import numpy as np
import xgboost as xgb

from utils import *
from models.XGBModel import generate_params_set, XGBModel

# In the future, FEATURE_FILE_LIST will be generated by a web-app
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
          'assign': 'LOG_DAYS_ID_PUBLISH'}]
     },

    {'file': 'data/bureau.csv',
     'features': [
         {'column_name': 'SK_ID_CURR', 'dtype': 'int'},
         {'column_name': 'DAYS_ENDDATE_FACT', 'dtype': None},
         {'column_name': 'AMT_CREDIT_SUM_DEBT', 'dtype': None},
         {'column_name': 'AMT_CREDIT_SUM', 'dtype': None},
         {'column_name': 'AMT_CREDIT_MAX_OVERDUE', 'dtype': None},
         {'column_name': 'CREDIT_DAY_OVERDUE', 'dtype': None},
         {'column_name': 'CNT_CREDIT_PROLONG', 'dtype': None}],
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
               'DAYS_ENDDATE_FACT': 'mean',
               'CNT_CREDIT_PROLONG': 'mean',
               'AMT_CREDIT_SUM': 'mean',
               'DEBT_TO_CREDIT': 'mean'},
          'rename': {
              'CREDIT_DAY_OVERDUE': 'AVG_CREDIT_DAY_OVERDUE',
              'SK_ID_CURR': 'CB_RECORD_COUNT',
              'AMT_CREDIT_MAX_OVERDUE': 'AVG_AMT_CREDIT_MAX_OVERDUE',
              'DAYS_ENDDATE_FACT': 'AVG_DAYS_ENDDATE_FACT',
              'CNT_CREDIT_PROLONG': 'AVG_CNT_CREDIT_PROLONG',
              'AMT_CREDIT_SUM': 'AVG_AMT_CREDIT_SUM',
              'DEBT_TO_CREDIT': 'AVG_DEBT_TO_CREDIT'},
          'post_transformation': [
              {'type': 'series',
               'column_name': 'AVG_AMT_CREDIT_MAX_OVERDUE',
               'action': 'fillna',
               'parameters': {'value': 0},
               'assign': 'AVG_AMT_CREDIT_MAX_OVERDUE'}
          ]}
     }

]

DROP_COLS_BEFORE_TRAINING = ['SK_ID_CURR', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH']

LABEL_COLUMN = 'TARGET'

TRAINING_PARAMS = {
    'num_trials': 5,
    'num_rounds': 20
}


def train():
    # Load the file
    merged_features, target = load_data(FEATURE_FILE_LIST, set_index='SK_ID_CURR')

    # print(merged_features.info())

    # Split dataset
    X_train, X_val, y_train, y_val = split_data(merged_features, target)

    # Transform the dataset for xgb
    dtrain = xgb.DMatrix(X_train, label=y_train)
    deval = xgb.DMatrix(X_val, label=y_val)

    evallist = [(deval, 'eval'), (dtrain, 'train')]

    # Generate hyperparameters
    hparams_set = generate_params_set(num_trials=TRAINING_PARAMS['num_trials'])

    best_scores = []
    best_round = []

    # Build model for each hparams set
    for i, hparam in enumerate(hparams_set):

        model = XGBModel(hparam, trial=i)
        bst, scores_dict = model.train(train_dataset=dtrain,
                                       eval_dataset=evallist,
                                       num_round=TRAINING_PARAMS['num_rounds'],
                                       verbose=True)

        best_scores.append(scores_dict)
        best_round.append(bst.best_iteration)

        model.val(eval_dataset=deval, y_true=y_val)

    print("Exporting result...")
    print(best_scores)

    result = pd.DataFrame.from_dict(hparams_set)
    scores_df = pd.DataFrame.from_dict(best_scores)
    result = pd.concat([result, scores_df], axis=1)

    result.to_csv("result.csv")


if __name__ == "__main__":
    train()