import os
import xgboost as xgb

from utils import *
from models.XGBModel import XGBModel

FEATURE_FILE_LIST = [
    {'file': 'data/application_test.csv',
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

OUTPUT_DIR = 'output'

def infer():

    # Load the model

    BEST_HPARAMS = {
        'max_depth': 5,
        'min_child_weight': 7,
        'gamma': 4,
        'eta': 0.65,
        'objective': 'binary:logistic',
        'nthread': 4,
        'eval_metric': 'auc',
        'silent': 1
    }

    model_path = os.path.join(os.getcwd(), OUTPUT_DIR, 'Trial1/xgb_21.model')

    model = XGBModel(pretrained_model_path=model_path)

    print(model.model.attributes())

    # Load the data
    merged_features, _ = load_data(FEATURE_FILE_LIST, set_index='SK_ID_CURR')

    # Infer the data
    dtest = xgb.DMatrix(merged_features.values)

    pred = model.infer(dtest, index=None, output_as='proba', to_csv=True)

    print(pred.info())





if __name__ == "__main__":
    infer()