import pandas as pd
import numpy as np
import xgboost as xgb

from abc import ABCMeta, abstractclassmethod
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = [
    'SK_ID_CURR',
    'NAME_CONTRACT_TYPE',
    'CODE_GENDER',
    'CNT_CHILDREN',
    'NAME_INCOME_TYPE',
    'NAME_EDUCATION_TYPE',
    'NAME_HOUSING_TYPE',
    'DAYS_EMPLOYED',
    'DAYS_REGISTRATION',
    'DAYS_ID_PUBLISH',
    'OWN_CAR_AGE',
    'FLAG_MOBIL',
    'FLAG_EMP_PHONE',
    'FLAG_WORK_PHONE',
    'OCCUPATION_TYPE',
    'CNT_FAM_MEMBERS',
    'REGION_RATING_CLIENT',
    'ORGANIZATION_TYPE',
    'EXT_SOURCE_1',
    'EXT_SOURCE_2',
    'EXT_SOURCE_3',
    'APARTMENTS_AVG',
    'DAYS_LAST_PHONE_CHANGE',
    'FLAG_DOCUMENT_2',
    'FLAG_DOCUMENT_6',
    'FLAG_DOCUMENT_7',
    'FLAG_DOCUMENT_14',
    'FLAG_DOCUMENT_15',
    'FLAG_DOCUMENT_18',
    'FLAG_DOCUMENT_21',
    'AMT_REQ_CREDIT_BUREAU_YEAR'
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

LABEL_COLUMN = 'TARGET'

# Model
hparams = {
    'max_depth': 10,
    'eta': 1,
    'silent': 1,
    'objective': 'binary:logistic',

}


class Model(object):

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractclassmethod
    def train(self):
        raise NotImplementedError()

    @abstractclassmethod
    def val(self):
        raise NotImplementedError()




def load_data(file_path, feature_columns, label_column, set_index=None, verbose=False):

    dataframe = pd.read_csv(file_path)
    features = dataframe.loc[:, feature_columns]

    target = dataframe[label_column]

    if verbose:
        print(features.info(verbose=True))
        print(target.value_counts())

    if set_index is not None:
        dataframe.set_index(set_index)

    return features, target



def split_data(features, target, test_size=0.1, random_state=42, verbose=False):

    if verbose:
        print("Splitting data...")

    # Split the dataframe to features and target
    X_train, X_val, y_train, y_val = train_test_split(features,
                                                      target,
                                                      test_size=test_size,
                                                      random_state=random_state,
                                                      stratify=target)

    return X_train, X_val, y_train, y_val


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

    # 2. ADHOC TRANSFORMATION
    for _, (field_name, action) in enumerate(SERIES_TRANSFORMATION_COLS.items()):
        print("Transforming %s with function %s and parameters %s" %
              (field_name, action['action'], action['parameters']))
        series = getattr(dataframe[field_name], action['action'])(**action['parameters'])
        dataframe = dataframe.assign(**{action['assign']: series})

    return dataframe




def train():
    # Load the file
    features, target = load_data('data/application_train.csv',
                                 feature_columns=FEATURE_COLUMNS,
                                 label_column=LABEL_COLUMN)

    transformed_features = transform(features, verbose=True)
    print(transformed_features.info(verbose=True))
    # Validation
    print(transformed_features[transformed_features.DAYS_EMPLOYED == 365243])
    print(transformed_features.LOG_DAYS_REGISTRATION)

    # Split dataset
    X_train, X_val, y_train, y_val = split_data(features, target)




if __name__ == "__main__":
    train()