import pandas as pd

from functools import reduce
from sklearn.model_selection import train_test_split


def load_data(feature_dict, set_index=None, index_dtype=None, how_join='left', verbose=False):
    """

    :param feature_dict: List of Dictionary of features.
                         In the format of [{'filename': {'features': [], 'target': []}]
    :param set_index:
    :param index_dtype: String. Data type of the index. If set_index is none, this field does not matter.
    :param how_join: String. Joining method. Supports 'left', 'right', 'outer', 'inner' only.
    :param verbose:
    :return: merged dataframe
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

        features_column_names = [feature_column['column_name'] for feature_column in file['features']]

        try:
            file_features, target = load_individual_file(file['file'],
                                                         feature_columns=features_column_names,
                                                         label_column=file['target'])
        except KeyError:
            if verbose:
                print("No target found for file %s. Extract without target" % file['file'])
            file_features = load_individual_file(file['file'], feature_columns=features_column_names)

        # Transform the dataset
        transformed_features = transform(file_features, feature_dict[i], verbose=True)

        if set_index:
            transformed_features = transformed_features.set_index(set_index)
            transformed_features.index = transformed_features.index.astype(index_dtype)

        features_list.append(transformed_features)

    # Join the dataframes
    df_merged = reduce(lambda left, right: pd.merge(left, right, how='left', left_index=True, right_index=True),
                       features_list)

    return df_merged, target


def transform(dataframe, feature_param, verbose=False):
    """

    :param dataframe:
    :param feature_param: Dictionary. Contains, feature dtypes, transformation and aggregation.
    :param verbose:
    :return:
    """

    # 1. Coerce the dataset into the correct type
    try:
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
    except KeyError:
        print("No coercion required.")

    # # 2. ADHOC TRANSFORMATION
    try:
        for _, transformation in enumerate(feature_param['transformation']):

            dataframe = make_transform(dataframe, transformation)

    except KeyError:
        print("No transformation required. ")

    # 3. Aggregation

    try:
        agg_feature = feature_param['aggregation']
        groupby = agg_feature['groupby']
        agg_params = agg_feature['agg_params']
        rename_after_agg = agg_feature['rename']
        post_transformation = agg_feature['post_transformation']

        dataframe_agg = dataframe.groupby(groupby).agg(agg_params)
        dataframe_agg = dataframe_agg.rename(index=str, columns=rename_after_agg)

        for _, transformation in enumerate(post_transformation):

            dataframe = make_transform(dataframe_agg, transformation)

        dataframe = dataframe.reset_index()

    except KeyError as err:
        print("No aggregation required. ")
        print(err)


    #
    # # 4. Drop unused columns
    # dataframe = dataframe.drop(columns=DROP_COLS_BEFORE_TRAINING, axis=1)

    return dataframe


def make_transform(dataframe, transformation, verbose=False):
    """

    :param dataframe:
    :param transformation: Dictionary.
    :param verbose: Boolean.
    :return:
    """

    transform_type = transformation['type']
    transform_column = transformation['column_name']
    action = transformation['action']
    params = transformation['parameters']
    assign = transformation['assign']

    if transform_type == 'series':
        print("Transforming %s with function %s and parameters %s. Assign to %s" %
              (transform_column, action, params, assign))
        series = getattr(dataframe[transform_column], action)(**params)
        dataframe = dataframe.assign(**{assign: series})

    elif transform_type == 'cross-series':
        other = dataframe[transformation['other']]
        assert type(other) == pd.Series
        print("Transforming %s with function %s and parameters %s, cross column %s. Assign to %s" %
              (transform_column, action, params, other.name, assign))
        params = {'other': other, **params}

        series = getattr(dataframe[transform_column], action)(**params)
        dataframe = dataframe.assign(**{assign: series})

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