import logging
import numpy as np
import os
import sys
from cryptonalysis.config import load_cryptonalysis_config_grid, PreprocessingConfig, TrainingConfig, \
    CryptonalysisConfigGrid
from datetime import datetime
from joblib import dump, load
from pandas import DataFrame
from preprocessing import run_preprocessing_pipeline
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier


# Logging
logger = logging.getLogger()

RESULTS_DIR = os.path.join('cryptonalysis', 'results')
MODELS_DIR = os.path.join('cryptonalysis', 'models')


def split_datasets(df, shuffle, training_size=0.7, dev_size=0.5, shuffled_indices=None):
    """
    Get split datasets from a DataFrame used for non-CV (no Cross Validation), CV, and Grid Search CV.
    For non-CV, the DF is split into X_training, y_training, X_testing, y_testing.
    For CV, the given DF is simply split into X (attributes) and y (target/classes).
    For Grid Search CV, the DF is split into X_dev, y_dev, X_eval, y_eval. In Grid Search CV,
    the X and y must be split into development and evaluation subsets, as the development data
    is used for finding the best hyperparameter values, and the evaluation data is used
    to test a model with those values (and avoid overfitting).
    :param df: The DataFrame to split
    :type df: DataFrame
    :param shuffle: Whether or not to shuffle the rows of the DF
    :type shuffle: bool
    :param training_size: (default: 0.7) The size (0~1) of the training dataset (used in non-CV)
    :type training_size: float
    :param dev_size: (default: 0.5) The size (0~1) of the development dataset (used in Grid Search CV)
    :type dev_size: float
    :param shuffled_indices: A dictionary of fixed indices to use to shuffle the data. Each key represents the index for
    a split type (namely: 'cv', 'training', 'testing', 'dev', 'eval').
    This is useful when multiple dataframes need to be shuffled with the same arrangement (e.g. when using multiple
    channels in a neural network).
    If shuffle=False, this parameter is ignored.
    :type shuffled_indices: dict
    :return: a tuple with different splits for the given DataFrame
    :rtype: (DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame,
    DataFrame)
    """
    if shuffle:
        if shuffled_indices:
            # Used for non-CV
            training_data = df.reindex(shuffled_indices['training'])
            testing_data = df.reindex(shuffled_indices['testing'])

            # Used for CV
            shuffled_df = df.reindex(shuffled_indices['cv'])
            X = shuffled_df.iloc[:, :-1]
            y = shuffled_df['transaction']

            # Used for Grid Search CV
            X_dev = df.reindex(shuffled_indices['dev']).iloc[:, :-1]
            y_dev = df.reindex(shuffled_indices['dev'])['transaction']
            X_eval = df.reindex(shuffled_indices['eval']).iloc[:, :-1]
            y_eval = df.reindex(shuffled_indices['eval'])['transaction']
        else:
            shuffled_df = df.sample(frac=1)

            # Used for non-CV
            training_data = shuffled_df.sample(frac=training_size)
            testing_data = shuffled_df[~shuffled_df.index.isin(training_data.index)]

            # Used for CV
            X = shuffled_df.iloc[:, :-1]
            y = shuffled_df['transaction']

            # Used for Grid Search CV
            X_dev, X_eval, y_dev, y_eval = train_test_split(X, y, test_size=1-dev_size)
    else:
        # Used for non-CV
        training_data = df[:int(len(df) * training_size)]
        testing_data = df[~df.index.isin(training_data.index)]

        # Used for CV
        X = df.iloc[:, :-1]
        y = df['transaction']

        # Used for Grid Search CV
        dev_set = df[:int(len(df) * dev_size)]
        eval_set = df[~df.index.isin(dev_set.index)]
        X_dev = dev_set.iloc[:, :-1]
        y_dev = dev_set.iloc[:, -1]
        X_eval = eval_set.iloc[:, :-1]
        y_eval = eval_set.iloc[:, -1]

    # Used for non-CV
    X_training = training_data.iloc[:, :-1]
    y_training = training_data.iloc[:, -1]
    X_testing = testing_data.iloc[:, :-1]
    y_testing = testing_data.iloc[:, -1]

    logger.info("Length X: {0}".format(len(X)))
    logger.info("Length X_training: {0}".format(len(X_training)))
    logger.info("Length X_testing: {0}".format(len(X_testing)))
    logger.info("Length X_dev: {0}".format(len(X_dev)))
    logger.info("Length X_eval: {0}".format(len(X_eval)))

    return X, y, X_training, y_training, X_testing, y_testing, X_dev, y_dev, X_eval, y_eval


def get_optimized_classifier(classifier, tuned_parameters, X_dev, y_dev, X_eval, y_eval, k=4):
    """
    Perform a Grid Search algorithm with Cross Validation to find the optimal hyperparameter values and test the
    optimized model.
    :param classifier: The classifier to use
    :param tuned_parameters: A dictionary of parameter combinations to test
    :param X_dev: The development dataset attributes
    :param y_dev: The development dataset targets (labels)
    :param X_eval: The evaluation dataset attributes
    :param y_eval: The evaluation dataset targets (labels)
    :param k: (default: 4) The number of folds used in cross validation
    :return: A tuple containing the trained Grid Search classifier with optimized hyperparameters, the training accuracy
    and the evaluation accuracy (classifier, training accuracy, evaluation accuracy).
    :rtype: (GridSearchCV, float, float)
    """
    logger.info("Tuning hyperparameters for {0} for accuracy...".format(classifier.__class__.__name__))

    clf = GridSearchCV(classifier, tuned_parameters, cv=k, scoring='accuracy')
    clf.fit(X_dev, y_dev)

    logger.info("Best parameters set found on development dataset:")
    logger.info(clf.best_params_)
    best_index = clf.best_index_
    best_score = clf.cv_results_['mean_test_score'][best_index]
    best_std = clf.cv_results_['std_test_score'][best_index]
    logger.info("Training accuracy (%s): %0.3f (+/-%0.03f)" % (classifier.__class__.__name__, best_score, best_std * 2))

    logger.debug("\nGrid scores on development set:\n")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    parameters = clf.cv_results_['params']
    for mean, std, params in zip(means, stds, parameters):
        logger.debug("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    # Evaluation dataset
    logger.info("Evaluation results:")
    y_true, y_pred = y_eval, clf.predict(X_eval)
    logger.info('\n' + classification_report(y_true, y_pred))
    accuracy = accuracy_score(y_true, y_pred)
    logger.info("Evaluation accuracy ({}): {}\n".format(classifier.__class__.__name__, accuracy))

    return clf, best_score, accuracy


def _get_training_run_name(crypto, preprocessing_config, training_config):
    """
    Get a string with the name of the training run given preprocessing and training configurations.

    >>> _get_training_run_name('LTC',
    ...     PreprocessingConfig(
    ...     {
    ...         'window_size': 60,
    ...         'predictor_params': {
    ...             'lookahead_days': 3, 'prob_buy': 1, 'prob_sell': 1, 'starting_investment': 100, 'daily_allowance': 5
    ...         },
    ...         'start_date': '2018-01-01',
    ...         'end_date': '2019-05-04',
    ...         'price_column': 'Close',
    ...         'normalize': True,
    ...         'normalize_by_row': True,
    ...         'standardize': False,
    ...         'predictor_cls': 'BiffPredictor',
    ...     }),
    ...     TrainingConfig(
    ...     {
    ...         'dev_size': 0.5,
    ...         'shuffle_data': True,
    ...         'training_size': 0.7,
    ...         'cv_folds': 4
    ...     })
    ... )
    'LTC_2019-05-04TrueTrueBiffPredictor5311100CloseFalse2018-01-0160_40.5True0.7'

    :param crypto: The crypto name
    :type crypto: str
    :param preprocessing_config
    :type preprocessing_config: PreprocessingConfig
    :param training_config
    :type training_config: TrainingConfig
    :return: A file name
    :rtype: str
    """

    return "{}_{}_{}".format(crypto, preprocessing_config.to_single_line_str(), training_config.to_single_line_str())


def save_training_results(classifiers, crypto, preprocessing_config, training_config):
    """
    Save training results to a file.
    :param classifiers: dictionary of classifiers, containing classifier, training accuracy and eval accuracy.
    :type classifiers: dict
    :param crypto: The name of the crypto
    :type crypto: str
    :param preprocessing_config
    :type preprocessing_config: PreprocessingConfig
    :param training_config
    :type training_config: TrainingConfig
    """

    if not os.path.exists(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)

    # Human-readable
    hr_file_name = "results_{}.txt".format(datetime.strftime(datetime.now(), '%Y-%m-%d'))
    logger.info("Saving results to {}...".format(os.path.join(RESULTS_DIR, hr_file_name)))
    with open(os.path.join(RESULTS_DIR, hr_file_name), 'a') as f:
        f.write(crypto + '\n')
        f.write(str(preprocessing_config) + '\n')
        f.write(str(training_config) + '\n')
        f.write(str(classifiers) + '\n')
        f.write('\n**************************************************************\n')

    # CSV
    csv_file_name = "results_{}.csv".format(datetime.strftime(datetime.now(), '%Y-%m-%d'))
    logger.info("Saving results to {}...".format(os.path.join(RESULTS_DIR, csv_file_name)))
    classifier_names = sorted(classifiers.keys())
    classifier_strings = ["{},{}".format(classifiers[k]['training_acc'], classifiers[k]['eval_acc'])
                          for k in classifier_names]
    classifiers_header = ','.join(["{}_training,{}_eval".format(k, k)
                                  for k in classifier_names])
    classifiers_csv = ','.join(classifier_strings)
    headers = "{},{},{},{}".format('crypto', preprocessing_config.to_csv_str()[0], training_config.to_csv_str()[0],
                                   classifiers_header)
    line = "{},{},{},{}".format(crypto, preprocessing_config.to_csv_str()[1], training_config.to_csv_str()[1],
                                classifiers_csv)

    file_exists = os.path.isfile(os.path.join(RESULTS_DIR, csv_file_name))
    with open(os.path.join(RESULTS_DIR, csv_file_name), 'a') as f:
        if not file_exists:
            f.write(headers + '\n')

        f.write(line + '\n')

    training_run_touch_file = _get_training_run_name(crypto, preprocessing_config, training_config)
    with open(os.path.join(RESULTS_DIR, training_run_touch_file), 'w') as f:
        f.write('complete')


def get_model_name(classifier, crypto, preprocessing_config, training_config):
    """
    Get a string with the name of the trained model to save/load given preprocessing and training configurations.

    >>> get_model_name('SVC', 'LTC',
    ...     PreprocessingConfig(
    ...     {
    ...         'window_size': 60,
    ...         'predictor_params': {
    ...             'lookahead_days': 3, 'prob_buy': 1, 'prob_sell': 1, 'starting_investment': 100, 'daily_allowance': 5
    ...         },
    ...         'start_date': '2018-01-01',
    ...         'end_date': '2019-05-04',
    ...         'price_column': 'Close',
    ...         'normalize': True,
    ...         'normalize_by_row': True,
    ...         'standardize': False,
    ...         'predictor_cls': 'BiffPredictor',
    ...     }),
    ...     TrainingConfig(
    ...     {
    ...         'dev_size': 0.5,
    ...         'shuffle_data': True,
    ...         'training_size': 0.7,
    ...         'cv_folds': 4
    ...     })
    ... )
    'SVC_LTC_2019-05-04TrueTrueBiffPredictor5311100CloseFalse2018-01-0160_40.5True0.7.joblib'

    :param classifier: The classifier name
    :type classifier: str
    :param crypto: The crypto name
    :type crypto: str
    :param preprocessing_config
    :type preprocessing_config: PreprocessingConfig
    :param training_config
    :type training_config: TrainingConfig
    :return: A file name
    :rtype: str
    """
    training_run_name = _get_training_run_name(crypto, preprocessing_config, training_config)
    return "{}_{}.joblib".format(classifier, training_run_name)


def save_model(classifier, crypto, preprocessing_config, training_config, model_dir=None):
    """
    Save a trained model to a file.
    :param classifier: A trained classifier to save as a model
    :type classifier: GridSearchCV
    :param crypto: The name of the crypto
    :type crypto: str
    :param preprocessing_config
    :type preprocessing_config: PreprocessingConfig
    :param training_config
    :type training_config: TrainingConfig
    :param model_dir: (Default None) The (overridden) directory where the model should be saved.
    If None, the default `MODELS_DIR` is used.
    :type model_dir: str
    """

    dir_to_use = MODELS_DIR if model_dir is None else model_dir
    if not os.path.exists(dir_to_use):
        os.mkdir(dir_to_use)

    classifier_type = classifier.estimator.__class__.__name__
    file_name = get_model_name(classifier_type, crypto, preprocessing_config, training_config)
    logger.info("Saving model {}...".format(file_name))
    dump(classifier, os.path.join(dir_to_use, file_name))


def load_model(classifier_type, crypto, preprocessing_config, training_config, model_dir=None):
    """
    Load trained model from a file.
    :param classifier_type: name of the classifier whose model will be loaded, such as 'SVC' or 'MLPClassifier'
    :type classifier_type: str
    :param crypto: The name of the crypto
    :type crypto: str
    :param preprocessing_config
    :type preprocessing_config: PreprocessingConfig
    :param training_config
    :type training_config: TrainingConfig
    :param model_dir: (Default None) The (overridden) directory where the model should be loaded from.
    If None, the default `MODELS_DIR` is used.
    :type model_dir: str
    :return A trained model
    :rtype: GridSearchCV
    """

    dir_to_use = MODELS_DIR if model_dir is None else model_dir
    file_name = get_model_name(classifier_type, crypto, preprocessing_config, training_config)
    logger.info("Loading model {}...".format(file_name))
    try:
        model = load(os.path.join(dir_to_use, file_name))
        return model
    except IOError:
        raise ValueError("Model with name '{}' not found!".format(file_name))


def run_training_pipeline(preprocessed_df, training_config):
    """
    Run the classification pipeline. The function returns a dictionary of optimized and trained classification models.
    :param preprocessed_df: The preprocessed DataFrame ready for the classification pipeline
    :param training_config
    :type training_config: TrainingConfig
    :return A dictionary containing the optimized trained classification models with some metadata.
    :rtype: dict
    """

    logger.info("Running training pipeline...")
    logger.info("Training config:\n" + str(training_config))

    # Split dataset for classification
    X, y, X_training, y_training, X_testing, y_testing, X_dev, y_dev, X_eval, y_eval = \
        split_datasets(preprocessed_df, training_config.shuffle_data, training_config.training_size,
                       training_config.dev_size)

    # Grid Search CV with SVMs
    svc = svm.SVC()
    svm_tuned_parameters = [{'kernel': ["rbf", "poly"], 'gamma': [1e-4, 1e-3, 0.01, 0.1, 1],
                             'C': [0.01, 0.1, 1, 10]},
                            {'kernel': ["linear"], 'C': [0.01, 0.1, 1, 10]}]
    grid_search_cv_svc, svc_training_acc, svc_eval_acc = \
        get_optimized_classifier(svc, svm_tuned_parameters, X_dev, y_dev, X_eval, y_eval, training_config.cv_folds)

    # Grid Search CV with NNs
    mlp = MLPClassifier()
    mlp_tuned_parameters = {
        'learning_rate': ["constant", "invscaling", "adaptive"],
        'hidden_layer_sizes': [(10, 10, 10), (20, 20, 20), (30, 30, 30), (40, 40, 40), (50, 50, 50)],
        'alpha': [0.1, 1, 10],
        'activation': ["identity", "logistic", "tanh", "relu"]
    }
    grid_search_cv_mlp, mlp_training_acc, mlp_eval_acc = \
        get_optimized_classifier(mlp, mlp_tuned_parameters, X_dev, y_dev, X_eval, y_eval)

    logger.info("Classification pipeline complete!\n")

    trained_classifiers = {
        'SVC': {
            'classifier': grid_search_cv_svc,
            'training_acc': svc_training_acc,
            'eval_acc': svc_eval_acc
        },
        'MLPClassifier': {
            'classifier': grid_search_cv_mlp,
            'training_acc': mlp_training_acc,
            'eval_acc': mlp_eval_acc
        }
    }

    return trained_classifiers


def run_classic_training(cryptonalysis_config_grid):
    """
    Run classic training using grid search hyperparameter optimization.
    :param cryptonalysis_config_grid
    :type cryptonalysis_config_grid: CryptonalysisConfigGrid
    """

    logger.info("Config grid size: {}".format(cryptonalysis_config_grid.grid_size))

    # Grid search preprocessing pipeline parameters
    count = 1
    for crypto in cryptonalysis_config_grid.cryptos:
        logger.info("Crypto: {}".format(crypto))
        for preprocessing_config in cryptonalysis_config_grid.preprocessing_config_grid:
            try:
                preprocessed_data = run_preprocessing_pipeline(crypto, preprocessing_config,
                                                               cryptonalysis_config_grid.save_preprocessing_data,
                                                               cryptonalysis_config_grid.save_preprocessing_roi)
            except Exception as e:
                logger.error("Error in preprocessing pipeline! Skipping...", exc_info=e)
                break

            # Grid search training pipeline parameters
            for training_config in cryptonalysis_config_grid.training_config_grid:
                logger.info("Processing configuration {}/{}...".format(count, cryptonalysis_config_grid.grid_size))
                count += 1

                # Check if training run has been executed
                if cryptonalysis_config_grid.save_training_results:
                    training_run_touch_file = _get_training_run_name(crypto, preprocessing_config, training_config)
                    if os.path.exists(os.path.join(RESULTS_DIR, training_run_touch_file)):
                        logger.info("Training run has already been executed, skipping...")
                        continue

                try:
                    classifiers = run_training_pipeline(preprocessed_data, training_config)
                    if cryptonalysis_config_grid.save_training_results:
                        save_training_results(classifiers, crypto, preprocessing_config, training_config)
                    if cryptonalysis_config_grid.save_training_model:
                        for classifier_data in classifiers.values():
                            save_model(classifier_data['classifier'], crypto, preprocessing_config, training_config)
                except Exception as e:
                    logger.error("Error in training pipeline {}! Skipping...".format(e.message), exc_info=e)


if __name__ == '__main__':
    # random seed for reproducibility
    np.random.seed(202)

    if len(sys.argv) < 2:
        raise ValueError('Config file not given in args!')

    config_file = sys.argv[1]

    config_path = 'config'
    config_grid = load_cryptonalysis_config_grid(config_path, config_file)

    run_classic_training(config_grid)
