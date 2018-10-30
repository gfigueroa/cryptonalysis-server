import logging
from datetime import date
from transaction_builders import BiffPredictorSmart
from preprocessing import run_data_pipeline, HISTORICAL_DATA_FILE
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
import config


# Logging
logging.basicConfig(level=config.LOGGING_LEVEL)
logger = logging.getLogger()


def split_datasets(df, shuffle, training_size, dev_size):
    """
    type: (pd.DataFrame, bool, int, int) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                             pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame)
    Get split datasets from a DataFrame used for non-CV (no Cross Validation), CV, and Grid Search CV.
    For non-CV, the DF is split into X_training, y_training, X_testing, y_testing.
    For CV, the given DF is simply split into X (attributes) and y (target/classes).
    For Grid Search CV, the DF is split into X_dev, y_dev, X_eval, y_eval. In Grid Search CV,
    the X and y must be split into development and evaluation subsets, as the development data
    is used for finding the best hyperparameter values, and the evaluation data is used
    to test a model with those values (and avoid overfitting).
    :param df: The DataFrame to split
    :param shuffle: Whether or not to shuffle the rows of the DF (TODO: use in time series?)
    :param training_size: The size of the training dataset (0~1)
    :param dev_size: The size of the development dataset (0~1)
    :return:
    """
    if shuffle:
        shuffled_df = df.sample(frac=1)

        # Used for non-CV
        training_data = shuffled_df.sample(frac=training_size)
        testing_data = shuffled_df[~shuffled_df.index.isin(training_data.index)]

        # Used for CV
        X = shuffled_df.iloc[:, :-1]
        y = shuffled_df['transaction']

        # Used for Grid Search CV
        X_dev, X_eval, y_dev, y_eval = train_test_split(X, y, test_size=dev_size)
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


# Grid Search CV (development and evaluation datasets) function


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
    :return: The trained Grid Search classifier with optimized hyperparameters
    :rtype: GridSearchCV
    """
    logger.info("Tuning hyperparameters for {0} for accuracy...".format(classifier.__class__.__name__))

    clf = GridSearchCV(classifier, tuned_parameters, cv=k, scoring='accuracy')
    clf.fit(X_dev, y_dev)

    logger.info("Best parameters set found on development dataset:")
    logger.info(clf.best_params_)
    best_index = clf.best_index_
    best_score = clf.cv_results_['mean_test_score'][best_index]
    best_std = clf.cv_results_['std_test_score'][best_index]
    logger.info("Accuracy: %0.3f (+/-%0.03f)" % (best_score, best_std * 2))

    logger.debug("\nGrid scores on development set:\n")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    parameters = clf.cv_results_['params']
    for mean, std, params in zip(means, stds, parameters):
        logger.debug("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print '\n'

    # Evaluation dataset
    logger.info("Detailed classification report:")
    logger.info("The model is trained on the full development set (size {0}).".format(len(X_dev)))
    logger.info("The scores are computed on the full evaluation set (size {0}).".format(len(X_eval)))
    y_true, y_pred = y_eval, clf.predict(X_eval)
    logger.info('\n' + classification_report(y_true, y_pred))

    return clf


def run_classification_pipeline(preprocessed_df, shuffle_data=True, training_size=0.7, dev_size=0.5):
    """
    Run the classification pipeline. The function returns an optimized and trained classification model.
    :param preprocessed_df: The preprocessed DataFrame ready for the classification pipeline
    :param shuffle_data: Whether or not to shuffle (rows) the training and testing data (default is True)
    :param training_size: The size (0~1) of the training dataset (used in non-CV)
    :param dev_size: The size (0~1) of the development dataset (used in Grid Search CV)
    :return:
    """

    logger.info("Running classification pipeline...")
    logger.info("Parameters:\nShuffle data: {0}, Training size: {1}, Development size: {2}".format(shuffle_data,
                training_size, dev_size))

    # 1. Split dataset for classification
    X, y, X_training, y_training, X_testing, y_testing, X_dev, y_dev, X_eval, y_eval = \
        split_datasets(preprocessed_df, shuffle_data, training_size, dev_size)

    # 2. Grid Search CV with SVMs
    svc = svm.SVC()
    svm_tuned_parameters = [{'kernel': ["rbf", "poly"], 'gamma': [1e-3, 1e-4], 'C': [0.01, 0.1, 1, 10, 100, 1000]},
                            {'kernel': ["linear"], 'C': [0.01, 0.1, 1, 10, 100, 1000]}]
    grid_search_cv_svc = get_optimized_classifier(svc, svm_tuned_parameters, X_dev, y_dev, X_eval, y_eval)

    # 3. Grid Search CV with NNs
    mlp = MLPClassifier()
    mlp_tuned_parameters = {
        'learning_rate': ["constant", "invscaling", "adaptive"],
        'hidden_layer_sizes': [(10, 10, 10), (20, 20, 20), (30, 30, 30)],
        'alpha': [0.1, 1, 10, 100],
        'activation': ["identity", "logistic", "tanh", "relu"]
    }
    grid_search_cv_nn = get_optimized_classifier(mlp, mlp_tuned_parameters, X_dev, y_dev, X_eval, y_eval)

    logger.info("Classification pipeline complete!\n")


if __name__ == '__main__':
    # Grid search data pipeline parameters
    start_date = date(2016, 1, 1)
    predictor_cls = BiffPredictorSmart
    predictor_params = {
        'prob_buy': 1,
        'prob_sell': 1,
        'starting_investment': 100,
        'daily_allowance': 5,
        'lookahead_days': 4
    }
    data_pipeline_parameters = {
        'window_size': [10, 20, 30, 40, 50, 60],
        'normalize_by_row': [True,  False]
    }
    # Grid search classification pipeline parameters
    classification_pipeline_parameters = {
        'shuffle_data': [True, False]
    }

    # Grid search data pipeline parameters
    for window_size in data_pipeline_parameters['window_size']:
        for normalize_by_row in data_pipeline_parameters['normalize_by_row']:
            try:
                preprocessed_data = run_data_pipeline(HISTORICAL_DATA_FILE, starting_date=start_date,
                                                      window_size=window_size, normalize_by_row=normalize_by_row,
                                                      predictor_class=predictor_cls, **predictor_params)
            except Exception as e:
                logger.error("Error in data pipeline! Skipping...")
                logger.error(e.message)
                break

            # Grid search classification pipeline parameters
            for shu in classification_pipeline_parameters['shuffle_data']:
                try:
                    run_classification_pipeline(preprocessed_data, shuffle_data=shu)
                except Exception as e:
                    logger.error("Error in classification pipeline! Skipping...")
                    logger.error(e.message)
