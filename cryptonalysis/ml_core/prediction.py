import logging
import numpy as np
import os
import pandas as pd
import sys
from config import load_cryptonalysis_config, PreprocessingConfig, TrainingConfig, CryptonalysisConfig
from datetime import datetime
from preprocessing import run_preprocessing_pipeline
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier


# Logging
logger = logging.getLogger()

RESULTS_DIR = os.path.join(os.path.pardir, 'results')


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
        'svc': {
            'classifier': grid_search_cv_svc,
            'training_acc': svc_training_acc,
            'eval_acc': svc_eval_acc
        },
        'mlp': {
            'classifier': grid_search_cv_mlp,
            'training_acc': mlp_training_acc,
            'eval_acc': mlp_eval_acc
        }
    }

    return trained_classifiers


def save_simulation_results(classifiers, crypto, preprocessing_config, training_config):
    """
    Save simulation results to a file.
    :param classifiers: dictionary of classifiers, containing classifier info, training accuracy and eval accuracy.
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
    logger.info("Saving results to {}...".format(os.path.join(RESULTS_DIR, hr_file_name)))
    classifier_names = sorted(classifiers.keys())
    classifier_strings = ["{}_training:{},{}_eval:{}".format(k, classifiers[k]['training_acc'],
                                                             k, classifiers[k]['eval_acc'])
                          for k in classifier_names]
    classifiers_csv = [','.join(classifier_strings)]
    line = "{},{},{},{}".format(crypto, preprocessing_config.to_csv_str()[1], training_config.to_csv_str()[1],
                                classifiers_csv)
    with open(os.path.join(RESULTS_DIR, csv_file_name), 'a') as f:
        f.write(line + '\n')


def run_prediction_simulation(cryptonalysis_config):
    """
    Run a prediction simulation with unused data.
    :param cryptonalysis_config
    :type cryptonalysis_config: CryptonalysisConfig
    """

    # Preprocessing
    logger.info("Crypto: {}".format(cryptonalysis_config.crypto))
    try:
        preprocessed_data = run_preprocessing_pipeline(cryptonalysis_config.crypto, cryptonalysis_config.preprocessing)
    except Exception as e:
        logger.error("Error in preprocessing pipeline! Skipping...")
        logger.error(e.message)
        break

    # Training
    try:
        classifiers = run_training_pipeline(preprocessed_data, cryptonalysis_config.training)
        if cryptonalysis_config.training.save_results:
            save_simulation_results(classifiers, cryptonalysis_config.crypto, cryptonalysis_config.preprocessing,cryptonalysis_config.training)
    except Exception as e:
        logger.error("Error in training pipeline! Skipping...")
        logger.error(e.message)


if __name__ == '__main__':
    # random seed for reproducibility
    np.random.seed(202)

    if len(sys.argv) < 2:
        raise ValueError('Config file not given in args!')

    config_file = sys.argv[1]

    config_path = os.path.join(os.path.pardir, os.path.join(os.path.pardir, 'config'))
    config_grid = load_cryptonalysis_config(config_path, config_file)

    run_prediction_simulation(config_grid)
