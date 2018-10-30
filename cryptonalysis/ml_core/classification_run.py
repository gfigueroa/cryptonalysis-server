import logging
from datetime import date
from transaction_builders import BiffPredictorSmart
from preprocessing import run_data_pipeline, HISTORICAL_DATA_FILE
from classification import run_classification_pipeline
import config


# Logging
logging.basicConfig(level=config.LOGGING_LEVEL)
logger = logging.getLogger()


if __name__ == '__main__':
    # Data pipeline parameters
    start_date = date(2015, 1, 1)
    predictor_cls = BiffPredictorSmart
    predictor_params = {
        'prob_buy': 1,
        'prob_sell': 1,
        'starting_investment': 100,
        'daily_allowance': 5,
        'lookahead_days': 4
    }
    data_pipeline_parameters = {
        'window_size': [60],
        'normalize_by_row': [True]
    }
    # Grid search classification pipeline parameters
    classification_pipeline_parameters = {
        'shuffle_data': [False]
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
