import glob
import logging
import os

# Logging
logger = logging.getLogger()

RESULTS_DIR = os.path.join('cryptonalysis', 'results')
MERGED_RESULTS_FILE = 'results_merged.csv'


def merge_results(results_dir=None):
    """
    Merge CSV results files into a single CSV file.
    :param results_dir: (Default None) The (overridden) directory where the results should be loaded from.
    If None, the default `RESULTS_DIR` is used.
    :type results_dir: str
    :return:
    """
    dir_to_use = results_dir or RESULTS_DIR
    merged_csv_path = os.path.join(dir_to_use, MERGED_RESULTS_FILE)
    logger.info("Merging results into {}...".format(merged_csv_path))

    file_filter = "{}{}*.csv".format(dir_to_use, os.path.sep)
    results_csv_paths = glob.glob(file_filter)
    header = None
    with open(merged_csv_path, 'w') as merged_csv:
        for results_csv_path in results_csv_paths:
            with open(results_csv_path, 'r') as results_csv:
                for i, line in enumerate(results_csv.readlines()):
                    if not line.strip():  # Ignore empty lines
                        continue

                    if not line.endswith('\n'):  # Add new-line char if missing
                        line += '\n'

                    if i == 0 and not header:  # Only write header once
                        header = line
                        merged_csv.write(header)
                    elif i > 0:
                        merged_csv.write(line)

    logger.info('Finished merging results...')


if __name__ == '__main__':
    merge_results()
