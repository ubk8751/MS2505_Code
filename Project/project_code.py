import os
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from VectorDictionary import VectorDictionary
from BayesClassifier import BayesClassifier

def gen_parser():
    parser = argparse.ArgumentParser(description='Bayesian Spam Classifier with Argument Parsing')

    # Dataset arguments
    parser.add_argument('--filepath',
                        type=str,
                        default='Project/data/mail_data.csv',
                        help='Path to the dataset file.')
    parser.add_argument('--label_col',
                        type=str,
                        default='Category',
                        help='Name of the label column.')
    parser.add_argument('--text_col',
                        type=str,
                        default='Message',
                        help='Name of the text column.')
    parser.add_argument('--bin-labels',
                        action='store_false',
                        help='Wether labels should be binary or not.')
    parser.add_argument('--test_size',
                        type=float,
                        default=0.2,
                        help='Proportion of the dataset to include in the test split.')
    parser.add_argument('--random_seed',
                        type=int,
                        default=None,
                        help='Random seed for replication.')

    # Vectorizer arguments
    parser.add_argument('--optimize_dictionary',
                        type=bool,
                        default=True,
                        help='Whether to optimize the dictionary.')
    parser.add_argument('--max_features',
                        type=int,
                        default=None,
                        help='Maximum number of features for the vectorizer.')
    parser.add_argument('--max_df',
                        type=float,
                        default=1.0,
                        help='Max document frequency for the vectorizer.')
    parser.add_argument('--min_df',
                        type=int,
                        default=1,
                        help='Min document frequency for the vectorizer.')

    # Classifier arguments
    parser.add_argument('--alpha',
                        type=float,
                        default=1.0,
                        help='Smoothing parameter for Laplace smoothing.')
    parser.add_argument('--verbose',
                        type=int,
                        default=0,
                        help='Verbosity level for debugging and messages.')

    args = parser.parse_args()

    return args

def warning_print(message: str):
    '''
    Print warning messages in a formatted way.

    :param message: Warning message string.
    '''
    print(message)

def load_data(filepath: str) -> pd.DataFrame:
    '''
    Load the dataset from the given filepath.

    :param filepath: Path to the dataset file.
    :return: Loaded DataFrame.
    '''
    full_path = os.path.join(os.getcwd(), filepath)
    print(full_path, os.path.exists(full_path))
    if not filepath:
        raise ValueError('Error! No filepath provided!')
    if not filepath.lower().endswith('.csv'):
        raise ValueError(f"Error! The file '{filepath}' does not have a .csv extension.")
    if not os.path.exists(full_path):
        raise FileNotFoundError(f'{filepath} does not exist!')

    try:
        ds = pd.read_csv(full_path)
    except Exception as e:
        raise ValueError(f"Error! The file '{filepath}' could not be read as a CSV. Details: {e}")
    
    print(f'Read dataset {os.path.basename(filepath)}!')
    print(ds.info())
    print(ds.head(5))

    nan_rows = ds.isna().any(axis=1).sum()
    if nan_rows > 0:
        warning_print(
            f'WARNING: There {"is" if nan_rows == 1 else "are"} {nan_rows} '
            f'row{"s" if nan_rows > 1 else ""} containing NaN values in the dataset!'
        )

    return ds

def preprocess_data(ds: pd.DataFrame,
                    label_col: str,
                    text_col: str,
                    bin_labels: bool = True) -> pd.DataFrame:
    '''
    Preprocess the dataset by binarizing labels and ensuring no missing data.

    :param ds: Input DataFrame.
    :param label_col: The name of the column containing labels.
    :param text_col: The name of the column containing text data.
    :param bin_labels: Whether to binarize the labels.
    :return: Preprocessed DataFrame.
    '''
    if bin_labels:
        ds[label_col] = ds[label_col].map({'spam': 1, 'ham': 0})

    ds = ds.dropna(subset=[text_col, label_col])
    return ds

def train_and_evaluate(train_ds: pd.DataFrame = None,
                       test_ds: pd.DataFrame = None,
                       text_col: str = 'Message',
                       label_col: str = 'Category'):
    '''
    Train and evaluate the Bayes Classifier.

    :param train_ds: Training DataFrame.
    :param test_ds: Testing DataFrame.
    :param text_col: The name of the column containing text data.
    :param label_col: The name of the column containing labels.
    '''
    
    dictionary = VectorDictionary(
        docs=train_ds[text_col],
        optimize_dictionary=True,
        max_features=None,
        max_df=1.0,
        min_df=1
    )

    classifier = BayesClassifier(
        max_features=dictionary.vectorizer_params.get('max_features'),
        alpha=1.0,
        max_df=dictionary.vectorizer_params.get('max_df'),
        min_df=dictionary.vectorizer_params.get('min_df'),
        verbose=1
    )
    classifier.fit(docs=train_ds[text_col],
                   labels=train_ds[label_col],
                   vectorizer=dictionary.vectorizer)
    print('Bayesian Classifier trained successfully!')

    results = classifier.evaluate(docs=test_ds[text_col], labels=test_ds[label_col])

    return results

def main(filepath: str,
         label_col: str,
         text_col: str,
         test_size: float,
         bin_labels: bool,
         random_seed: bool):
    '''
    Main function to run the script.

    :param filepath: Path to the dataset file.
    '''
    ds = load_data(filepath)
    ds = preprocess_data(ds,
                         label_col=label_col,
                         text_col=text_col,
                         bin_labels=bin_labels)

    train_ds, test_ds = train_test_split(ds,
                                         test_size=test_size,
                                         random_state=random_seed,
                                         stratify=ds[label_col])

    results = train_and_evaluate(train_ds=train_ds,
                                 test_ds=test_ds,
                                 text_col=text_col,
                                 label_col=label_col)
    print(results)

def out_project(filepath: str,
                label_col: str,
                text_col: str,
                test_size: float,
                bin_labels: bool,
                random_seed: bool,
                **kwargs):
    ds = load_data(filepath=filepath)
    print(f'\nDescription of {os.path.basename(filepath)}!')
    print(ds.describe())
    ds = preprocess_data(ds=ds,
                         label_col=label_col,
                         text_col=text_col,
                         bin_labels=bin_labels)
    vectorizer = VectorDictionary(docs=ds[text_col],
                                  optimize_dictionary=True)

    classifier = BayesClassifier(**vectorizer.vectorizer_params)
    train_ds, test_ds = train_test_split(ds,
                                         test_size=test_size,
                                         random_state=random_seed,
                                         stratify=ds[label_col])

    classifier.fit(docs=train_ds[text_col],
                   labels=train_ds[label_col],
                   vectorizer=vectorizer)
    results = classifier.evaluate(docs=test_ds[text_col],
                                  labels=test_ds[label_col])
    for key in results:
        print(f'\n{key}:')
        print(results[key])
    
    

if __name__ == '__main__':
    args = gen_parser()
    '''main(filepath=args.filepath,
         label_col=args.label_col,
         text_col=args.text_col,
         test_size=args.test_size,
         bin_labels=args.bin_labels,
         random_seed=args.random_seed)'''
    out_project(filepath=args.filepath,
                label_col=args.label_col,
                text_col=args.text_col,
                test_size=args.test_size,
                bin_labels=args.bin_labels,
                random_seed=args.random_seed)