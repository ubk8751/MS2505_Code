import os
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from VectorDictionary import VectorDictionary
from BayesClassifier import BayesClassifier, BayesBinomBetaClassifier

def gen_parser():
    parser = argparse.ArgumentParser(description='Bayesian Spam Classifier with Argument Parsing')

    # Dataset arguments
    parser.add_argument('--dirpath',
                        type=str,
                        default='Project/data',
                        help='Path to the dataset directory.')
    parser.add_argument('--filename',
                        type=str,
                        default='mail_data.csv',
                        help='Path to the dataset directory.')
    parser.add_argument('--label-col',
                        type=str,
                        default='Category',
                        help='Name of the label column.')
    parser.add_argument('--text-col',
                        type=str,
                        default='Message',
                        help='Name of the text column.')
    parser.add_argument('--bin-labels',
                        action='store_false',
                        help='Wether labels should be binary or not.')
    parser.add_argument('--test-size',
                        type=float,
                        default=0.2,
                        help='Proportion of the dataset to include in the test split.')
    parser.add_argument('--random-seed',
                        type=int,
                        default=None,
                        help='Random seed for replication.')
    parser.add_argument('--predict-on-ds',
                        action='store_true',
                        help='Toggles if you are going to predict on the DS or use the Binary-Beta bayes.')

    # Vectorizer arguments
    parser.add_argument('--optimize-dictionary',
                        action='store_true',
                        help='Whether to optimize the dictionary.')
    parser.add_argument('--max-features',
                        type=int,
                        default=None,
                        help='Maximum number of features for the vectorizer.')
    parser.add_argument('--max-df',
                        type=float,
                        default=1.0,
                        help='Max document frequency for the vectorizer.')
    parser.add_argument('--min-df',
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

    # Binary-Beta classifier arguments
    parser.add_argument('--bb-alpha',
                        type=int,
                        default=1,
                        help='Alpha for the binary-beta bayes prior.')
    parser.add_argument('--bb-beta',
                        type=int,
                        default=1,
                        help='Beta for the binary-beta bayes prior.')
    args = parser.parse_args()

    return args

def warning_print(message: str):
    '''
    Print warning messages in a formatted way.

    :param message: Warning message string.
    '''
    print(message, file=f)

def load_data(filepath: str) -> pd.DataFrame:
    '''
    Load the dataset from the given filepath.

    :param filepath: Path to the dataset file.
    :return: Loaded DataFrame.
    '''
    full_path = os.path.join(os.getcwd(), filepath)
    if not filepath:
        raise ValueError('Error! No filepath provided!')
    if not filepath.lower().endswith('.csv'):
        raise ValueError(f"Error! The file '{filepath}' does not have a .csv extension.")
    if not os.path.exists(full_path):
        raise FileNotFoundError(f'{filepath} does not exist!')

    try:
        ds = pd.read_csv(full_path)
    except Exception as e:
        raise ValueError(f"Error! The file '{filepath}' could not be read as a CSV. Details: {e}", file=f)
    
    print(f'NOTICE: Read dataset {os.path.basename(filepath)}!', file=f)
    print(f'\n{os.path.basename(filepath)} information:', file=f)
    print(ds.info(buf=f))
    print(f'\n First 5 rows of {os.path.basename(filepath)}', file=f)
    print(ds.head(5), file=f)

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

def out_project(filename: str,
                dirpath:str,
                label_col: str,
                text_col: str,
                alpha: float,
                test_size: float,
                max_df: float,
                min_df: int,
                max_features: int,
                bin_labels: bool,
                random_seed: bool,
                optimize_dictionary: bool,
                pred_on_ds: bool,
                bin_beta_alpha: int,
                bin_beta_beta: int,
                verbose: bool,
                **kwargs):
    ds = load_data(filepath=os.path.join(dirpath, filename))
    print(f'\nDescription of {filename}:', file=f)
    print(ds.describe(), file=f)
    ds = preprocess_data(ds=ds,
                         label_col=label_col,
                         text_col=text_col,
                         bin_labels=bin_labels)
    vectorizer = VectorDictionary(docs=ds[text_col],
                                  max_features=max_features,
                                  max_df=max_df,
                                  min_df=min_df,
                                  optimize_dictionary=optimize_dictionary,
                                  verbose=verbose)
    train_ds, test_ds = train_test_split(ds,
                                         test_size=test_size,
                                         random_state=random_seed,
                                         stratify=ds[label_col])
    print(train_ds)
    if pred_on_ds:
        classifier = BayesClassifier(**vectorizer.vectorizer_params,
                                 verbose=verbose,
                                 alpha=alpha)

        classifier.fit(docs=train_ds[text_col],
                       labels=train_ds[label_col],
                       vectorizer=vectorizer)
        results = classifier.evaluate(docs=test_ds[text_col],
                                      labels=test_ds[label_col])
        for key in results:
            print(f'\n{key}:', file=f)
            print(results[key], file=f)
    else:
        classifier = BayesBinomBetaClassifier(alpha=bin_beta_alpha,
                                              beta_prior=bin_beta_beta,
                                              random_seed=random_seed,
                                              verbose=verbose)
        classifier.fit(train_ds[label_col].values)
        
        print(f'\nModel information:', file=f)
        print(f'Posterior Alpha: {classifier.posterior_params["alpha"]}', file=f)
        print(f'Posterior Beta: {classifier.posterior_params["beta"]}', file=f)

        # Predict
        n_trials = len(test_ds)
        expected_successes = classifier.predict(n_trials)

        print(f'\nExpected number of successes in test set: {expected_successes}', file=f)

        # Posterior sampling and analysis
        posterior_samples = classifier.sample_posterior(size=1000)
        print(f'Posterior Mean of p: {np.mean(posterior_samples):.4f}', file=f)
    print('Done!')

if __name__ == '__main__':
    args = gen_parser()
    log_dir = os.path.join(os.getcwd(), 'Project/logs')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log_file_path = os.path.join(log_dir, 'output.log')
    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w') as f:
            f.write('')

    with open(os.path.join(log_dir, 'output.log'), 'w+') as f:
        out_project(filename=args.filename,
                    dirpath=args.dirpath,
                    label_col=args.label_col,
                    text_col=args.text_col,
                    test_size=args.test_size,
                    bin_labels=args.bin_labels,
                    random_seed=args.random_seed,
                    optimize_dictionary=args.optimize_dictionary,
                    max_features=args.max_features,
                    max_df=args.max_df,
                    min_df=args.min_df,
                    alpha=args.alpha,
                    pred_on_ds=args.predict_on_ds,
                    bin_beta_alpha=args.bb_alpha,
                    bin_beta_beta=args.bb_beta,
                    verbose=args.verbose)
        