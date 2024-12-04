import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from itertools import product


class VectorDictionary:
    def __init__(self,
                 docs: pd.DataFrame = None,
                 verbose: int = 0,
                 optimize_dictionary: bool = True,
                 max_features: int = None,
                 max_df: float = 1.0,
                 min_df: int = 1,
                 **kwargs):
        """
        Initialize the VectorDictionary class.

        :param docs: Input documents as a DataFrame.
        :param verbose: Verbosity level for optimization.
        :param optimize_dictionary: Whether to optimize the dictionary.
        :param max_features: Maximum number of features for the vectorizer.
        :param max_df: Max document frequency for the vectorizer.
        :param min_df: Min document frequency for the vectorizer.
        """
        self._docs = docs
        self._verbose = verbose
        self._vectorizer_params = {
            'max_features': max_features,
            'max_df': max_df,
            'min_df': min_df
        }
        self._vectorizer = None
        self._vocabulary = None

        # Optimize parameters if needed
        if optimize_dictionary and docs is not None:
            self._vectorizer_params = self._optimize(docs=docs)

        # Initialize the vectorizer
        self._initialize_vectorizer()

    def _initialize_vectorizer(self, **kwargs):
        """
        Initialize the CountVectorizer with the optimized or provided parameters.
        """
        self._vectorizer = CountVectorizer(
            max_features=self._vectorizer_params.get('max_features'),
            max_df=self._vectorizer_params.get('max_df'),
            min_df=self._vectorizer_params.get('min_df'),
            **kwargs
        )

    def _optimize(self, docs: pd.DataFrame) -> dict:
        """
        Optimize vectorizer parameters manually without scoring.

        :param docs: Input documents as a DataFrame.
        :return: Best parameters for the vectorizer.
        """
        if self._verbose:
            print("Optimizing vectorizer parameters...")

        param_grid = {
            'max_features': [1000, 3000, 5000, 8000, 9000],
            'min_df': [1, 5, 10],
            'max_df': [0.5, 0.8, 1.0],
        }

        best_params = None
        best_vocab_size = 0

        for max_features, min_df, max_df in product(
                param_grid['max_features'], param_grid['min_df'], param_grid['max_df']):
            vectorizer = CountVectorizer(max_features=max_features, min_df=min_df, max_df=max_df)
            vectorizer.fit(docs)
            vocab_size = len(vectorizer.vocabulary_)

            if self._verbose:
                print(f"Params: max_features={max_features}, min_df={min_df}, max_df={max_df} -> Vocab size: {vocab_size}")

            if vocab_size > best_vocab_size:
                best_vocab_size = vocab_size
                best_params = {'max_features': max_features, 'min_df': min_df, 'max_df': max_df}

        if self._verbose:
            print("Optimization completed.")
            print("Best Parameters:", best_params)

        return best_params

    def fit_transform(self, docs: pd.DataFrame) -> np.array:
        """
        Fit the vectorizer on the documents and transform them into a document-term matrix.

        :param docs: Input documents as a DataFrame.
        :return: Numpy array representing the document-term matrix.
        """
        if self._vectorizer is None:
            raise ValueError("Vectorizer is not initialized. Please initialize the vectorizer first.")

        matrix = self._vectorizer.fit_transform(docs).toarray()
        self._vocabulary = self._vectorizer.get_feature_names_out()
        if self._verbose:
            print(f"Vocabulary created with {len(self._vocabulary)} tokens.")
        return matrix

    def transform(self, docs: pd.DataFrame) -> np.array:
        """
        Transform the documents using the fitted vectorizer.

        :param docs: Input documents as a DataFrame.
        :return: Numpy array representing the document-term matrix.
        """
        if self._vectorizer is None:
            raise ValueError("Vectorizer is not initialized. Please fit the vectorizer first.")

        return self._vectorizer.transform(docs).toarray()

    def get_feature_names_out(self):
        return self._vectorizer.get_feature_names_out()
    # Properties
    @property
    def docs(self):
        return self._docs

    @docs.setter
    def docs(self, new_docs: pd.DataFrame):
        self._docs = new_docs

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, verbosity: int):
        self._verbose = verbosity

    @property
    def vectorizer_params(self):
        return self._vectorizer_params

    @vectorizer_params.setter
    def vectorizer_params(self, new_params: dict):
        self._vectorizer_params = new_params
        self._initialize_vectorizer()

    @property
    def token_dictionary(self):
        return self._token_dictionary

    @property
    def vectorizer(self):
        return self._vectorizer
    
    @property
    def vocab(self):
        return self._vocab

