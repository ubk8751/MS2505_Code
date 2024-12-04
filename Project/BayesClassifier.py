import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report


class BayesClassifier:
    def __init__(self,
                 max_features: int = None,
                 alpha: float = 1.0,
                 max_df: float = 1.0,
                 min_df: int = 1,
                 verbose: int = 0,
                 **kwargs):
        '''
        Initialize the BayesClassifier.

        :param max_features: Maximum number of features for the vectorizer.
        :param alpha: Smoothing parameter for Laplace smoothing.
        :param max_df: Max document frequency for the vectorizer.
        :param min_df: Min document frequency for the vectorizer.
        :param verbose: Verbosity level for debugging and messages.
        '''
        self._alpha = alpha
        self._max_features = max_features
        self._max_df = max_df
        self._min_df = min_df
        self._verbose = verbose
        self._vectorizer = None
        self._class_priors = {}
        self._word_probs = {}

    def fit(self, docs: pd.DataFrame, labels: pd.Series, vectorizer: CountVectorizer = None, **kwargs):
        '''
        Train the Bayesian classifier.

        :param docs: The documents to train on.
        :param labels: Corresponding labels for the documents.
        :param vectorizer: Vectorizer to use for feature extraction.
        '''
        self._vectorizer = vectorizer or CountVectorizer(
            max_features=self._max_features,
            max_df=self._max_df,
            min_df=self._min_df,
            **kwargs
        )
        if type(self._vectorizer) == CountVectorizer: 
            X_vectorized = self._vectorizer.fit_transform(docs).toarray()
        else:
            X_vectorized = self._vectorizer.fit_transform(docs)
        self._vocabulary = self._vectorizer.get_feature_names_out()

        classes, class_counts = np.unique(labels, return_counts=True)
        total_samples = len(labels)
        self._class_priors = {cls: class_counts[i] / total_samples for i, cls in enumerate(classes)}

        self._word_probs = {}
        for cls in classes:
            class_docs = X_vectorized[labels == cls]
            word_counts = np.sum(class_docs, axis=0)
            total_words = np.sum(word_counts)
            self._word_probs[cls] = (word_counts + self._alpha) / (total_words + self._alpha * len(self._vocabulary))

    def predict(self, docs: pd.DataFrame = None, **kwargs) -> np.array:
        '''
        Predict class labels for input documents.

        :param docs: The documents to predict labels for.
        :return: The predictions as a NumPy array.
        '''
        if not self._vectorizer:
            raise ValueError('The model is not trained yet. Please call `fit` first.')

        X_vectorized = self._vectorizer.transform(docs)
        predictions = []

        for doc_vector in X_vectorized:
            doc_dense = doc_vector

            class_scores = {}
            for cls in self._class_priors:
                score = np.log(self._class_priors[cls])
                for word_idx, count in enumerate(doc_dense):
                    if count > 0:
                        try:
                            score += count * np.log(self._word_probs[cls][word_idx])
                        except IndexError:
                            if self._verbose == 1:
                                print(f'Index {word_idx} out of range for class {cls}. Skipping.')
                        except KeyError:
                            if self._verbose == 1:
                                print(f'Word index {word_idx} not found in probabilities for class {cls}. Skipping.')

                class_scores[cls] = score
            predictions.append(max(class_scores, key=class_scores.get))

        return np.array(predictions)

    def evaluate(self, docs: pd.DataFrame, labels: pd.Series) -> dict:
        '''
        Evaluate the classifier's performance on a given dataset.

        :param docs: The documents to evaluate on.
        :param labels: The true labels corresponding to the documents.
        :return: A dictionary containing accuracy and classification report.
        '''
        predictions = self.predict(docs)
        accuracy = accuracy_score(labels, predictions)
        report = classification_report(labels, predictions, zero_division=1)
        
        if self._verbose:
            print('\nEvaluation Results:')
            print(f'Accuracy: {accuracy:.2%}')
            print('\nClassification Report:')
            print(report)

        return {
            'accuracy': accuracy,
            'classification_report': report
        }
    
    # Properties
    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, new_alpha: float):
        self._alpha = new_alpha

    @property
    def max_features(self):
        return self._max_features

    @max_features.setter
    def max_features(self, new_max_features: int):
        self._max_features = new_max_features

    @property
    def max_df(self):
        return self._max_df

    @max_df.setter
    def max_df(self, new_max_df: float):
        self._max_df = new_max_df

    @property
    def min_df(self):
        return self._min_df

    @min_df.setter
    def min_df(self, new_min_df: int):
        self._min_df = new_min_df

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, verbosity: int):
        self._verbose = verbosity

    @property
    def vectorizer(self):
        return self._vectorizer

    @vectorizer.setter
    def vectorizer(self, new_vectorizer: CountVectorizer):
        self._vectorizer = new_vectorizer

    @property
    def class_priors(self):
        return self._class_priors

    @property
    def word_probs(self):
        return self._word_probs
