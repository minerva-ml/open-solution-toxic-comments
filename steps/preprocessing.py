import re
import string

import pandas as pd
from sklearn.externals import joblib
from sklearn.feature_extraction import text

from .base import BaseTransformer


class BadWordCountVectorizer(BaseTransformer):
    def __init__(self, bad_word_filepath):
        self.bad_word_filepath = bad_word_filepath
        self.bad_words = joblib.load(bad_word_filepath)
        self.vectorizer = text.CountVectorizer()
    
    def fit(self, X):
        X = self._prepare(X)
        self.vectorizer.fit(X)
        return self
        
    def transform(self, X):
        X = self._prepare(X)
        X = self.vectorizer.transform(X)
        return {'X': X}
    
    def _prepare(self, X):
        X = pd.DataFrame(X, columns=['text']).astype(str)
        X['text'] = X['text'].apply(self._filter_words)
        return X['text'].values
    
    def _filter_words(self, x):
        x = x.lower()
        x = ' '.join([w for w in x.split() if w in self.bad_words])
        return x

    
    def load(self, filepath):
        params = joblib.load(filepath)
        self.bad_words = params['bad_words']
        self.vectorizer = params['vectorizer']
        return self

    def save(self, filepath):
        params = {'bad_words': self.bad_words,
                  'vectorizer': self.vectorizer
                  }
        joblib.dump(params, filepath)
    
    
class BadWordCounter(BaseTransformer):
    def __init__(self, bad_word_filepath):
        self.bad_word_filepath = bad_word_filepath
        self.bad_words = joblib.load(bad_word_filepath)
        
    def transform(self, X):
        X = self._transform(X)
        return {'X': X}
    
    def _transform(self, X):
        X = pd.DataFrame(X, columns=['text']).astype(str)
        X['text'] = X['text'].apply(self._count_bad_words)
        return X['text'].values
    
    def _count_bad_words(self, x):
        x = x.lower()
        x = [w for w in x.split() if w in self.bad_words]
        return len(x)
    
    def load(self, filepath):
        return self

    def save(self, filepath):
        joblib.dump({}, filepath)
    
class TextCleaner(BaseTransformer):
    def __init__(self, drop_punctuation, all_lower_case, fill_na_with):
        self.drop_punctuation = drop_punctuation
        self.all_lower_case = all_lower_case
        self.fill_na_with = fill_na_with

    def transform(self, X):
        X = pd.DataFrame(X, columns=['text']).astype(str)
        X['text'] = X['text'].apply(self._transform)
        X['text'] = X['text'].fillna(self.fill_na_with).values
        return {'X': X['text'].values}

    def _transform(self, x):
        x = self._lower(x)
        x = self._remove_punctuation(x)
        x = self._remove_newline(x)
        x = self._substitute_multiple_spaces(x)
        return x

    def _lower(self, x):
        return x.lower()

    def _remove_punctuation(self, x):
        return re.sub(r'[^\w\s]', ' ', x)

    def _remove_newline(self, x):
        x = x.replace('\n', ' ')
        x = x.replace('\n\n', ' ')
        return x
    
    def _substitute_multiple_spaces(self, x):
        return ' '.join(x.split())

    def load(self, filepath):
        params = joblib.load(filepath)
        self.drop_punctuation = params['drop_punctuation']
        self.all_lower_case = params['all_lower_case']
        self.fill_na_with = params['fill_na_with']
        return self

    def save(self, filepath):
        params = {'drop_punctuation': self.drop_punctuation,
                  'all_lower_case': self.all_lower_case,
                  'fill_na_with': self.fill_na_with,
                  }
        joblib.dump(params, filepath)


class XYSplit(BaseTransformer):
    def __init__(self, x_columns, y_columns):
        self.x_columns = x_columns
        self.y_columns = y_columns

    def transform(self, meta, train_mode):
        X = meta[self.x_columns].values
        if train_mode:
            y = meta[self.y_columns].values
        else:
            y = None

        return {'X': X,
                'y': y}

    def load(self, filepath):
        params = joblib.load(filepath)
        self.columns_to_get = params['x_columns']
        self.target_columns = params['y_columns']
        return self

    def save(self, filepath):
        params = {'x_columns': self.x_columns,
                  'y_columns': self.y_columns
                  }
        joblib.dump(params, filepath)


class TfidfVectorizer(BaseTransformer):
    def __init__(self, **kwargs):
        self.vectorizer = text.TfidfVectorizer(**kwargs)

    def fit(self, text):
        self.vectorizer.fit(text)
        return self

    def transform(self, text):
        return {'features': self.vectorizer.transform(text)}

    def load(self, filepath):
        self.vectorizer = joblib.load(filepath)
        return self

    def save(self, filepath):
        joblib.dump(self.vectorizer, filepath)
        

class TextCounter(BaseTransformer):   
    def transform(self, X):
        X = pd.DataFrame(X, columns=['text']).astype(str)
        X = X['text'].apply(self._transform)
        return {'X': X}
    
    def _transform(self, x):
        features = {}
        features['char_count'] = char_count(x)
        features['word_count'] = word_count(x)
        features['punctuation_count'] = punctuation_count(x)
        features['upper_case_count'] = upper_case_count(x)
        features['lower_case_count'] = lower_case_count(x)
        features['digit_count'] = digit_count(x)
        features['space_count'] = space_count(x)
        return pd.Series(features)

    def load(self, filepath):
        return self

    def save(self, filepath):
        joblib.dump({}, filepath)
    
def char_count(x):
    return len(x)

def word_count(x):
    return len(x.split())

def upper_case_count(x):
    return sum(c.isupper() for c in x)

def lower_case_count(x):
    return sum(c.islower() for c in x)

def digit_count(x):
    return sum(c.isdigit() for c in x)

def space_count(x):
    return sum(c.isspace() for c in x)

def punctuation_count(x):
    return  occurence(x, string.punctuation)

def occurence(s1, s2):
    return sum([1 for x in s1 if x in s2])
