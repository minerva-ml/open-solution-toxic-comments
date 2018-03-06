import re
import string
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.feature_extraction import text
import sklearn.preprocessing as sk_prep
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from .base import BaseTransformer

lem = WordNetLemmatizer()
tokenizer=TweetTokenizer()
eng_stopwords = set(stopwords.words("english"))
APPO = {"aren't": 'are not', "can't": 'cannot', "couldn't": 'could not', "didn't": 'did not', "doesn't": 'does not', "don't": 'do not', "hadn't": 'had not', "hasn't": 'has not', "haven't": 'have not', "he'd": 'he would', "he'll": 'he will', "he's": 'he is', "i'd": 'I had', "i'll": 'I will', "i'm": 'I am', "isn't": 'is not', "it's": 'it is', "it'll": 'it will', "i've": 'I have', "let's": 'let us', "mightn't": 'might not', "mustn't": 'must not', "shan't": 'shall not', "she'd": 'she would', "she'll": 'she will', "she's": 'she is', "shouldn't": 'should not', "that's": 'that is', "there's": 'there is', "they'd": 'they would', "they'll": 'they will', "they're": 'they are', "they've": 'they have', "we'd": 'we would', "we're": 'we are', "weren't": 'were not', "we've": 'we have', "what'll": 'what will', "what're": 'what are', "what's": 'what is', "what've": 'what have', "where's": 'where is', "who'd": 'who would', "who'll": 'who will', "who're": 'who are', "who's": 'who is', "who've": 'who have', "won't": 'will not', "wouldn't": 'would not', "you'd": 'you would', "you'll": 'you will', "you're": 'you are', "you've": 'you have', "'re": ' are', "wasn't": 'was not', "we'll": ' will'}

class WordListFilter(BaseTransformer):
    def __init__(self, word_list_filepath):
        self.word_set = self._read_data(word_list_filepath)

    def transform(self, X):
        X = self._transform(X)
        return {'X': X}

    def _transform(self, X):
        X = pd.DataFrame(X, columns=['text']).astype(str)
        X['text'] = X['text'].apply(self._filter_words)
        return X['text'].values

    def _filter_words(self, x):
        x = x.lower()
        x = ' '.join([w for w in x.split() if w in self.word_set])
        return x

    def _read_data(self, filepath):
        with open(filepath, 'r+') as f:
            data = f.read()
        return set(data.split('\n'))

    def load(self, filepath):
        return self

    def save(self, filepath):
        joblib.dump({}, filepath)


class TextCleaner(BaseTransformer):
    def __init__(self, drop_punctuation, drop_newline, drop_multispaces,
                 all_lower_case, fill_na_with, deduplication_threshold, anonymize, apostophes):
        self.drop_punctuation = drop_punctuation
        self.drop_newline = drop_newline
        self.drop_multispaces = drop_multispaces
        self.all_lower_case = all_lower_case
        self.fill_na_with = fill_na_with
        self.deduplication_threshold = deduplication_threshold
        self.anonymize = anonymize
        self.apostophes = apostophes

    def transform(self, X):
        X = pd.DataFrame(X, columns=['text']).astype(str)
        X['text'] = X['text'].apply(self._transform)
        if self.fill_na_with:
            X['text'] = X['text'].fillna(self.fill_na_with).values
        return {'X': X['text'].values}

    def _transform(self, x):
        if self.all_lower_case:
            x = self._lower(x)
        if self.drop_punctuation:
            x = self._remove_punctuation(x)
        if self.drop_newline:
            x = self._remove_newline(x)
        if self.drop_multispaces:
            x = self._substitute_multiple_spaces(x)
        if self.deduplication_threshold is not None:
            x = self._deduplicate(x)
        if self.anonymize:
            x = self._anonymize(x)
        if self.apostophes:
            x = self._apostophes(x)
        return x

    def _apostophes(self,x):
        words=tokenizer.tokenize(x)
        words=[APPO[word] if word in APPO else word for word in words]
        words=[lem.lemmatize(word, "v") for word in words]
        words = [w for w in words if not w in eng_stopwords]
        clean_sent=" ".join(words)
        return(clean_sent)   

    def _anonymize(self,x):
        # remove leaky elements like ip,user
        x=re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"," ",x)
        #removing usernames
        x=re.sub("\[\[.*\]"," ",x)        
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

    def _deduplicate(self,x):
        word_list = x.split()
        num_words = len(word_list)
        if num_words ==0:
            return x
        else:
            num_unique_words = len(set(word_list))
            unique_ratio = num_words/num_unique_words
            if unique_ratio > self.deduplication_threshold:
                x = ' '.join(x.split()[:num_unique_words])
            return x

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
        X['caps_vs_length'] = X.apply(lambda row: float(row['upper_case_count']) / float(row['char_count']), axis=1)
        X['num_symbols'] = X['text'].apply(lambda comment: sum(comment.count(w) for w in '*&$%'))
        X['num_words'] = X['text'].apply(lambda comment: len(comment.split()))
        X['num_unique_words'] = X['text'].apply(lambda comment: len(set(w for w in comment.split())))
        X['words_vs_unique'] = X['num_unique_words'] / X['num_words']
        X['mean_word_len'] = X['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
        X.drop('text', axis=1, inplace=True)
        X.fillna(0.0, inplace=True)
        return {'X': X}

    def _transform(self, x):
        features = {}
        features['text'] = x
        features['char_count'] = char_count(x)
        features['word_count'] = word_count(x)
        features['punctuation_count'] = punctuation_count(x)
        features['upper_case_count'] = upper_case_count(x)
        features['lower_case_count'] = lower_case_count(x)
        features['digit_count'] = digit_count(x)
        features['space_count'] = space_count(x)
        features['newline_count'] = newline_count(x)
        return pd.Series(features)

    def load(self, filepath):
        return self

    def save(self, filepath):
        joblib.dump({}, filepath)


class Normalizer(BaseTransformer):
    def __init__(self):
        self.normalizer = sk_prep.Normalizer()

    def fit(self, X):
        self.normalizer.fit(X)
        return self

    def transform(self, X):
        X = self.normalizer.transform(X)
        return {'X': X}

    def load(self, filepath):
        self.normalizer = joblib.load(filepath)
        return self

    def save(self, filepath):
        joblib.dump(self.normalizer, filepath)


def char_count(x):
    return len(x)


def word_count(x):
    return len(x.split())


def newline_count(x):
    return x.count('\n')


def upper_case_count(x):
    return sum(c.isupper() for c in x)


def lower_case_count(x):
    return sum(c.islower() for c in x)


def digit_count(x):
    return sum(c.isdigit() for c in x)


def space_count(x):
    return sum(c.isspace() for c in x)


def punctuation_count(x):
    return occurence(x, string.punctuation)


def occurence(s1, s2):
    return sum([1 for x in s1 if x in s2])
