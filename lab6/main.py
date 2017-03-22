# -*- coding: utf-8 -*-
import io
import json
import re
from os.path import join
from urlparse import urlparse

import numpy as np
import pandas as pd
from nltk.tokenize import LineTokenizer
from nltk.tokenize import RegexpTokenizer
from sklearn.cross_validation import ShuffleSplit, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


dom_reg_2 = re.compile(r'^www\.', re.U)
path_tokenizer_2 = RegexpTokenizer(u'[A-Za-z\.0-9]+')
query_reg_2 = re.compile(u'(\?|&)[a-zA-Z_]+=([ a-zA-Zа-яА-Я0-9\.:\/\-=_+]+)', re.U)
RE_DIGITAL = re.compile(r'\d', re.U)
RE_HTTP = re.compile(r'http://http://', re.U)


def prepare_data(data):
    del data['user_json']
    data.columns = [u'gender', u'uid', u'user_json']

    train_data = data.drop(data[data['gender'] == '-'].index)
    del train_data['uid']
    le = LabelEncoder()
    train_data['gender'] = le.fit_transform(train_data['gender'])

    test_data = data.drop(data[data['gender'] != '-'].index)
    test_data.sort_values('uid', inplace=True)
    del test_data['gender']
    del test_data['uid']

    return train_data, test_data


def read_file(file_path):
    with io.open(file_path, encoding='utf-8') as ins:
        return ins.read()


def load_stop_words():
    STOP_WORDS_PATH = join('lab06', 'sw.txt')
    content = read_file(STOP_WORDS_PATH)
    lt = LineTokenizer()
    return lt.tokenize(content)


def no_digits(string):
    return RE_DIGITAL.search(string) is None


def parse_url(url):
    url = RE_HTTP.sub('http://', url)
    parsed = urlparse(url)

    domains = [dom_reg_2.sub('', parsed.netloc)]
    paths = [token
             for token in path_tokenizer_2.tokenize(parsed.path)
             if len(token) > 2 and token.isalpha()]
    query = [param.strip().split() for sym, param in query_reg_2.findall(parsed.query)
             if len(param) > 2 and no_digits(param)]
    queries = []
    if len(query) != 0:
        for param in sum(query, []):
            if u'http' in param:
                d, p, q = parse_url(param)
                domains.extend(d)
                paths.extend(p)
                queries.extend(q)
            else:
                queries.append(param)
    return domains, paths, queries

def handle_query(query, domains, queries):
    if 'www' in query:
        domains.append(dom_reg_2.sub('', query))
    else:
        queries.append(query)

def extract_user_data2(user_json):
    domains = []
    paths = []
    queries = []

    udata = json.loads(user_json)
    for info in udata['visits']:
        d, p, q = parse_url(info['url'])
        if len(d) != 0: domains.extend(d)
        if len(p) != 0: paths.extend(p)
        if len(q) != 0: queries.extend(q)

    new_queries = []
    for query in queries:
        if '/' in query:
            for sub_query in qs.split('/'):
                handle_query(sub_query, domains, new_queries)
        else:
            handle_query(query, domains, new_queries)

    return domains, paths, new_queries

def extract_user_time(user_json):
    domains = []
    timestamps = []
    udata = json.loads(user_json)
    for info in udata['visits']:
        parsed = urlparse(info['url'])

        domain = dom_reg_2.sub('', parsed.netloc)
        domains.append(domain)

        timestamp = info['timestamp']
        timestamps.append(timestamp)
    return domains, timestamps

def extract_by_gender(gender_group):
    d = []
    p = []
    q = []
    i = 0
    for user_json in gender_group:
        domains, paths, queries = extract_user_data2(user_json)
        d.extend(domains)
        p.extend(paths)
        q.extend(queries)
    return d, p, q

def tokenize2(user_json):
    domains, paths, queries = extract_user_data2(user_json)
    result = []
    result.extend(domains)
    result.extend(paths)
    result.extend(queries)
    return result


def train(model, X, y):
#     cv = ShuffleSplit(X.shape[0], test_size=0.3)
    cv = KFold(X.shape[0], n_folds = 2, shuffle = True)
    grid = GridSearchCV(model[0], model[1], 'roc_auc', cv=cv, n_jobs=4, verbose=1)
    grid.fit(X, y)
    return grid


def log_model():
    return LogisticRegression(), {'C': 10.0 ** np.arange(-3, 5, 0.5), 'penalty': ['l1', 'l2']}


def tree_model():
    return DecisionTreeClassifier(), {'max_depth': [10, 20, 30], 'criterion': ['gini', 'entropy']}


def forest_model():
    return RandomForestClassifier(), {'n_estimators': [10, 20, 30], 'criterion': ['gini', 'entropy']}


def svc_model():
    return SVC(), [{'kernel': ['rbf'], 'C': np.logspace(-5, 2, 25), 'gamma': 10.0 ** np.arange(-4, 1, 1)},
                   {'kernel': ['linear'], 'C': np.logspace(-5, 2, 25)}]

def naive_baess():
    return MultinomialNB(), {'alpha': [0, 0.000625]}
#     , 0.00125, 0.0025, 0.005, 0.01


def best_features(coeff):
    indeces = np.arange(0, len(coeff))
    coef_log = zip(indeces, coeff)
    coef_log_sorted = sorted(coef_log, key=lambda x: x[1], reverse=True)
    pos_coef_log = map(lambda key: key[0], filter(lambda key: key[1] > 0, coef_log_sorted)[:1000])
    return pos_coef_log



data = pd.read_csv('/data/lab05/gender_dataset.txt', delimiter='\t')
train_data, test_data = prepare_data(data)
y = train_data['gender']

male = train_data.drop(train_data[train_data['gender'] == 0].index)
female = train_data.drop(train_data[train_data['gender'] == 1].index)

male_domains, male_paths, male_queries = extract_by_gender(male['user_json'])
female_domains, female_paths, female_queries = extract_by_gender(female['user_json'])

print len(male_domains), len(male_paths), len(male_queries)
print len(female_domains), len(female_paths), len(female_queries)

mds = set(male_domains)
mps = set(male_paths)
mqs = set(male_queries)

fds = set(female_domains)
fps = set(female_paths)
fqs = set(female_queries)

print '---------------------------'
print len(mds), len(mps), len(mqs)
print len(fds), len(fps), len(fqs)


sw = set(load_stop_words())
# sw.update(mds.intersection(fds))
sw.update(mps.intersection(fps))
sw.update(mqs.intersection(fqs))
print len(sw)

tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize2, stop_words=sw, sublinear_tf=True, min_df=2)
tfidf_features = tfidf_vectorizer.fit_transform(train_data['user_json'])
tfidf_nbsmodel = train(naive_baess(), tfidf_features, y)
print tfidf_nbsmodel.best_score_, tfidf_nbsmodel.best_params_