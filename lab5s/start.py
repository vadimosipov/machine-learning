import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

%matplotlib inline


def to_datetime(time):
    rpc_time = time.replace('-', '.')
    formatted_time = rpc_time + ':00' if rpc_time.count(':') == 1 else rpc_time
    return datetime.strptime(formatted_time, '%d.%m.%Y %H:%M:%S')


def get_idx_by_name(name, data):
    idx = 0
    for col in data.columns:
        if name == col:
            break
        idx += 1
    return idx


def find_wrong_ids(data, chk_field, idx_field):
    ids = []
    mark = False
    for row in data.values:
        if mark:
            ids.append(row[idx_field])
        elif not row[chk_field]:
            mark = True
    return ids


def cleanout(data):
    columns = data.columns.tolist()
    id_idx = columns.index('id')
    valid_idx = columns.index('valid')
    indeces = []
    for _, group in data.groupby('e-mail'):
        if False in group['valid'].unique():
            sorted_group = group.sort_values('datetime')
            indeces += find_wrong_ids(sorted_group, valid_idx, id_idx)

    return data.drop(data.index[indeces])


def split_time():
    return le.fit_transform(cleaned_data['time2'])


def prepare_data(data):
    data['id'] = [i for i in range(data.shape[0])]
    data['datetime'] = data.time.apply(to_datetime)
    cleaned_data = cleanout(data)

    le = LabelEncoder()
    X = []
    X.append(le.fit_transform(cleaned_data['e-mail']))
    X.append(le.fit_transform(cleaned_data['open']))
    X.append(le.fit_transform(cleaned_data['click']))
    y = le.fit_transform(cleaned_data['valid'])
    return X, y


data = pd.read_csv('/data/lab05s/emarsys_train.csv', delimiter=';')
X, y = prepare_data(data)

params = {'C': 10.0 ** np.arange(-4, 3, 0.5)}
model = GridSearchCV(LogisticRegression(), params)

cv = KFold(len(y), 10, True, 142)
scores = cross_val_score(model, X, y, 'roc_auc', cv)

