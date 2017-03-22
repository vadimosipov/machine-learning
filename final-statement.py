import datetime
import pandas
import numpy as np
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

features = pandas.read_csv('./features.csv', index_col='match_id')

ds_length = len(features)
features_count = features.count()

# Print name of column if it has omissions
print "###### The next features have omissions ######"
for (i, v) in enumerate(features_count):
    if v < ds_length:
        print features.columns.values[i]
print "##############################################"

features.fillna(0, inplace=True)

# Prepare data
y = features['radiant_win'].values.ravel()

# Remove future features
future_features = ['duration', 'radiant_win', 'tower_status_radiant', 'tower_status_dire', 'barracks_status_radiant',
                   'barracks_status_dire']
for feature in future_features:
    del features[feature]
X = features

cv = KFold(X.shape[0], n_folds=5, shuffle=True, random_state=241)

# Gradient Boosting for classification.
print "Gradient boosting"

for n_estimator in [10, 20, 30, 40, 50]:
    start_time = datetime.datetime.now()

    print 'The number of boosting stages to perform:', n_estimator
    scores = []
    for train_index, test_index in cv:
        X_train, X_test = X.values[train_index], X.values[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = GradientBoostingClassifier(n_estimators=n_estimator, verbose=False, random_state=241)
        clf.fit(X_train, y_train)
        scores.append(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))

    print 'Time elapsed:', datetime.datetime.now() - start_time
    print 'ROC-AUC score:', np.mean(scores)

print "##############################################"


# Logistic Regression classifier.
def set_up_classifier(X, y):
    # Looking for the best C
    grid = {'C': np.power(10.0, np.arange(-5, 6))}
    clf = LogisticRegression(penalty='l2')
    gs = GridSearchCV(clf, grid, scoring='roc_auc', cv=cv)
    gs.fit(X, y)

    best_c = gs.best_params_['C']
    print 'The best C: ', best_c

    start_time = datetime.datetime.now()
    clf = LogisticRegression(penalty='l2', C=best_c)
    score = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    print 'ROC-AUC score:', np.mean(score)
    print 'Time elapsed:', datetime.datetime.now() - start_time
    return clf


print 'Calculate metric'

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
set_up_classifier(X_scaled, y)

print "##############################################"
# Remove categorical features
categorical_features = ['lobby_type', 'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero',
                        'd3_hero', 'd4_hero', 'd5_hero']

print 'Remove categorical features and calculate metric'
for categorical_feature in categorical_features:
    del X[categorical_feature]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
set_up_classifier(X_scaled, y)

print "##############################################"
# Generate words bag
print 'Generate words bag and calculate metric'
features = pandas.read_csv('./features.csv', index_col='match_id')
heroes_features_headers = ['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero',
                           'd3_hero', 'd4_hero', 'd5_hero']

heroes = set()
for heroes_feature in heroes_features_headers:
    for hero in features[heroes_feature].unique():
        heroes.add(hero)

print "There are ", len(heroes), " unique heroes"


# words bag
def generate_word_bag(dataset):
    X_pick = np.zeros((dataset.shape[0], 112))

    for i, match_id in enumerate(dataset.index):
        for p in xrange(5):
            X_pick[i, dataset.ix[match_id, 'r%d_hero' % (p + 1)] - 1] = 1
            X_pick[i, dataset.ix[match_id, 'd%d_hero' % (p + 1)] - 1] = -1
    return X_pick


X_union = np.concatenate([X_scaled, generate_word_bag(features)], axis=1)

model = set_up_classifier(X_union, y)
model.fit(X_union, y)

print "##############################################"
print 'Test'
# Calculate probabilities
features_test = pandas.read_csv('./features_test.csv', index_col='match_id')
features_test.fillna(0, inplace=True)
heroes_ds = features_test[heroes_features_headers]

for feature in categorical_features:
    del features_test[feature]

X = np.concatenate([scaler.fit_transform(features_test), generate_word_bag(heroes_ds)], axis=1)

probabilities = model.predict_proba(X)[:, 1]
print "Max probability:", np.max(probabilities)
print "Min probability:", np.min(probabilities)
