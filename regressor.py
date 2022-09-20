import matplotlib.pyplot as plt
import statsmodels.api as sm

from statistics import mean

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, plot_confusion_matrix
from sklearn.preprocessing import LabelEncoder


def do_regression_sklearn(coin_df, label, symbol):
    """ regression using scikit learn """
    coin_df.dropna(inplace=True)
    X = coin_df['sentiment'].values
    y = coin_df[label].values
    le = LabelEncoder()
    X = le.fit_transform(X)
    le_name_mapping_X = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f'Sentiment mapping: {le_name_mapping_X}')
    y = le.fit_transform(y)
    le_name_mapping_y = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f'{label} mapping: {le_name_mapping_y}')
    labels = {v: k for k, v in le_name_mapping_y.items()}

    cv = KFold(n_splits=5, shuffle=True, random_state=1)
    bacc = []
    f1 = []
    for train, test in cv.split(X, y):
        x_train, y_train = X[train], y[train]
        x_test, y_test = X[test], y[test]

        x_train = x_train.reshape(-1, 1)
        x_test = x_test.reshape(-1, 1)

        clf = LogisticRegression(random_state=0)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        bacc.append(balanced_accuracy_score(y_test, y_pred))
        f1.append(f1_score(y_test, y_pred, average='macro'))

    print('Classifier accuracy')
    print(F'coin symbol: {symbol}\npredicting: {label}')
    print(f'Mean Balanced accuracy: {mean(bacc)}')
    print(f'Mean F1 score: {mean(f1)}')
    plot_confusion_matrix(clf, x_test, y_test)
    plt.title(f'Confusion matrix, {symbol} {label}')
    plt.show()


def do_regression(coin_df, label, symbol, lag):
    """ regression using statsmodels"""
    coin_df.dropna(inplace=True)
    X = coin_df['sentiment'].values
    y = coin_df[label].values
    # encode columns to categorical
    le = LabelEncoder()
    X = le.fit_transform(X)
    le_name_mapping_X = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f'Sentiment mapping: {le_name_mapping_X}')
    y = le.fit_transform(y)
    le_name_mapping_y = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f'{label} mapping: {le_name_mapping_y}')

    mod = sm.Logit(y, X)
    fii = mod.fit()
    print(fii.summary(title=F'Logit Regression Results {symbol}'))
    print('#' * 100)
    if lag:
        filename = f'results_logit_{symbol}_lag.txt'
    else:
        filename = f'results_logit_{symbol}.txt'
    with open(filename, 'w')as f_out:
        f_out.write(str(fii.summary(title=F'Logit Regression Results {symbol}\n')))


def le_regress(data, lag=False):
    """works as main for regression (to be called from main.py)"""
    symbols = ['BTC', 'ETH', 'USDT']
    for symbol in symbols:
        coin_df = data.loc[:, ['sentiment', f'{symbol}_vol_direction']]
        # do_regression_sklearn(coin_df, f'{symbol}_vol_direction', symbol)
        do_regression(coin_df, f'{symbol}_vol_direction', symbol, lag)
