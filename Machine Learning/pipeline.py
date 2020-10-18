# http://maruo51.com/2020/06/04/pipeline/

#-------------
# モデルごとに異なる前処理
#-------------

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
 
# データの準備
dataset = load_breast_cancer()
x_train, x_val, t_train, t_val = train_test_split(dataset.data, dataset.target, random_state=2020)

pipe = Pipeline([("preprocessing", StandardScaler()), ("classifier", SVC())])
 
param_grid = [
    {
        "classifier": [SVC()], 
        "preprocessing": [MinMaxScaler()],
        "classifier__gamma": [0.001, 0.01, 0.1, 1, 10, 100],
        "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100]
    },
    {
        "classifier": [LogisticRegression()],
        "preprocessing": [StandardScaler()],
        "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100]
    },
    {
        "classifier": [RandomForestClassifier()],
        "preprocessing": [None],
        "classifier__max_depth": [2, 3, 4, 5, 6],
        "classifier__n_estimators": [50, 100, 150, 200]
    }
]

grid = GridSearchCV(pipe, param_grid, cv=3)
grid.fit(x_train, t_train)

result = pd.DataFrame(grid.cv_results_)
result.sort_values("rank_test_score").loc[:, ["param_classifier", "param_preprocessing", "mean_test_score", "rank_test_score"]]


#-------------
# 前処理段階のハイパーパラメータ調整
#-------------

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, RobustScaler
from sklearn.datasets import load_boston

# データの準備
dataset = load_boston()
x_train, x_val, t_train, t_val = train_test_split(dataset.data, dataset.target, random_state=2020)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures()),
    ("regressor", Ridge())
])

param_grid = {
        "poly__degree": [1, 2, 3],
        "regressor__alpha": [0.001, 0.01, 0.1, 1, 10, 100]
}

grid = GridSearchCV(pipe, param_grid, cv=3)
grid.fit(x_train, t_train)

result = pd.DataFrame(grid.cv_results_)
result.sort_values("rank_test_score").loc[:, ["param_poly__degree", "param_regressor__alpha", "param_scaler", "mean_test_score"]]