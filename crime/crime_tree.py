import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pydot
# from IPython.display import Image

from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.externals.six import StringIO  
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error

from sklearn import preprocessing
pd.set_option('display.notebook_repr_html', False)

#%matplotlib inline
plt.style.use('seaborn-white')

def print_tree(estimator, features, class_names=None, filled=True):
    tree = estimator
    names = features
    color = filled
    classn = class_names
    
    dot_data = StringIO()
    export_graphviz(estimator, out_file=dot_data, feature_names=features, class_names=classn, filled=filled)
    (graph,) = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('1.png')
    return(graph)

df = pd.read_csv('crimedata.csv').dropna()
# df.info()
X = df[['rincpc','econgrow','unemp','citypop','a0_5','a5_9','a10_14','a15_19','a20_24','a25_29','citybla','cityfemh','sta_educ','sta_welf','price','sworn','civil','elecyear','governor','term2','term3','termlim','mayor']].as_matrix()
X = preprocessing.scale(X)
#Y = df[[murder,rape,robbery,assault,burglary,larceny,auto]].as_matrix()
y = preprocessing.scale(df[['larceny']].as_matrix())
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=0)
regr = DecisionTreeRegressor(max_leaf_nodes=8)
regr.fit(X_train, y_train)

graph = print_tree(regr, features=['rincpc','econgrow','unemp','citypop','a0_5','a5_9','a10_14','a15_19','a20_24','a25_29','citybla','cityfemh','sta_educ','sta_welf','price','sworn','civil','elecyear','governor','term2','term3','termlim','mayor'])

pred = regr.predict(X_test)
print mean_squared_error(y_test, pred)

print X.shape

# bagging using all features
regr1 = RandomForestRegressor(max_features=23, random_state=1)
regr1.fit(X_train, y_train.ravel())
pred1 = regr1.predict(X_test)
print mean_squared_error(y_test, pred1)

# less features
regr2 = RandomForestRegressor(max_features=10, random_state=1)
regr2.fit(X_train, y_train.ravel())
pred2 = regr2.predict(X_test)
print mean_squared_error(y_test, pred2)

# boosting
regr3 = GradientBoostingRegressor(n_estimators=500, learning_rate=0.01, random_state=1)
regr3.fit(X_train, y_train.ravel())
print mean_squared_error(y_test, regr3.predict(X_test))
Importance = pd.DataFrame({'Importance':regr1.feature_importances_*100}, index=['rincpc','econgrow','unemp','citypop','a0_5','a5_9','a10_14','a15_19','a20_24','a25_29','citybla','cityfemh','sta_educ','sta_welf','price','sworn','civil','elecyear','governor','term2','term3','termlim','mayor'])

Importance.sort_values('Importance', axis=0, ascending=True).plot(kind='barh', color='r', )
plt.xlabel('Variable Importance')
plt.gca().legend_ = None
plt.savefig("importance.png")
