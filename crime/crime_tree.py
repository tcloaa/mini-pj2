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
