"""
Program:     
Programmer:  Vanessa Hamlett
Date:        
Overview:    
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import learning_curve
from sklearn.ensemble import AdaBoostClassifier


def plot_learning_curves(model, name, X_train, y_train):
    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X_train,
        y_train,
        cv=5,
        n_jobs=5,
        # scoring= ,
        train_sizes=np.linspace(0.01, 1.0, 50)
    )

    # Get the means and standard deviations of the outputted data
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    # Plot learning curve
    plt.subplots(1, figsize=(7, 5))
    plt.plot(train_sizes, train_scores_mean, label='Training Error')
    plt.plot(train_sizes, test_scores_mean, label='Validation Error')
    plt.ylabel('RMSE', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    plt.title('Learning curves: {}'.format(name), fontsize=18, y=1.03)
    plt.show()


sys.exit()

data = np.loadtxt('glass.data', dtype=str)

# Cleaning data
d = []
for line in data:
    a = line.split(",")
    li = []
    for thing in a:
        li.append(float(thing))
    d.append(li)

da = np.array(d)
X = da[:, 1:10]
y = da[:, 10]

# split data into 80/20 for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# building a gradient booster with the best features
# gb = GradientBoostingClassifier(
# max_depth=1,
# n_estimators=10,
# learning_rate=1.5,
# warm_start=True,
# )
# print(gb.get_params())
tuned_params = [
    {'learning_rate': [.5],
     # 'loss': ['deviance', 'accuracy'],
     'max_depth': [1],
     # 'max_features': [None,1,2,3,4,5],
     # 'max_leaf_nodes': [None,1,2,3],
     # 'min_samples_leaf': [1,2,3],
     # 'n_estimators': [1,5,10,15,20,25],
     # 'warm_start': [False, True]
     }
]
# grid_clf = GridSearchCV(gb, tuned_params, scoring="precision")
# grid_clf.fit(X_train, y_train)
# print(grid_clf.best_params_)
############################## Classifier Section #####################################
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(
        max_depth=2,
        max_leaf_nodes=2,
        max_features=2,
        min_samples_leaf=1
    ),
    n_estimators=20,
    algorithm="SAMME.R",
    learning_rate=3
)

# plot the learning curve
plot_learning_curves(ada_clf, 'Adaboost', X_train, y_train)

ada_clf.fit(X_train, y_train)  # fit the data
y_pred = ada_clf.predict(X_test)  # get the predictions
print("AdaBoost:", accuracy_score(y_test, y_pred))  # print accuracy

# Bagging clf for comparison
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(
        max_depth=1,
        max_leaf_nodes=2,
        max_features=2,
        min_samples_leaf=1
    ),
    n_estimators=20,  # number of trees to train
    max_samples=.5,  # This is the percentage if a float, but if a number that is the amount of the set
    max_features=2,
    bootstrap=False,  # if left false, it would do pasting, not a lot of data, so needs to be False
    n_jobs=-1
)

# plot the learning curve
plot_learning_curves(bag_clf, 'Bagging', X_train, y_train)

bag_clf.fit(X_train, y_train)  # fit the data
y_pred = bag_clf.predict(X_test)  # predict the outputs for the test data
print("Bagging:", accuracy_score(y_test, y_pred))  # print accuracy

# gradient boost takes weak regressors
# use basic case for everything except for warm_start
gb = GradientBoostingClassifier(
    max_depth=1,
    learning_rate=.5,
    # warm_start=True, # overfits!
)

# plot the learning curve
plot_learning_curves(gb, 'GradientBoostingCLF', X_train, y_train)

gb.fit(X_train, y_train)  # fit the classifier
y_pred = gb.predict(X_test)  # get the predictions
print("GradientBoost:", accuracy_score(y_test, y_pred))  # print

