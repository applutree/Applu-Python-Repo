# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 21:32:43 2017

@author: applu
"""

# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

# Common imports
import numpy as np
import os
import pandas as pd
#import datetime

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
#%matplotlib inline
#import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "noshow"

def image_path(fig_id):
    return os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id)

def save_fig(fig_id, tight_layout=True):
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(image_path(fig_id) + ".png", format='png', dpi=300)

NOSHOW_PATH = os.path.join("datasets", "noshow")

def load_noshow_data(noshow_path=NOSHOW_PATH):
    csv_path = os.path.join(noshow_path, "no_show_2016.csv")
    return pd.read_csv(csv_path)

noshow = load_noshow_data()

y = noshow["No-show"].copy()
X = pd.DataFrame(noshow, columns=noshow.columns, index = list(noshow.index.values))
X = noshow.drop("PatientId", axis=1)
X = X.drop("No-show", axis=1)
X['ScheduledDay'] = pd.to_datetime(X['ScheduledDay'])
X['AppointmentDay'] = pd.to_datetime(X['AppointmentDay'])
X['AppointmentDay_of_Week'] = X['AppointmentDay'].dt.weekday
X['ScheduledDay_of_Week'] = X['ScheduledDay'].dt.weekday
X['ScheduledDay_hour'] = X['ScheduledDay'].dt.hour
X = X.drop("ScheduledDay", axis=1)
X = X.drop("AppointmentDay", axis=1)
X = X.drop("AppointmentID", axis=1)
X = pd.get_dummies(X)

tree_clf_d2 = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf_d2.fit(X, y)

tree_clf_d3 = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_clf_d3.fit(X, y)

tree_clf_d4 = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_clf_d4.fit(X, y)

export_graphviz(
        tree_clf_d2,
        out_file=image_path("noshow_tree_d2_v1.dot"),
        feature_names=list(X),
        class_names=y,
        rounded=True,
        filled=True
    )

export_graphviz(
        tree_clf_d3,
        out_file=image_path("noshow_tree_d3_v1.dot"),
        feature_names=list(X),
        class_names=y,
        rounded=True,
        filled=True
    )

export_graphviz(
        tree_clf_d4,
        out_file=image_path("noshow_tree_d4_v1.dot"),
        feature_names=list(X),
        class_names=y,
        rounded=True,
        filled=True
    )

y_pred_d2 = tree_clf_d2.predict(X_test)