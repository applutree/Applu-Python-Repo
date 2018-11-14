# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 17:52:43 2017

@author: applu
"""

import os
import pandas as pd

HOUSING_PATH = "C:\Users\applu\Documents\datasets\housing"

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()
housing.head()
