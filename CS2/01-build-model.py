#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 09:15:43 2020

@author: emkaerr
"""

import pandas as pd
import numpy as np



np.random.seed(10)

df_raw = pd.read_csv('/home/emkaerr/PycharmProjects/test/CS2/dataset/data.csv',
                     index_col=0)
df_raw.head()

cols = ['Year', 'Fuel_Type', 'Transmission', 'Engine', 'Power', 'Seats', 'Price']
df = df_raw.copy()
df = df[cols]
df2 = df.copy()



df.Engine = df.Engine.str.split(' ').str[0].replace('null',np.nan)
df.Power = df.Power.str.split(' ').str[0].replace('null',np.nan)

df = df.dropna()

df.Engine = df.Engine.astype('float32')
df.Power = df.Power.astype('float32')

df=pd.get_dummies(df, drop_first=True)

df.to_csv('/home/emkaerr/PycharmProjects/test/CS2/dataset/datacleaned.csv')

X = df.copy()
y = X.pop('Price')

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor()
reg.fit(X_train, y_train)

reg_score = (reg.score(X_test, y_test))


from sklearn.model_selection import GridSearchCV

model = RandomForestRegressor()
param_grid = [{'max_depth':[3,4,5,6,7,8,10,20],
               'min_samples_leaf':[3,4,5,10,15]}]
gs = GridSearchCV(model, param_grid=param_grid, scoring='r2')
gs.fit(X_train, y_train)

gs_score = gs.score(X_test, y_test)





