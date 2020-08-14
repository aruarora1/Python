# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 05:58:17 2020

@author: aruar
"""
import pandas as pd
import numpy as np
import sys
#!{sys.executable} -m pip install pandas-profiling
from sklearn.preprocessing import OneHotEncoder

testDF = pd.DataFrame({"empid":["emp1","emp2","emp4"],"dept":["hr","it","it"]})
testDF

ohEncoder = OneHotEncoder(sparse=False)

#encoder returns series
encodedSeries = ohEncoder.fit_transform(testDF[["dept"]])
# once transfromed, it returns series.. doesn't print great but we can check 
#column names by categories
ohEncoder.categories_

# this series will ONLY have the transformed columns.. 
# the untransformed are to be concatenated back

#before i concatenate, i need to convert transformed series to Dataframe
encodedDF=pd.DataFrame(encodedSeries,dtype=np.int8,columns=ohEncoder.categories_)


pd.concat([testDF,encodedDF],axis=1)

# OPTION 2 - USE COLUMN TRANSFORMER.. 

from sklearn.compose import make_column_transformer

# column transformer allows us to do multiple transformations on multiple columns
# in a single command, below i'm doing only one transformation. but this can be
# easily changed to multiple by adding more parameters

transformtestDF = make_column_transformer((OneHotEncoder(),['dept']),remainder='passthrough')

transformtestDF.fit_transform(testDF)



