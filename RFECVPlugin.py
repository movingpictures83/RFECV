
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import linear_model


class RFECVPlugin:
 def input(self, inputfile):
  self.data_path = inputfile
 def run(self):
  pass
 def output(self, outputfile):
  #categorical_cols = ["Race_Ethnicity"]


  data_df = pd.read_csv(self.data_path)

  # # Tramsform categorical data to categorical format:
  # for category in categorical_cols:
  #     data_df[category] = data_df[category].astype('category')
  #

  # Clean numbers:
  #"Cocain_Use": {"yes":1, "no":0},
  cleanup_nums = { "Cocain_Use": {"yes":1, "no":0},
                 "race": {"White":1, "Black":0, "BlackIsraelite":0, "Latina":1},
  }

  data_df.replace(cleanup_nums, inplace=True)

  # Drop id column:
  data_df = data_df.drop(["pilotpid"], axis=1)

  # remove NaN:
  data_df = data_df.fillna(0)

  # Standartize variables
  from sklearn import preprocessing
  names = data_df.columns
  scaler = preprocessing.StandardScaler()
  data_df_scaled = scaler.fit_transform(data_df)
  data_df_scaled = pd.DataFrame(data_df_scaled, columns=names)

  y_col = "interleukin6"
  test_size = 0.25
  validate = True
  random_state = 2

  y = data_df[y_col]

  X = data_df_scaled.drop([y_col], axis=1)


  # Try RFE
  from sklearn.linear_model import LinearRegression
  from sklearn.feature_selection import RFECV
  from itertools import compress

  model = LinearRegression()
  #Initializing RFE model
  rfe = RFECV(estimator=model, step=1, scoring='neg_mean_squared_error', cv=5)

  # #Transforming data using RFE
  # X_rfe = rfe.fit_transform(X_corr_scaled,y)
  # #Fitting the data to model
  # model.fit(X_rfe,y)
  # # print(rfe.support_)
  # # print(rfe.ranking_)
 
  #Transforming data using RFE
  X_rfe = rfe.fit_transform(X,y)
  #Fitting the data to model
  model.fit(X,y)
  # print(rfe.support_)
  # print(rfe.ranking_)


  # #Transforming data using RFE
  # X_rfe = rfe.fit_transform(X_train,y_train)
  # #Fitting the data to model
  # model.fit(X,y)
  # # print(rfe.support_)
  # # print(rfe.ranking_)

  # #Transforming data using RFE
  # X_rfe = rfe.fit_transform(X_train_corr,y_train_corr)
  # #Fitting the data to model
  # model.fit(X_rfe,y_train_corr)
  # # print(rfe.support_)
  # # print(rfe.ranking_)


  columns = X.columns

  columns_rfe = list(compress(columns, rfe.support_))

  print("Selected {} columns out of {}".format(len(columns_rfe), len(columns)))
  print(columns_rfe)




  # In[363]:


  len(X.columns)


