import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

df=pd.read_csv("students_old one.csv")

df=pd.get_dummies(df,columns=["gender"],drop_first=True)

x=df.drop("score",axis=1)
y=df["score"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

rf=RandomForestRegressor(n_estimators=100)

rf.fit(x_train,y_train)

joblib.dump(rf,"score_train_model.pkl")


