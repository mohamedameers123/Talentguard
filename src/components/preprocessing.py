from src.components.data_load import load_data
from src.components.feature_target import feature_targets
from pandas import pandas
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from src.config.pa_th import model_path
from sklearn.impute import SimpleImputer
import joblib
import numpy as np


def num_pipeline():
    return Pipeline(steps=[
    ("imputer",SimpleImputer()),
    ("scaler",StandardScaler())])
def cat_pipeline():
    return Pipeline(steps=[
    ("imputer",SimpleImputer(strategy='most_frequent')),
    ("encoder",OneHotEncoder())])

def create_preprocess(num_data,cat_data):
    preprocess=ColumnTransformer(transformers=[
        ("num",num_pipeline(),num_data),
        ("cat",cat_pipeline(),cat_data)])
    return preprocess

def call_preprocess(X_train,X_test,num,cat):
    preprocess=create_preprocess(num,cat)
    X_train=preprocess.fit_transform(X_train)
    X_test=preprocess.transform(X_test)
    joblib.dump(preprocess,model_path/"preprocessing.pkl")
    return X_train,X_test,preprocess
def encode_y(y_train,y_test):
    mapping={
        "Yes":0,
        "No":1

    }
    y_train=y_train.replace(mapping)
    y_test=y_test.replace(mapping)
    y_train=y_train.astype("int")
    y_test=y_test.astype("int")
    return y_train,y_test

