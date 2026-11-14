from components.data_load import load_data
import pandas as pd
def feature_target(data:pd.DataFrame):
    data=load_data()
    data=data.drop("EmployeeNumber",axis=1)
    X=data.drop("Attrition",axis=1)
    y=data[["Attrition"]]

    return X,y
if __name__=="__main__":
    X,y=feature_target(load_data)