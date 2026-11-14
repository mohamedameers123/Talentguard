import pandas as pd
from src.components.data_load import load_data
def transform_data(data:pd.DataFrame):
    #method 1
    num_data=data.select_dtypes(include="number").columns.tolist()
    #method 2
    cat_data=[i for i in data.columns if data[i].dtype=="str"
              and i !="Attrition"]
    target_data=data["Attrition"]
    return cat_data,num_data,target_data

if __name__=="__main__":
    data=load_data()
    cat_data,num_data,target_data=transform_data(data)
    print("cat_data", "\n" + "-" * len("cat_data"))
    print(cat_data)
    print("num_data", "\n" + "-" * len("num_data"))
    print(num_data)
    print("target_data", "\n" + "-" * len("target_data"))
    print(target_data)

