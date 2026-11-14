from data_load import load_data
def feature_target():
    data=load_data()
    print(data.head())
if __name__=="__main__":
    feature_target()