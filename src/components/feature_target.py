from components.data_load import load_data
def feature_target():
    data=load_data()
    print(data.head())
    print("this is the end".center(70))
if __name__=="__main__":
    feature_target()