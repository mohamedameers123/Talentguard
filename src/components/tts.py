from sklearn import model_selection
from sklearn.model_selection import train_test_split

from src.components.feature_target import feature_targets
from src.components.data_load import load_data
def train_test(X,y,t_size=0.2):
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=t_size,stratify=y)
    return X_train,X_test,y_train,y_test

def main():
    data=load_data()
    X,y=feature_targets(data)
    X_train,X_test,y_train,y_test=train_test(X,y)

if __name__=="__main__":
    main()