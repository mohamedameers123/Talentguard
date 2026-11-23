from src.components.data_load import load_data
from src.components.data_validation import dataset_check
from src.components.eda import eda_operations
from src.components.visualizations import save_charts
from src.components.feature_target import feature_targets
from src.components.transform import transform_data
from src.components.tts import train_test
from src.components.preprocessing import create_preprocess,call_preprocess,encode_y
from src.components.model_factory import load_models


def main():
    #data=load_data()
    #dataset_check(data)
    #save_charts(data)
    #X,y=feature_targets(data)
    #cat_data,num_data,target_data=transform_data(data)
    #X_train,X_test,y_train,y_test=train_test(X,y)
    #preprocess=create_preprocess(num_data,cat_data)
    #X_train,X_test,preprocess=call_preprocess(X_train,X_test,num_data,cat_data)
    #y_train,y_test=encode_y(y_train,y_test)
    print(load_models())


if __name__=='__main__':
    main()