from src.components.data_load import load_data
#from src.components.data_validation import dataset_check
#from src.components.eda import eda_operations
#from src.components.visualizations import save_charts
#from src.components.feature_target import feature_target
from src.components.transform import transform_data
def main():
    data=load_data()
    #dataset_check(data)
    #save_charts(data)
    #X,y=feature_target(data)
    cat_data,num_data,target_data=transform_data(data)
if __name__=='__main__':
    main()