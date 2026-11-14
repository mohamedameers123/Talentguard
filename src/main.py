from components.data_load import load_data
from components.data_validation import dataset_check
from components.eda import eda_operations
from components.visualizations import save_charts
from components.feature_target import feature_target
def main():
    data=load_data()
    #dataset_check(data)
    #save_charts(data)
    X,y=feature_target(data)
if __name__=='__main__':
    main()