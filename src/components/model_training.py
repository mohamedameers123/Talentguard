import pandas as pd

from src.components.model_factory import load_models
from sklearn.metrics import (accuracy_score,precision_score,
                             recall_score,f1_score)
import joblib
from src.config.pa_th import report_path,model_path

def evolution_metrics(model,X_test,y_test):
    pred=model.predict(X_test)
    return {"accuracy":accuracy_score(y_test,pred),
            "precision":precision_score(y_test,pred),
            "recall":recall_score(y_test,pred),
            "f1":f1_score(y_test,pred)}

def model_training(X_train,X_test,y_train,y_test):
    models=load_models()
    lb=[]
    best_model=""
    best_model_name=""
    best_f1=0
    for model_name,model in models.items():
        print(model,"Training started")
        model.fit(X_train,y_train)
        metrics=evolution_metrics(model,X_test,y_test)
        lb.append({"model":model,
                   "accuracy":round(metrics["accuracy"],4),
                   "precision":round(metrics["accuracy"],4),
                   "recall":round(metrics["recall"],4),
                   "f1":round(metrics["f1"],4)
                   })
        if metrics["f1"]>best_f1:
            best_f1=metrics["f1"]
            best_model=model
            best_model_name=model_name

    lb=pd.DataFrame(lb)
    lb.sort_values("f1",ascending=False,inplace=True)
    lb.reset_index(drop=True,inplace=True)

    return lb,best_model,best_model_name

def save_models(model,model_name):
    joblib.dump(model,model_path/f"{model_name}.pkl")

def mod_call(X_train,X_test,y_train,y_test):
    lb,best_model,best_model_name=model_training(X_train,
                                                 X_test,
                                                 y_train,y_test)
    print(lb)
    save_models(best_model,best_model_name)
    print(best_model_name,best_model)




