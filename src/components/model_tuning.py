from src.components.model_factory import load_models
from sklearn.model_selection import GridSearchCV
import joblib
from src.config.pa_th import model_tun_path

models={
    "Logistic_Regression":load_models()["Logistic_Regression"],
    "knn":load_models()["knn"],
    "Decision Tree":load_models()["Decision Tree"],
    "Random Forest":load_models()["Random Forest"],
    "Ada boost":load_models()["Ada boost"],
    "Gradient Boosting":load_models()["Gradient Boosting"],
    "XGBoost":load_models()["XGBoost"]
}

param={
        "Logistic_Regression":{
            "random_state":[42]},
        "knn":{
            "n_neighbors":[5,7,10]},
        "Decision Tree":{
            "max_depth":[8,10,15]},
        "Random Forest":{
            "n_estimators":[100,200],
            "max_depth":[10,12]},
        "Ada boost":{
            "n_estimators":[100,200],
            "learning_rate":[0.1,0.01]},
        "Gradient Boosting":{
            "max_depth":[5,8],
            "learning_rate":[0.1,0.01],
            "n_estimators":[150]},
        "XGBoost":{
            "n_estimators":[100,150],
            "random_state":[42]}
        }

def tuning(X_train,y_train):
    tune={}
    best_param=""
    best_score=0
    best_model=""
    best_model_name=""
    for model_name,model in models.items():
        print(f"Tuning {model_name}")
        params=param[model_name]
        grid_search=GridSearchCV(estimator=model,param_grid=params,
                                 cv=5,n_jobs=-1)
        grid_search.fit(X_train,y_train)
        current_param=grid_search.best_params_
        current_model=grid_search.best_estimator_
        current_model_name=model_name
        current_score=grid_search.best_score_
        tune[model_name]={
            "best_score":current_score,
            "best_param":current_param,
            "best_model":current_model
        }
        print(f"{model_name}Tuning completed")
        if current_score > best_score:
            best_score=current_score
            best_param=current_param
            best_model=grid_search.best_estimator_
            best_model_name=current_model_name=model_name
    joblib.dump(best_model,model_tun_path/f"{best_model_name}.pkl")
    return (best_model_name,best_model,best_param,
            best_score,tune,)