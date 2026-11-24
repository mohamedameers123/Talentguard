from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,
                              AdaBoostClassifier,
                              GradientBoostingClassifier)
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

def load_models():
    return {"Logistic_Regression":LogisticRegression(random_state=42,),
            "knn":KNeighborsClassifier(n_neighbors=5,),
            "Decision Tree":DecisionTreeClassifier(random_state=42),
            "Random Forest":RandomForestClassifier(random_state=42),
            "Ada boost":AdaBoostClassifier(),
            "Gradient Boosting":GradientBoostingClassifier(),
            "XGBoost":XGBClassifier()}

def main():
    models=load_models()
    print("This is model factory")
    for model_name,model in models.items():
        print(model_name,"\n",model)


if __name__=="__main__":
    main()