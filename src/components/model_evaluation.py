from cProfile import label


from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             roc_auc_score, classification_report,
                             confusion_matrix, roc_curve, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt


def model_metrics(model,X_test,y_test):
    y_pred=model.predict(X_test)
    proba=model.predict_proba(X_test)[:,1]
    metrics={"accuracy":accuracy_score(y_test,y_pred),
             "precision":precision_score(y_test,y_pred,zero_division=0),
             "recall":recall_score(y_test,y_pred,zero_division=0),
             "f1":f1_score(y_test,y_pred,zero_division=0),
             "roc_auc_score":roc_auc_score(y_test,y_pred)}
    return y_pred,proba,metrics

def display_metrics(metrics):
    for metrics_name,metric in metrics.items():
        print(f"{metrics_name}:{metric:.2f}")

def display_calssification_report(y_test,y_pred):
    report=classification_report(y_test,y_pred,target_names=[
        "Attrition",
        "No Attrition"],zero_division=0)
    print("Classification Report :\n")
    print(report)

def display_confusion_matrix(y_test,y_pred):
    cm=confusion_matrix(y_test,y_pred)
    disp=ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=["Attrition",
                                                "No Attrition"])
    disp.plot()
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

def display_roc_curve(y_test,proba,roc):
    false_pos_rate,true_pos_rate,_=roc_curve(y_test,proba)
    plt.plot(false_pos_rate,true_pos_rate,
             label=f"ROC SCORE :{roc}")
    plt.plot([0,1],[0,1],ls="--",label="Random_Classifier")
    plt.tight_layout()
    plt.legend()
    plt.show()


def mod_evaluation(model,model_name,X_test,y_test):
    print(f"{model_name}")
    y_pred,proba,metrics=model_metrics(model,X_test,y_test)
    display_metrics(metrics)
    display_calssification_report(y_test,y_pred)
    display_confusion_matrix(y_test,y_pred)
    display_roc_curve(y_test,proba,metrics["roc_auc_score"])
    return metrics