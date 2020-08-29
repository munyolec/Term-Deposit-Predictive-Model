import numpy as np
import pandas as pd
from sklearn.model_selection import KFold 
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
# import xgboost as xgb


#function to do fit our models and perform k-fold cross validation
def modelKfold(model, x,y,X_train):
    logReg=LogisticRegression()
    mlp=MLPClassifier()
    dectree=DecisionTreeClassifier()
    
    scores=[]
    cv=KFold(n_splits=5, random_state=42,shuffle=True)
    for trainIndex, testIndex in cv.split(X_train):
        cvX_train, cvX_test=x.iloc[trainIndex], x.iloc[testIndex]
        cvy_train, cvy_test=y.iloc[trainIndex], y.iloc[testIndex]

        modelTrained=model.fit(cvX_train,cvy_train)
        ypred=modelTrained.predict(x.iloc[testIndex])

        scores.append(accuracy_score(cvy_test,ypred))
    return modelTrained, scores

#function to make our predictions
def predictor(model,X_test,y_test):
    y_pred=model.predict(X_test)
    df_pr=pd.DataFrame({'actual':y_test.values.flatten(),'predicted':y_pred.flatten()})
    return df_pr['predicted'].value_counts()

#function to evaluate our models
def evaluate_model(metric,model,X_test,y_test):
    y_pred=model.predict(X_test)
    evaluation=metric(y_test, y_pred)
    return evaluation

def strKfold(model,X,y,X_test):
    
    kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    pred_test_full =0
    cv_score =[]
    i=1
    for train_index,test_index in kf.split(X,y):
        print('{} of KFold {}'.format(i,kf.n_splits))
        xtr,xvl = X.loc[train_index],X.loc[test_index]
        ytr,yvl = y.loc[train_index],y.loc[test_index]
        
        #model
        mlp = MLPClassifier()
        modelTrained1=model.fit(xtr,xvl)
        score = roc_auc_score(yvl,model.predict(xvl))
        print('ROC AUC score:',score)
        cv_score.append(score)

        pred_test=model.predict_proba(X_test)[:,1]
        pred_test_full +=pred_test
        i+=1
    return cv_score,modelTrained1

