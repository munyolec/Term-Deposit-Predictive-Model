from sklearn.model_selection import KFold 
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# import xgboost as xgb



def modelKfold(model, x,y,X_train):
    scores=[]
    cv=KFold(n_splits=5, random_state=42,shuffle=True)
    for trainIndex, testIndex in cv.split(X_train):
        cvX_train, cvX_test=x.iloc[trainIndex], x.iloc[testIndex]
        cvy_train, cvy_test=y.iloc[trainIndex], y.iloc[testIndex]

        modelTrained=model.fit(cvX_train,cvy_train)
        ypred=modelTrained.predict(x.iloc[testIndex])

        scores.append(accuracy_score(cvy_test,ypred))
    return modelTrained, scores