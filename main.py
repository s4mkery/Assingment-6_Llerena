from sklearn import datasets
from sklearn import svm, tree, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

cancer = datasets.load_breast_cancer()
x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=250, stratify= y)

svc_model = svm.SVC(random_state=250)
svc_model.fit(x_train, y_train)
svc_y_predict = svc_model.predict(x_test)

tree_model = tree.DecisionTreeClassifier(random_state=250)
tree_model.fit(x_train, y_train)
tree_y_predict = tree_model.predict(x_test)

logistic_model = linear_model.LogisticRegression(max_iter=5000,random_state=250)
logistic_model.fit(x_train, y_train)
logistic_y_predict = logistic_model.predict(x_test)

preds = {
    "SVC": svc_y_predict,
    "Decision Tree Classifier": tree_y_predict,
    "Logistic Regression": logistic_y_predict
}
print(f"{'Model':<32} | {'Accuracy':<8.4} - {'Precision':<9.4} - {'Recall':<8.8} - {'F1-Score':<8.8}")
for model, y_pred in preds.items():
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"{model:<32} | {acc:<8.4f} - {prec:<9.4f} - {rec:<8.4f} - {f1:<8.4f}")