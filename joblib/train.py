import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import joblib
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
names=['preg', 'plas','pres','skin','test','mass','pedi','age','class']
df = pd.read_csv(url, names=names)
print(df)
print(df.info())

X = df.iloc[:,0:8]
y = df.iloc[:,8]
print(X)
print(y)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size =0.2,random_state =101)

model = LogisticRegression()
model.fit(X_train,y_train)
print("[Info] Model Has been trained")

result = model.score(X_test,y_test)
print(f"Accuracy of the model is {result}")

joblib.dump(model,'dib_79.pkl')


