import joblib

loaded_model = joblib.load('dib_79.pkl') 
pred = loaded_model.predict([[10,20,20,30,30,20,30,20]])
print(pred)

if pred[0]==1:
    print('Person is a diabetic')
else:
    print('Person is non-diabetic')
    