import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

## import ridge regresor model and standard scaler pickle
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))

## Route for home page
 

@app.route('/',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        bedrooms=float(request.form.get('bedrooms'))
        bathrooms = float(request.form.get('bathrooms'))
        sqft_living = float(request.form.get('sqft_living'))
        floors = float(request.form.get('floors'))

        new_data_scaled=standard_scaler.transform([[bedrooms,bathrooms,sqft_living,floors]])
        result=ridge_model.predict(new_data_scaled)

        return render_template('home.html',result=result[0])

    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")
