from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)
app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('home.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('index.html')
    else:
        data=CustomData(
            long_hair=request.form.get('long_hair'),
            forehead_width_cm=request.form.get('forehead_width_cm'),
            forehead_height_cm=request.form.get('forehead_height_cm'),
            nose_wide=request.form.get('nose_wide'),
            nose_long=request.form.get('nose_long'),
            lips_thin=request.form.get('lips_thin'),
            distance_nose_to_lip_long=float(request.form.get('distance_nose_to_lip_long')),
            gender=float(request.form.get('gender'))

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('index.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0")