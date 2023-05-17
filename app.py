from flask import Flask
from flask import render_template
from flask import request

import pickle
import pandas as pd

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/input')
def input():
    return render_template('input.html')


@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'GET':
        return render_template('index.html')
    elif request.method == 'POST':
        HighBP = request.form['HighBP']
        HighChol = request.form['HighChol']
        CholCheck = request.form['CholCheck']
        Height = request.form['Height']
        Weight = request.form['Weight']
        Smoker = request.form['Smoker']
        Stroke = request.form['Stroke']
        Diabetes = request.form['Diabetes']
        PhysActivity = request.form['PhysActivity']
        Fruits = request.form['Fruits']
        Veggies = request.form['Veggies']
        HvyAlcoholConsump = request.form['HvyAlcoholConsump']
        MentHlth = request.form['MentHlth']
        GenHlth = request.form['GenHlth']
        PhysHlth = request.form['PhysHlth']
        DiffWalk = request.form['DiffWalk']
        Sex = request.form['Sex']
        Age = request.form['Age']
        Height = float(Height)/100
        BMI = float(Weight)//(Height*Height)
        df = pd.DataFrame([[HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, Diabetes, PhysActivity,Fruits, Veggies, HvyAlcoholConsump, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age]],columns=["HighBP","HighChol","CholCheck","BMI","Smoker","Stroke","Diabetes","PhysActivity","Fruits","Veggies","HvyAlcoholConsump","GenHlth","MentHlth","PhysHlth","DiffWalk","Sex","Age"])

        pred = predict(df)
        print(pred)
        result = pred[0]
        return render_template('result.html', res=result)
    else:
        return render_template('index.html')


@app.errorhandler(404)
def page_not_found(error):
    return render_template('err404.html'), 404


def predict(input):
    model = pickle.load(open('model/model.pkl', 'rb'))
    loaded_sc = pickle.load(open('model/scaler.pkl', 'rb'))
    df = loaded_sc.transform(input)
    res = model.predict(df)
    return res
