from flask import Flask, render_template, request, redirect
import sklearn
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
cors = CORS(app)
model = pickle.load(open("LinearRegressionModel.pkl",'rb'))
rate = pd.read_csv('data.csv')


@app.route('/', methods=['GET', 'POST'])
def index():
    ReverseRepoRate = rate['ReverseRepoRate'].unique()
    BankRate = rate['BankRate'].unique()
    BaseRate = rate['BaseRate'].unique()
    SavingsDepositRate = rate['SavingsDepositRate'].unique()
    Inflation = rate['Inflation'].unique()

    return render_template('index.html', ReverseRepoRate=ReverseRepoRate, BankRate=BankRate, BaseRate=BaseRate, SavingsDepositRate=SavingsDepositRate, Inflation=Inflation)


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():

    ReverseRepoRate = request.form.get('ReverseRepoRate')
    BankRate = request.form.get('BankRate')
    BaseRate = request.form.get('BaseRate')
    SavingsDepositRate = request.form.get('SavingsDepositRate')
    Inflation = request.form.get('Inflation')

    prediction = model.predict(pd.DataFrame([[ReverseRepoRate, BankRate, BaseRate, SavingsDepositRate, Inflation]], columns=	['ReverseRepoRate',	'BankRate',	'BaseRate',	'SavingsDepositRate', 'Inflation']))

    print(prediction)

    return str(np.round(prediction[0], 2))


if __name__ == '__main__':
    app.run()
