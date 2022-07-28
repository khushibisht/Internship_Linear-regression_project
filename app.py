# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 10:40:05 2022

@author: khushi bisht
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
#from flask_ngrok import run_with_ngrok
import pickle


app = Flask(__name__)
model = pickle.load(open('lr_assignment1.pkl','rb')) 
#run_with_ngrok(app)

@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
    
    '''
    For rendering results on HTML GUI
    '''
    exp = float(request.args.get('exp'))
    
    prediction = model.predict([[exp]])
    
        
    return render_template('index.html', prediction_text='Regression Model  has predicted price for given square-feet is : {}'.format(prediction))


if(__name__)=='main':
  app.run(debug==True)