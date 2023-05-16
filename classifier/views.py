from django.shortcuts import render
from django.shortcuts import HttpResponse
from django import forms

import pandas as pd
import json
import category_encoders as ce
from sklearn.naive_bayes import GaussianNB

class UploadFileForm(forms.Form):
    title = forms.CharField(max_length=50)
    file = forms.FileField()

def index(request):
    training = pd.read_csv('classifier/watch_movies.csv')
    testing = pd.read_csv('classifier/watch_movies_testing.csv')

    prediction = naive_bayes(training, testing)
    testing = testing.assign(ver=prediction)
    
    json_data = training.reset_index().to_json(orient ='records')
    training_data = []
    training_data = json.loads(json_data)

    json_data = testing.reset_index().to_json(orient ='records')
    testing_data = []
    testing_data = json.loads(json_data)

    context = {'training_headers': training.columns, 'training_data': training_data, 'testing_headers': testing.columns, 'testing_data': testing_data}

    return render(request, 'classifier/table.html', context)


def naive_bayes(training, testing):
    X = training.drop('ver', axis=1)
    y = training['ver']

    enc = ce.OrdinalEncoder(cols=X.columns)
    X = enc.fit_transform(X)
    testing = enc.transform(testing)

    bnb = GaussianNB()
    bnb.fit(X, y)

    prediction = bnb.predict(testing)
    return prediction