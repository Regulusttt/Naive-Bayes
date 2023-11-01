import pandas as pd
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import numpy as np

curr_dir = os.path.dirname(os.path.abspath(__file__))
training_file = os.path.join(curr_dir, 'Training.csv')
data = pd.read_csv(training_file)

num = LabelEncoder()
inputs = data.drop(['Scenario', 'PlayTennis'], axis='columns')
result = data['PlayTennis']

inputs['New_Outlook'] = num.fit_transform(inputs['Outlook'])
inputs['New_Temperature'] = num.fit_transform(inputs['Temperature'])
inputs['New_Humidity'] = num.fit_transform(inputs['Humidity'])
inputs['New_Wind'] = num.fit_transform(inputs['Wind'])

new_input = inputs.drop(['Outlook', 'Temperature', 'Humidity', 'Wind'], axis = 'columns')

Classifier = GaussianNB()
Classifier.fit(new_input, result)
print(Classifier.score(new_input, result))

user_outlook = int(input("Masukan Outlook(Overcast=0/Rain=1/Sunny=2) = "))
user_temperature = int(input("Masukan Temperature(Cool=0/Hot=1/Mild=2) = "))
user_humidity = int(input("Masukan Humidity(High=0/Normal=1) = "))
user_wind = int(input("Masukan Wind(Strong=0/Weak=1) = "))

prediction = Classifier.predict([[user_outlook,user_temperature,user_humidity,user_wind]])
print(f"Hasil Prediksi: Play Tennis = {prediction[0]}")