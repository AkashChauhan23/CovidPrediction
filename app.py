import pickle
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

# importing Model
model = load_model("covid_prediction.keras")

def pickel_loader(pickler_name, input_data):
    with open(pickler_name+'.pkl', 'rb') as file:
        pickle_load = pickle.load(file)
        input_data[pickler_name] = pickle_load.transform([input_data[pickler_name]])
        return input_data

# title
st.title('Covid Prediction')

# user inputs
# 'Cough_symptoms', 'Fever', 'Sore_throat', 'Shortness_of_breath', 'Headache', 'Age_60_above', 'Sex', 'Known_contact', 'Corona'

binary_select = (True, False)
cough = st.checkbox('Do you have symptoms of Cough: ', binary_select)
fever = st.checkbox('Do you have symptoms of Fever: ', binary_select)
sore_throat = st.checkbox('Do you have symptoms of Sore throat: ', binary_select)
breathing_shortness = st.checkbox('Do you have Breathing issues: ', binary_select)
headache = st.checkbox('Do you have Headache: ', binary_select)
age = st.radio('Is your age greater than 60? ', ('Yes', 'No'))
sex = st.radio('Is your gender male? ', ('male', 'female'))
known_contact = st.radio('Do you any covid patient in your relatives: ', ('Other', 'Contact with confirmed'))

analysis = st.button('Check the Covid analysis: ' , type='primary' , use_container_width=True )

if analysis:
    input = {
        'Cough_symptoms' : cough,
        'Fever' : fever,
        'Sore_throat' : sore_throat,
        'Shortness_of_breath' : breathing_shortness,
        'Headache' : headache,
        'Age_60_above' : age,
        'Sex' : sex,
        'Known_contact' : known_contact,
    }

    for key, value in input.items():
        input = pickel_loader(key, input)
    input_df = pd.DataFrame(input)

    with open('scaller.pkl', 'rb') as file:
        scaller_file = pickle.load(file)
        input_df = scaller_file.transform(input_df)
        predict = model.predict(input_df)

        if predict > .5:
            st.info(f'You are COVID Positive. Prediction: {int(predict[0][0]*100)} %')
        else:
            st.info(f'You are not COVID Positive. Prediction: {int(predict[0][0]*100)} %')
