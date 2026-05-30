import gradio as gr
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open('insurance_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Gradio web app

def predict_insurance(age, sex, bmi, children, smoker, region):
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })

    prediction = model.predict(input_data)
    prediction = np.exp(prediction) - 1

    return prediction[0]

inputs=[
    gr.Number(label='Age', value=int),
    gr.Radio(['male', 'female'], label='Sex'),
    gr.Number(label='BMI'),
    gr.Slider(minimum=0, maximum=5, step=1, label='Number of Children'),
    gr.Radio(['yes', 'no'], label='Smoker'),
    gr.Radio(['southwest', 'southeast', 'northwest', 'northeast'], label='Region')
]

app = gr.Interface(
    fn=predict_insurance,
    inputs=inputs,
    outputs=gr.Number(label='Insurance Cost'),
    title='Medical Insurance Cost Prediction'
)

app.launch(share=True)