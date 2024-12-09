import pandas as pd
import numpy as np
import sys
import streamlit as st
from src.exception import CustomException
from src.logger import logging
from src.pipeline.predict import Prediction

# Streamlit app title
st.title("Prediction of Math Score")

# Input fields
gender = st.selectbox("Select your gender", ["male", "female"])
race_ethnicity = st.selectbox("Select your race/ethnicity", ['group A', 'group B', 'group C', 'group D', 'group E'])
parental_level_of_education = st.selectbox(
    "Select your parental level of education",
    ["bachelor's degree", 'some college', "master's degree", "associate's degree", 'high school', 'some high school']
)
lunch = st.selectbox("Select your lunch type", ['standard', 'free/reduced'])
test_preparation_course = st.selectbox("Select your test preparation course", ['none', 'completed'])

reading_score = st.number_input("Enter your reading score", min_value=0.0, max_value=100.0, step=1.0)
writing_score = st.number_input("Enter your writing score", min_value=0.0, max_value=100.0, step=1.0)

if st.button("submit",use_container_width=True):
    # Preparing input data
    feed_df = pd.DataFrame([[gender, race_ethnicity, parental_level_of_education, lunch, 
                test_preparation_course, reading_score, writing_score]],
                columns=["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course","reading_score","writing_score"])

    try:
        # Instantiate the Prediction class
        predictor = Prediction()

        # Load preprocessor and model
        preprocessor, model = predictor.upload_models()
        logging.info("Models loaded successfully.")

        # Preprocess input data
        feed = preprocessor.transform(feed_df)
        logging.info("Transformed Input Data:")

        # Make prediction
        pred = model.predict(feed)
        st.markdown("---")
        st.markdown("---")
        st.write("Predicted Math Score:", round(pred[0], 2))
        st.markdown("---")
        st.markdown("---")
    except Exception as e:
        raise CustomException(e,sys)
