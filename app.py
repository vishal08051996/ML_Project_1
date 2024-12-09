import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from src.exception import CustomException
from src.logger import logging
from src.pipeline.predict import Prediction

# Navigation Bar
selected_page = option_menu(
    menu_title="Navigation",  # Title of the navigation bar
    options=["Home", "Parameter Info", "Prediction"],  # Pages
    icons=["house", "info-circle", "calculator"],  # Icons from Bootstrap (optional)
    menu_icon="menu-app",  # Icon for the menu
    default_index=0,  # Default selected page
    orientation="horizontal",  # Horizontal menu
)

# Page 1: Home
if selected_page == "Home":
    st.title("Welcome to the Math Score Prediction App!")
    st.markdown("""
    This app predicts the **Math Score** based on several parameters:
    - Gender
    - Race/Ethnicity
    - Parental Level of Education
    - Lunch Type
    - Test Preparation Course
    - Reading and Writing Scores
    
    **How to Use the App:**
    1. Navigate to the "Parameter Info" page for details on the parameters.
    2. Go to the "Prediction" page, input values, and get the predicted Math Score.
    """)

# Page 2: Parameter Info
elif selected_page == "Parameter Info":
    st.title("Parameter Information")
    st.markdown("""
    ### Input Parameters:
    1. **Gender**: Select the gender of the student (male or female).
    2. **Race/Ethnicity**: Choose one of the racial/ethnic groups (e.g., Group A, Group B).
    3. **Parental Level of Education**: Specify the highest education level achieved by the parent (e.g., bachelor's degree, high school).
    4. **Lunch Type**: Indicate whether the student had standard or free/reduced lunch.
    5. **Test Preparation Course**: Select whether the student completed or skipped the test preparation course.
    6. **Reading Score**: Provide the student's reading score (0-100).
    7. **Writing Score**: Provide the student's writing score (0-100).
    """)

# Page 3: Prediction
elif selected_page == "Prediction":
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

    if st.button("Submit", use_container_width=True):
        # Preparing input data
        feed_df = pd.DataFrame([[gender, race_ethnicity, parental_level_of_education, lunch, 
                    test_preparation_course, reading_score, writing_score]],
                    columns=["gender", "race_ethnicity", "parental_level_of_education", "lunch", 
                             "test_preparation_course", "reading_score", "writing_score"])

        try:
            # Instantiate the Prediction class
            predictor = Prediction()

            # Load preprocessor and model
            preprocessor, model = predictor.upload_models()
            logging.info("Models loaded successfully.")

            # Preprocess input data
            feed = preprocessor.transform(feed_df)
            logging.info("Transformed Input Data")

            # Make prediction
            pred = model.predict(feed)
            st.markdown("---")
            st.write("### Predicted Math Score:", round(pred[0], 2))
            st.markdown("---")
        except Exception as e:
            raise CustomException(e)
