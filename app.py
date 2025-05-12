# import streamlit as st
# import pickle
# import numpy as np
# https://github.com/rakibnsajib/Crop-Recommendation-Using-Machine-Learning/blob/main/README.md
# # Load the MinMaxScaler and the model
# with open("minmaxscaler.pkl", "rb") as scaler_file:
#     scaler = pickle.load(scaler_file)

# with open("model.pkl", "rb") as model_file:
#     model = pickle.load(model_file)

# # Crop dictionary for mapping model output to crop names
# crop_dict = {
#     1: 'rice',
#     2: 'maize',
#     3: 'chickpea',
#     4: 'kidneybeans',
#     5: 'pigeonpeas',
#     6: 'mothbeans',
#     7: 'mungbean',
#     8: 'blackgram',
#     9: 'lentil',
#     10: 'pomegranate',
#     11: 'banana',
#     12: 'mango',
#     13: 'grapes',
#     14: 'watermelon',
#     15: 'muskmelon',
#     16: 'apple',
#     17: 'orange',
#     18: 'papaya',
#     19: 'coconut',
#     20: 'cotton',
#     21: 'jute',
#     22: 'coffee'
# }

# # Function to make predictions


# def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
#     # Input as a numpy array
#     input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

#     # Scale the input
#     scaled_input = scaler.transform(input_data)

#     # Make prediction
#     prediction = model.predict(scaled_input)

#     # Map the numerical prediction to the crop name
#     crop_prediction = crop_dict.get(int(prediction[0]), "Unknown crop")

#     return crop_prediction


# # Streamlit interface
# st.title("Crop Recommendation using Machine Learning")

# # Create input fields
# N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=100.0, step=0.1)
# P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=100.0, step=0.1)
# K = st.number_input("Potassium (K)", min_value=0.0, max_value=100.0, step=0.1)
# temperature = st.number_input(
#     "Temperature", min_value=-10.0, max_value=60.0, step=0.1)
# humidity = st.number_input("Humidity", min_value=0.0,
#                            max_value=100.0, step=0.1)
# ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, step=0.1)
# rainfall = st.number_input("Rainfall", min_value=0.0,
#                            max_value=500.0, step=0.1)

# # Button to trigger prediction
# if st.button("Predict Crop"):
#     result = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
#     st.success(f'The predicted crop is: {result}')
import streamlit as st
import pickle
import numpy as np
import time

# Load the MinMaxScaler and the model
with open("minmaxscaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Crop dictionary for mapping model output to crop names
crop_dict = {
    1: 'rice',
    2: 'maize',
    3: 'chickpea',
    4: 'kidneybeans',
    5: 'pigeonpeas',
    6: 'mothbeans',
    7: 'mungbean',
    8: 'blackgram',
    9: 'lentil',
    10: 'pomegranate',
    11: 'banana',
    12: 'mango',
    13: 'grapes',
    14: 'watermelon',
    15: 'muskmelon',
    16: 'apple',
    17: 'orange',
    18: 'papaya',
    19: 'coconut',
    20: 'cotton',
    21: 'jute',
    22: 'coffee'
}

# Function to make predictions


def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    # Input as a numpy array
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    # Scale the input
    scaled_input = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_input)

    # Map the numerical prediction to the crop name
    crop_prediction = crop_dict.get(int(prediction[0]), "Unknown crop")

    return crop_prediction


# Streamlit interface setup
st.set_page_config(
    page_title="Crop Recommendation using Machine Learning", layout="wide")

# Title for the app
st.title("Crop Recommendation using Machine Learning")

# Sidebar Title and Navigation
st.sidebar.title("Crop Recommendation using Machine Learning")
page = st.sidebar.radio("Select a page:", ["About", "Recommendation"])

# Add the "Made by" text to the sidebar
st.sidebar.markdown("---")
st.sidebar.markdown(
    "Made by [the-amazing-atharva](https://github.com/the-amazing-atharva)")

if page == "About":
    # About section: Provide information about the project
    st.header("About this Project")

    # Display image below the title
    st.image("https://camo.githubusercontent.com/1bdf0f0577d91db85f8f22e7f8ed20e301498a172302c0d4455b58669114195c/68747470733a2f2f6d656469612e6c6963646e2e636f6d2f646d732f696d6167652f76322f4335313132415145366466527a5633454a74412f61727469636c652d636f7665725f696d6167652d736872696e6b5f3630305f323030302f61727469636c652d636f7665725f696d6167652d736872696e6b5f3630305f323030302f302f313535373738373830353739393f653d3231343734383336343726763d6265746126743d5f33394447695868377236307046704a4176486c756454794652344d726143616e4c325353377532787255", caption="Crop Prediction Model")

    # Subtitles to break down the information
    st.subheader("How It Works")
    st.markdown("""
    The model has been trained on various datasets containing information about soil nutrients, temperature, humidity, 
    pH levels, and rainfall, which influence the growth of different crops. Based on this data, it makes predictions 
    about which crop would thrive under the given conditions.
    """)

    st.subheader("Features")
    st.markdown("""
    - Input the values for key parameters like Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, 
      pH level, and rainfall.
    - Get an instant prediction of the crop.
    - Designed for farmers, researchers, and anyone interested in agricultural prediction.
    """)

    st.subheader("Technologies Used")
    st.markdown("""
    - **Machine Learning Model**: Trained using scikit-learn.
    - **Web Framework**: Streamlit.
    - **Data Processing**: MinMaxScaler for feature scaling.
    """)

    st.subheader("Disclaimer")
    st.markdown("""
    This app is a prototype and should be used as a guide for crop predictions. For accurate and real-world applications, 
    consulting agricultural experts is advised.
    """)

    st.markdown(
        "For more information, you can visit our [project documentation](https://github.com/the-amazing-atharva/Crop-Recommendation).")

    # Add "Made by" text below content in the About section
    st.markdown("---")
    st.markdown(
        "Made by [the-amazing-atharva](https://github.com/the-amazing-atharva)")

elif page == "Recommendation":
    # User Info Guide for the Recommendation Page
    st.subheader("How to Use the Recommendation Tool")
    st.markdown("""
    In this section, you can enter the required parameters based on your agricultural conditions. Here's a guide to 
    what each input means:
    
    - **Nitrogen (N)**: The amount of nitrogen in the soil (affects plant growth).
    - **Phosphorus (P)**: The phosphorus content in the soil (important for root development).
    - **Potassium (K)**: Potassium level in the soil (helps plant immunity and overall growth).
    - **Temperature**: Average temperature (in Celsius) of the growing environment.
    - **Humidity**: Relative humidity percentage in the air.
    - **pH Level**: pH level of the soil (ranges from 0 to 14, with most plants preferring slightly acidic to neutral soil).
    - **Rainfall**: Amount of rainfall (in millimeters) in the area.

    After filling in these details, click the **Predict Crop** button, and the app will suggest the best crop to grow based on the inputs.
    """)

    # Create input fields in two columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        N = st.number_input("Nitrogen (N)", min_value=0.0,
                            max_value=100.0, step=0.1)
        P = st.number_input("Phosphorus (P)", min_value=0.0,
                            max_value=100.0, step=0.1)
        K = st.number_input("Potassium (K)", min_value=0.0,
                            max_value=100.0, step=0.1)
        temperature = st.number_input(
            "Temperature (°C)", min_value=-10.0, max_value=60.0, step=0.1)
        humidity = st.number_input(
            "Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)

    with col2:
        ph = st.number_input("pH Level", min_value=0.0,
                             max_value=14.0, step=0.1)
        rainfall = st.number_input(
            "Rainfall (mm)", min_value=0.0, max_value=500.0, step=0.1)

    # Add some space for better visual separation
    st.markdown("<br>", unsafe_allow_html=True)

    # Prediction Button and Progress Indicator
    if st.button("Predict Crop", key="predict_button", help="Click to predict the crop based on the given parameters"):
        # Show loader (progress bar or spinner)
        with st.spinner("Predicting... please wait."):
            # Simulate some delay for the prediction process (remove this line in production)
            time.sleep(2)

            # Make the prediction
            result = predict_crop(N, P, K, temperature, humidity, ph, rainfall)

        # Show the result after the prediction is made
        st.success(f'The predicted crop is: {result}')

    # Add "Made by" text below content in the Recommendation section
    st.markdown("---")
    st.markdown(
        "Made by [the-amazing-atharva](https://github.com/the-amazing-atharva)")
