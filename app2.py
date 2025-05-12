# import streamlit as st
# import pickle
# import numpy as np
# import time

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

# # Language dictionary to store translations
# translations = {
#     "en": {
#         "title": "Crop Recommendation using Machine Learning",
#         "about_title": "About this Project",
#         "how_it_works": "How It Works",
#         "features": "Features",
#         "technologies_used": "Technologies Used",
#         "disclaimer": "Disclaimer",
#         "go_to_recommendation": "Go to Recommendation Page",
#         "user_guide": "How to Use the Recommendation Tool",
#         "predict_crop_button": "Predict Crop",
#         "prediction_result": "The predicted crop is: ",
#         "made_by": "Made by [the-amazing-atharva](https://github.com/the-amazing-atharva)",
#         "input_guide": "Enter the values for Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity, pH level, and rainfall to predict the crop.",
#         "about_project": "This app helps you predict the best crop for a specific location based on soil and environmental factors.",
#         "how_to_use": "After entering the values, press the 'Predict Crop' button to get the suggested crop based on your conditions."
#     },
#     "hi": {
#         "title": "फसल पूर्वानुमान ऐप",
#         "about_title": "इस परियोजना के बारे में",
#         "how_it_works": "यह कैसे काम करता है",
#         "features": "विशेषताएँ",
#         "technologies_used": "उपयोग की गई तकनीक",
#         "disclaimer": "अस्वीकरण",
#         "go_to_recommendation": "सिफारिश पृष्ठ पर जाएं",
#         "user_guide": "सिफारिश उपकरण का उपयोग कैसे करें",
#         "predict_crop_button": "फसल पूर्वानुमान करें",
#         "prediction_result": "पूर्वानुमानित फसल है: ",
#         "made_by": "बनाया [the-amazing-atharva](https://github.com/the-amazing-atharva)",
#         "input_guide": "फसल का पूर्वानुमान करने के लिए, नाइट्रोजन (N), फास्फोरस (P), पोटैशियम (K), तापमान, आर्द्रता, pH स्तर और वर्षा के मान दर्ज करें।",
#         "about_project": "यह ऐप आपको मिट्टी और पर्यावरणीय कारकों के आधार पर एक विशिष्ट स्थान के लिए सर्वोत्तम फसल का पूर्वानुमान करने में मदद करता है।",
#         "how_to_use": "मान दर्ज करने के बाद, 'फसल पूर्वानुमान करें' बटन दबाएं, और आपके द्वारा दिए गए परिस्थितियों के आधार पर सुझाई गई फसल प्राप्त करें।"
#     },
#     "mr": {
#         "title": "पिक पूर्वानुमान अ‍ॅप",
#         "about_title": "या प्रकल्पाबद्दल",
#         "how_it_works": "हे कसे कार्य करते",
#         "features": "वैशिष्ट्ये",
#         "technologies_used": "वापरलेली तंत्रज्ञान",
#         "disclaimer": "अस्वीकरण",
#         "go_to_recommendation": "सिफारशीसाठी पृष्ठावर जा",
#         "user_guide": "सिफारशी साधनाचा वापर कसा करावा",
#         "predict_crop_button": "पिक पूर्वानुमान करा",
#         "prediction_result": "पूर्वानुमान केलेली पिक: ",
#         "made_by": "बनवले [the-amazing-atharva](https://github.com/the-amazing-atharva)",
#         "input_guide": "पिक पूर्वानुमान करण्यासाठी, नायट्रोजन (N), फास्फोरस (P), पोटॅशियम (K), तापमान, आर्द्रता, pH स्तर आणि पर्जन्याच्या मानांची माहिती भरा.",
#         "about_project": "हे अ‍ॅप तुम्हाला माती आणि पर्यावरणीय घटकांवर आधारित विशिष्ट स्थानासाठी सर्वोत्तम पिकाचे पूर्वानुमान करण्यात मदत करते.",
#         "how_to_use": "माहिती भरल्यानंतर, 'पिक पूर्वानुमान करा' बटणावर क्लिक करा आणि तुमच्या परिस्थितींवर आधारित सुचवलेले पिक प्राप्त करा."
#     }
# }

# # Function to make predictions


# def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
#     input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
#     scaled_input = scaler.transform(input_data)
#     prediction = model.predict(scaled_input)
#     crop_prediction = crop_dict.get(int(prediction[0]), "Unknown crop")
#     return crop_prediction


# # Streamlit interface setup
# st.set_page_config(
#     page_title="Crop Recommendation using Machine Learning", layout="wide")

# # Language selection in the sidebar
# language = st.sidebar.selectbox(
#     "Select Language", ["English", "Hindi", "Marathi"])
# language_code = {"English": "en", "Hindi": "hi", "Marathi": "mr"}[language]

# # Get the translated strings
# text = translations[language_code]

# # Title for the app
# st.title(text["title"])

# # Sidebar Title and Navigation
# st.sidebar.title(text["title"])
# page = st.sidebar.radio("Select a page:", ["About", "Recommendation"])

# # Add the "Made by" text to the sidebar
# st.sidebar.markdown("---")
# st.sidebar.markdown(text["made_by"])

# if page == "About":
#     st.header(text["about_title"])

#     # Display image below the title
#     st.image("https://camo.githubusercontent.com/1bdf0f0577d91db85f8f22e7f8ed20e301498a172302c0d4455b58669114195c/68747470733a2f2f6d656469612e6c6963646e2e636f6d2f646d732f696d6167652f76322f4335313132415145366466527a5633454a74412f61727469636c652d636f7665725f696d6167652d736872696e6b5f3630305f323030302f61727469636c652d636f7665725f696d6167652d736872696e6b5f3630305f323030302f302f313535373738373830353739393f653d3231343734383336343726763d6265746126743d5f33394447695868377236307046704a4176486c756454794652344d726143616e4c325353377532787255", caption="Crop Prediction Model")

#     # Subtitles to break down the information
#     st.subheader(text["how_it_works"])
#     st.markdown(text["about_project"])

#     st.subheader(text["features"])
#     st.markdown("""
#     - Input the values for key parameters like Nitrogen (N), Phosphorus (P), Potassium (K), temperature, humidity,
#       pH level, and rainfall.
#     - Get an instant prediction of the crop.
#     - Designed for farmers, researchers, and anyone interested in agricultural prediction.
#     """)

#     st.subheader(text["technologies_used"])
#     st.markdown("""
#     - **Machine Learning Model**: Trained using scikit-learn.
#     - **Web Framework**: Streamlit.
#     - **Data Processing**: MinMaxScaler for feature scaling.
#     """)

#     st.subheader(text["disclaimer"])
#     st.markdown("""
#     This app is a prototype and should be used as a guide for crop predictions. For accurate and real-world applications,
#     consulting agricultural experts is advised.
#     """)

#     st.markdown(
#         "For more information, you can visit our [project documentation](https://github.com/the-amazing-atharva/Crop-Recommendation).")

#     # Add "Made by" text below content in the About section
#     st.markdown("---")
#     st.markdown(text["made_by"])

# elif page == "Recommendation":
#     # User Info Guide for the Recommendation Page
#     st.subheader(text["user_guide"])
#     st.markdown(text["how_to_use"])
#     st.markdown("""
#     In this section, you can enter the required parameters based on your agricultural conditions. Here's a guide to
#     what each input means:

#     - **Nitrogen (N)**: The amount of nitrogen in the soil (affects plant growth).
#     - **Phosphorus (P)**: The phosphorus content in the soil (important for root development).
#     - **Potassium (K)**: Potassium level in the soil (helps plant immunity and overall growth).
#     - **Temperature**: Average temperature (in Celsius) of the growing environment.
#     - **Humidity**: Relative humidity percentage in the air.
#     - **pH Level**: pH level of the soil (ranges from 0 to 14, with most plants preferring slightly acidic to neutral soil).
#     - **Rainfall**: Amount of rainfall (in millimeters) in the area.

#     After filling in these details, click the **Predict Crop** button, and the app will suggest the best crop to grow based on the inputs.
#     """)

#     # Create input fields in two columns for better layout
#     col1, col2 = st.columns(2)

#     with col1:
#         N = st.number_input("Nitrogen (N)", min_value=0.0,
#                             max_value=100.0, step=0.1)
#         P = st.number_input("Phosphorus (P)", min_value=0.0,
#                             max_value=100.0, step=0.1)
#         K = st.number_input("Potassium (K)", min_value=0.0,
#                             max_value=100.0, step=0.1)
#         temperature = st.number_input(
#             "Temperature (°C)", min_value=-10.0, max_value=60.0, step=0.1)
#         humidity = st.number_input(
#             "Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)

#     with col2:
#         ph = st.number_input("pH Level", min_value=0.0,
#                              max_value=14.0, step=0.1)
#         rainfall = st.number_input(
#             "Rainfall (mm)", min_value=0.0, max_value=500.0, step=0.1)

#     # Add some space for better visual separation
#     st.markdown("<br>", unsafe_allow_html=True)

#     # Input Validation
#     if (N <= 0 or P <= 0 or K <= 0 or temperature <= -10 or humidity <= 0 or ph <= 0 or rainfall <= 0):
#         st.warning("Please enter valid values for all fields!")

#     # Prediction Button and Progress Indicator
#     if st.button(text["predict_crop_button"], key="predict_button", help="Click to predict the crop based on the given parameters"):
#         # Show loader (progress bar or spinner)
#         with st.spinner("Predicting... please wait."):
#             time.sleep(2)  # Simulate some delay for the prediction process

#             # Make the prediction
#             result = predict_crop(N, P, K, temperature, humidity, ph, rainfall)

#             # Show the result after the prediction is made
#             st.success(f'{text["prediction_result"]} {result}')

#     # Add "Made by" text below content in the Recommendation section
#     st.markdown("---")
#     st.markdown(text["made_by"])
