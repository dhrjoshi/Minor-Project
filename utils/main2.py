import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load trained model
model = joblib.load('../model/fraud_detection_model.pkl')

# Load scaler and fit on known data
scaler = StandardScaler()

def preprocess_input(data):
    """ Preprocess user input """
    data[['Amount', 'Time']] = scaler.fit_transform(data[['Amount', 'Time']])
    return data

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Enter Data","Upload File"])

#Main Page
if(app_mode=="Home"):
    st.header("CREDIT CARD FRAUD DETECTION SYSTEM")
    image_path = "../home_page.jpeg"  # Make sure this image exists in your directory
    st.image(image_path, use_column_width=True)
    
    st.markdown("""
    Welcome to the Credit Card Fraud Detection System! 💳🔍

    Our mission is to provide a reliable, fast, and intelligent system to detect fraudulent credit card transactions. Upload your dataset, and let our machine learning models analyze and predict any suspicious activity. Together, we can build a safer digital financial environment!

    ### How It Works
    1. **Upload Data:** Navigate to the **Fraud Detection** page and upload your credit card transaction CSV file.
    2. **Analysis:** Our system processes the data using advanced machine learning models like Random Forest, XGBoost, and others.
    3. **Results:** Instantly view predictions highlighting potential fraud cases, along with confidence scores and visual analysis.

    ### Why Choose Us?
    - **Accuracy:** We employ SMOTE + Tomek techniques and feature selection to enhance prediction precision.
    - **User-Friendly:** An intuitive interface ensures even non-technical users can operate the system smoothly.
    - **Fast and Efficient:** Experience real-time analysis and results within seconds.
    - **Secure:** Data privacy is a top priority. Your uploaded data stays safe.

    ### Get Started
    Use the **Fraud Detection** page in the sidebar to upload your transaction data and experience the power of intelligent fraud detection.

    ### About Us
    Visit the **About** page to learn more about our project’s goals and methodology.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
            #### About the Dataset
            The dataset used in this project is a publicly available credit card transactions dataset that contains both legitimate and fraudulent transactions. It includes over **284,807 transactions** made by European cardholders over two days in September 2013. Out of these, **492 are fraudulent**, making the dataset highly imbalanced — a common challenge in fraud detection tasks.

            Each transaction is described by 30 features, most of which are the result of a **PCA transformation** to preserve confidentiality. Only two features — **Time** and **Amount** — are not transformed. The target variable is **Class**, where **1 indicates a fraudulent transaction** and **0 indicates a legitimate one**.

            #### Content
            1. `Time` - Seconds elapsed between each transaction and the first transaction in the dataset.
            2. `Amount` - Transaction amount.
            3. `V1-V28` - Principal components obtained using PCA.
            4. `Class` - Target variable (0 = Legit, 1 = Fraud)

            The dataset is split into:
            - **Training set**: Used for model training and validation.
            - **Testing set**: Used to evaluate model performance on unseen data.

            #### Purpose of the Project
            This project aims to build a **robust machine learning-based system** to automatically detect fraudulent transactions with high accuracy. It incorporates **data balancing techniques (SMOTE + Tomek links)**, **feature selection**, and **hyperparameter tuning** to improve performance. The system is also deployed using **Streamlit** for a seamless web interface that allows users to interact with the model in real time.
            """)

# Enter Data Page
elif(app_mode=="Enter Data"):
    # Streamlit UI
    st.title("Credit Card Fraud Detection")
    st.write("Enter transaction details to check if it's fraudulent or not.")

    # User input fields
    amount = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")
    time = st.number_input("Time since first transaction (seconds)", min_value=0.0, format="%.2f")
    v_features = [st.number_input(f"V{i}", value=0.0, format="%.4f") for i in range(1, 29)]

    # Convert to DataFrame
    input_data = pd.DataFrame([[time, *v_features, amount]], columns=['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount'])

    if st.button("Predict Fraud"):    
        processed_data = preprocess_input(input_data)
        prediction = model.predict(processed_data)[0]
        
        if prediction == 0:
            st.error("🚨 Fraudulent Transaction Detected!")
        else:
            st.success("✅ Legitimate Transaction")
    

# #Prediction Page
elif(app_mode=="Upload File"):
    st.header("Credit Card Fraud Detection")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")

            st.write(f"### Fraudulent Transactions:")
            st.dataframe(df.head(2))
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
    else:
        st.info("Please upload a CSV file.")