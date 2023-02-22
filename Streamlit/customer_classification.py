import streamlit as st
import pandas as pd 
import pickle5 as pickle
import numpy as np

st.write(" # Customer Classification" )

st.subheader("Predicting whether the customer will sign up for the Delivery Club or not")

#st.write(" #### Enter the values for different features to make predictionss: ")

st.sidebar.header("User Input Features")

uploaded_file = st.sidebar.file_uploader("Upload your csv file", type = ["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        distance_from_store = st.sidebar.slider('Distance From Store', min_value = 0.0, max_value = 10.0)
        transaction_count = st.sidebar.slider('Transaction Count', min_value = 0, max_value = 500, step = 1)
        total_items = st.sidebar.slider('Total Items Purchased', min_value = 0, max_value = 500, step = 1)
        avg_basket_value = st.sidebar.number_input('Average Basket Value')
        product_area_count = st.sidebar.slider('Product Area Count')
        credit_score = st.sidebar.slider('Credit Score', min_value = 0.0, max_value = 100.0)
        total_sales = st.sidebar.number_input('Total Sales')
        gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))


        data = {'distance_from_store': distance_from_store, 'transaction_count': transaction_count, 'total_items': total_items,
        'avg_basket_value': avg_basket_value, 'product_area_count': product_area_count, 'credit_score': credit_score,
        'total_sales': total_sales, 'gender': gender}

        features = pd.DataFrame(data, index=[0])
        return features 

input_df = user_input_features()

encode = ['gender']
for col in encode:
    dummy = pd.get_dummies(input_df[col], prefix = col)
    input_df = pd.concat([input_df,dummy], axis=1)
    del input_df[col]

input_df = input_df[:1]


st.subheader('User Input Features')

if uploaded_file is None:
    st.write(input_df)
else:
    st.write('Awaiting csv file to be uploaded')
    st.write(uploaded_file)

load_clf = pickle.load(open('Streamlit/RandomForest_Classifier.pkl', 'rb'))

prediction = load_clf.predict(input_df)
prediction_proba = load_clf.predict_proba(input_df)

st.subheader('Prediction')
customer_classification = np.array([0,1])
st.write(customer_classification[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)


st.subheader("Conclusions")

if prediction == 1:
    st.write('#### Customer will sign up for the Delivery Club')
else:
    st.write('#### Customer will not sign up for the Delivery Club')
