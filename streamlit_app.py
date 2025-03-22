import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lazypredict
import chardet
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier, LazyRegressor


def detect_encoding(upload_file):
    # Use chardet to detect encoding and read CSV correctly with the detected encoding
    raw_data = upload_file.read()
    upload_file.seek(0)  # Reset file pointer to the beginning after reading
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    try:
        dataframe = pd.read_csv(upload_file, encoding=encoding)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None
    return dataframe


def make_inference(target, type, trainSize, dataframe):
    if type == "classification":
        st.header("You are performing Classification", divider='green')
        try:
            x = dataframe.drop(target, axis=1)
            y = dataframe[target]
            xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size=trainSize, random_state=42)
            clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
            models, predictions = clf.fit(xTrain, xTest, yTrain, yTest)
            st.dataframe(models)
        except Exception as e:
            st.error(f"Error during classification: {e}")
    elif type == "regression":
        st.header("You are performing Regression", divider='red')
        try:
            x = dataframe.drop(target, axis=1)
            y = dataframe[target]
            xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size=trainSize, random_state=42)
            reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
            models, predictions = reg.fit(xTrain, xTest, yTrain, yTest)
            st.dataframe(models)
        except Exception as e:
            st.error(f"Error during regression: {e}")


# Streamlit interface
upload_file = st.sidebar.file_uploader("Upload file of type csv", type=["csv"])
if upload_file:
    uploaded_dataframe = detect_encoding(upload_file)
    if uploaded_dataframe is not None:
        st.sidebar.header("Some Minimal Configuration", divider='blue')

        # Allow user to select the target column
        target = st.sidebar.selectbox("Select the target column", options=uploaded_dataframe.columns)

        # Allow user to select the training dataset proportion
        train = st.sidebar.slider("Select the percent of portion for training dataset", min_value=0.1, max_value=0.9, value=0.7)

        # Allow user to select the inference type
        type = st.sidebar.selectbox("Infer type", ["classification", "regression"])

        # Checkbox to confirm settings
        if st.checkbox("Fix the above settings"):
            if target and train and type:
                make_inference(target, type, train, uploaded_dataframe)
            else:
                st.sidebar.info("Target column, proportion for slider, and infer type must be given")
