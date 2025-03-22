import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier, LazyRegressor
import chardet
from tqdm import tqdm
import time


# Function to detect file encoding and load the CSV
def detect_encoding(upload_file):
    raw_data = upload_file.read()
    upload_file.seek(0)  # Reset file pointer after reading
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    try:
        dataframe = pd.read_csv(upload_file, encoding=encoding)
        return dataframe
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


# Main inference function
def make_inference(target, type, train_size, dataframe):
    if type == "classification":
        if st.session_state.get('classification', False):
            st.header("You Already Performed Classification", divider='blue')
            st.dataframe(st.session_state['models_cls'])

            if st.button("Recompute Classification", use_container_width=True, type='primary'):
                st.session_state['classification'] = False
        else:
            st.header("You are performing Classification", divider='green')
            try:
                x = dataframe.drop(target, axis=1)
                y = dataframe[target]
                x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=42)

                # Add a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)

                # Simulate progress with tqdm-like behavior
                models, predictions = None, None
                for i in tqdm(range(100), desc="Training Models"):
                    # Update progress bar
                    progress_bar.progress(i + 1)
                    status_text.text(f"Training Models: {i + 1}% Complete")
                    # Simulate work being done (e.g., fitting the model)
                    if i == 0:
                        models, predictions = clf.fit(x_train, x_test, y_train, y_test)
                    time.sleep(0.01)  # Simulate delay

                # Store results in session state
                st.session_state['classification'] = True
                st.session_state['models_cls'] = models

                # Display results
                st.subheader("Classification Results", divider='blue')
                st.dataframe(models)

                # Clear progress text
                status_text.text("Training Complete!")

            except Exception as e:
                st.error(f"Error during classification: {e}")

    elif type == "regression":
        if st.session_state.get('regression', False):
            st.header("You Already Performed Regression", divider='blue')
            st.dataframe(st.session_state['models_reg'])

            if st.button("Recompute Regression", use_container_width=True, type='primary'):
                st.session_state['regression'] = False
        else:
            st.header("You are performing Regression", divider='red')
            try:
                x = dataframe.drop(target, axis=1)
                y = dataframe[target]
                x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=42)

                # Add a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)

                # Simulate progress with tqdm-like behavior
                models, predictions = None, None
                for i in tqdm(range(100), desc="Training Models"):
                    # Update progress bar
                    progress_bar.progress(i + 1)
                    status_text.text(f"Training Models: {i + 1}% Complete")
                    # Simulate work being done (e.g., fitting the model)
                    if i == 0:
                        models, predictions = reg.fit(x_train, x_test, y_train, y_test)
                    time.sleep(0.01)  # Simulate delay

                # Store results in session state
                st.session_state['regression'] = True
                st.session_state['models_reg'] = models

                # Display results
                st.subheader("Regression Results", divider='blue')
                st.dataframe(models)

                # Clear progress text
                status_text.text("Training Complete!")

            except Exception as e:
                st.error(f"Error during regression: {e}")


# Streamlit interface
st.title("Model Inference Tool")

# Sidebar for configuration
upload_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
if upload_file:
    uploaded_dataframe = detect_encoding(upload_file)
    if uploaded_dataframe is not None:
        st.sidebar.header("Configuration", divider='blue')

        # Target column selection
        target = st.sidebar.selectbox(
            "Select the target column",
            options=uploaded_dataframe.columns
        )

        # Training dataset proportion
        train_size = st.sidebar.slider(
            "Select the proportion for the training dataset",
            min_value=0.1,
            max_value=0.9,
            value=0.7,
            step=0.1
        )

        # Infer type (classification or regression)
        infer_type = st.sidebar.selectbox(
            "Inference Type",
            ["classification", "regression"]
        )

        # Checkbox to confirm settings
        if st.sidebar.checkbox("Confirm Settings"):
            if target and train_size and infer_type:
                make_inference(target, infer_type, train_size, uploaded_dataframe)
            else:
                st.sidebar.warning("Please provide all required inputs.")
