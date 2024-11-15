import streamlit as st
import pandas as pd
import pickle
import warnings
import numpy as np
warnings.filterwarnings('ignore')

if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = False #Used gemini to help with the form submission
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False

st.markdown("<div style = 'color:	#FFD700; text-align: center; font-size: 42px; font-weight: bold; '>Traffic Volume Predictor</div>",
            unsafe_allow_html=True)
st.markdown("<div style = 'text-align: center; font-size: 26px'> Utilize our advanced Machine Learning application to predict traffic volume. </div>",
            unsafe_allow_html=True)
#Used gemini to assist with CSS formatting here
st.image('traffic_image.gif', use_column_width = True)

@st.cache_resource
def load_model():
    xg_pickle = open('XGBoost_pickle.pkl', 'rb') 
    xg_model = pickle.load(xg_pickle)
    xg_pickle.close()

    mapie_pickle = open('mapie_pickle.pkl', 'rb') 
    mapie_model = pickle.load(mapie_pickle) 
    mapie_pickle.close()
    
    return xg_model, mapie_model

@st.cache_data
def load_resources():
    default_df = pd.read_csv('Traffic_Volume.csv')
    default_df['month'] = pd.to_datetime(default_df['date_time']).dt.strftime('%B')
    default_df['weekday'] = pd.to_datetime(default_df['date_time']).dt.strftime('%A') 
    default_df['hour'] = pd.to_datetime(default_df['date_time']).dt.hour
    default_df['hour'] = default_df['hour'].astype(str)
    default_df.drop(columns=['date_time'], inplace=True)
    return default_df

xg_model, mapie_model = load_model()
default_df = load_resources()
sample_df = default_df.copy()
sample_df['holiday'] = sample_df['holiday'].replace(np.NAN, 'Not a holiday')
col_dict = {'holiday': 'Choose if today is a designated holiday',
                          'temp': 'Average temperature in Kelvin',
                          'rain_1h': 'Amount of rain that occured in the hour(mm)',
                          'snow_1h': 'Amount of snow that occured in the hour(mm)',
                          'clouds_all': 'Percentage cloud cover',
                          'weather_main': 'Choose the current weather',
                          'month': 'Month of year',
                          'weekday': 'Day of week',
                          'hour': 'Hour of day'}
sample_df.rename(columns=col_dict
                ,inplace=True)

st.sidebar.image('traffic_sidebar.jpg', caption="Traffic Volume Predictor")
st.sidebar.header("Input Features")
st.sidebar.write("You can either upload your data file or manually enter input features.")

with st.sidebar.expander("Option 1: Upload CSV File"):
    def upload_file():
        st.session_state.file_uploaded = True
        st.session_state.form_submitted = False
    file_upload = st.file_uploader(label="Upload a CSV file", on_change=upload_file)
    st.write("Sample Data Format for Upload")
    st.write(default_df.head(5))
    st.warning(body = " *Ensure your uploaded file has the same column names and data types as shown above.*", icon = "⚠️")

with st.sidebar.expander("Option 2: Fill Out Form"):
    for col in sample_df.columns:
        if col == 'traffic_volume':
            pass
        elif col == 'Hour of day':
            options = list(range(24)) #google gemini generated the list(range(24)) snippet
            sample_df[col] = str(st.radio(label=col, options=options))
        elif col == 'Month of year':
            options = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
            sample_df[col] = st.radio(label=col, options=options)
        elif col == 'Day of week':
            options = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            sample_df[col] = st.radio(label=col, options=options)
        else:
            if (sample_df[col].dtype != 'object'):
                sample_df[col] = st.slider(label=col, 
                                                min_value=sample_df[col].min(),
                                                max_value=sample_df[col].max())
            elif(sample_df[col].dtype == 'object'):
                options = sample_df[col].unique()
                sample_df[col] = st.radio(label=col, options=options)
    def submit_form():
        st.session_state.form_submitted = True
        st.session_state.file_uploaded = False
    st.button(label = 'Submit', on_click=submit_form) #Used gemini to debug here

if not (st.session_state.form_submitted or st.session_state.file_uploaded):
    st.info(body = " *Please choose a data input method to proceed.*", icon="ℹ️")

elif st.session_state.form_submitted: 
    st.success(body = " *Form submitted successfully.*", icon="✅")
    alpha = st.slider(label = 'Select alpha value for prediction intervals', min_value = 0.01, max_value = 0.50)
    st.header("Predicting Traffic Volume...")

    flipped_dict = {value: key for key, value in col_dict.items()} #Used gemini to help flip dictionary
    sample_df = sample_df.head(1)
    sample_df['Choose if today is a designated holiday'] = sample_df['Choose if today is a designated holiday'].replace('Not a holiday', np.NAN)
    sample_df.rename(columns=flipped_dict
            ,inplace=True)
    append_df = pd.concat([sample_df, default_df])
    encode_dummy_df = pd.get_dummies(append_df, drop_first=True)
    flipped_dict = {value: key for key, value in col_dict.items()} #Used gemini to help flip dictionary
    user_encoded_df = encode_dummy_df.head(1).drop(columns=['traffic_volume'])
    prediction, intervals = mapie_model.predict(user_encoded_df, alpha = alpha)
    pred_value = prediction[0]
    lower_limit = intervals[:, 0]
    upper_limit = intervals[:, 1]

    # Ensure limits are within [0, 1]
    lower_limit = max(0, lower_limit[0][0])    
    upper_limit = max(0, upper_limit[0][0])

    st.metric(label = "Predicted Traffic Volume", value = f"{pred_value:.2f}")
    st.write(f"**Prediction Interval** [{100 * (1-alpha)}%]: [{lower_limit:.2f}, {upper_limit:.2f}]")

elif (st.session_state.file_uploaded & (file_upload is not None)):
    st.success(body = " *CSV file uploaded successfully.*", icon="✅")
    alpha = st.slider(label = 'Select alpha value for prediction intervals', min_value = 0.01, max_value = 0.50)    

    file_upload = pd.read_csv(file_upload)
    user_upload = pd.DataFrame(file_upload)
    append_df = pd.concat([default_df, user_upload])
    append_df['hour'] = append_df['hour'].astype(str)
    encode_dummy_df = pd.get_dummies(append_df, drop_first=True)
    user_encoded_df = encode_dummy_df.tail(user_upload.shape[0]).drop(columns=['traffic_volume'])
    prediction_1, intervals_1 = mapie_model.predict(user_encoded_df, alpha = alpha)

    datafreme = pd.DataFrame()
    for i in range(0, user_upload.shape[0]):
        pred_value_1 = prediction_1[i]
        lower_limit_1 = intervals_1[i, 0]
        upper_limit_1 = intervals_1[i, 1]
        lower_limit_1 = max(0, lower_limit_1[0])
        upper_limit_1 = max(0, upper_limit_1[0])

        diction = {'Predicted Volume': pred_value_1, 
                   'Lower Limit': lower_limit_1, 
                   'Upper Limit': upper_limit_1,
                   'Alpha': alpha}
        
        diction = pd.DataFrame(diction, index = [0])
        datafreme = pd.concat([datafreme, diction])
    datafreme = datafreme.reset_index()
    final_df = pd.merge(user_upload, datafreme[['Predicted Volume', 
                                                'Lower Limit',
                                                'Upper Limit',
                                                'Alpha']], left_index=True, right_index=True)
    final_df = final_df.style.format({'Predicted Volume': '{:.1f}', 'Lower Limit': '{:.1f}', 'Upper Limit': '{:.1f}',})
    st.write(final_df)


if st.session_state.form_submitted or st.session_state.file_uploaded:
    st.subheader("Model Performance and Inference")
    tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", 
                                "Histogram of Residuals", 
                                "Predicted Vs. Actual", 
                                "Coverage Plot"])
    with tab1:
        st.write("### Feature Importance")
        st.image('feature_imp.svg')
        st.caption("Relative importance of features in prediction.")
    with tab2:
        st.write("### Histogram of Residuals")
        st.image('residuals.svg')
        st.caption("Distribution of residuals to evaluate prediction quality.")
    with tab3:
        st.write("### Plot of Predicted Vs. Actual")
        st.image('scatter_plot.svg')
        st.caption("Visual comparison of predicted and actual values.")
    with tab4:
        st.write("### Coverage Plot")
        st.image('coverage.svg')
        st.caption("Range of predictions with confidence intervals.")