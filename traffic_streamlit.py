import streamlit as st
import pandas as pd
import pickle
import warnings
import numpy as np
warnings.filterwarnings('ignore')

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
sample_df.rename(columns={'holiday': 'Choose if today is a designated holiday',
                          'temp': 'Average temperature in Kelvin',
                          'rain_1h': 'Amount of rain that occured in the hour(mm)',
                          'snow_1h': 'Amount of snow that occured in the hour(mm)',
                          'clouds_all': 'Percentage cloud cover',
                          'weather_main': 'Choose the current weather',
                          'month': 'Month of year',
                          'weekday': 'Day of week',
                          'hour': 'Hour of day'}
                ,inplace=True)

st.sidebar.image('traffic_sidebar.jpg', caption="Traffic Volume Predictor")
st.sidebar.header("Input Features")
st.sidebar.write("You can either upload your data file or manually enter input features.")

with st.sidebar.expander("Option 1: Upload CSV File"):
    file_upload = st.file_uploader(label="Upload a CSV file")
    st.write("Sample Data Format for Upload")
    st.write(default_df.head(5))
    st.warning(body = " *Ensure your uploaded file has the same column names and data types as shown above.*", icon = "⚠️")

with st.sidebar.expander("Option 2: Fill Out Form"):
    for col in sample_df.columns:
        if col == 'traffic_volume':
            pass
        elif col == 'Hour of day':
            options = list(range(24)) #google gemini generated the list(range(24)) snippet
            sample_df[col] = st.radio(label=col, options=options)
        elif col == 'Month of year':
            options = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
            sample_df[col] = st.radio(label=col, options=options)
        elif col == 'Day of week':
            options = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        else:
            if (sample_df[col].dtype != 'object'):
                sample_df[col] = st.slider(label=col, 
                                                min_value=sample_df[col].min(),
                                                max_value=sample_df[col].max())
            elif(sample_df[col].dtype == 'object'):
                options = sample_df[col].unique()
                sample_df[col] = st.radio(label=col, options=options)