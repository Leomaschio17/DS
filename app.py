
import pandas as pd
import numpy as np
import os
import math
from math import sqrt
from sklearn import preprocessing
from datetime import date
import calendar
import datetime as dt
import streamlit as st


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

## copmmand to suppress chain warnings
pd.options.mode.chained_assignment = None  # default='warn'

## plot
from plotly.offline import init_notebook_mode, iplot
from plotly.graph_objs import *
import plotly.express as px
import plotly.graph_objects as go

import pickle



## load dataframe

## get file path
path = os.getcwd()
filename = 'factory_20240131.csv'

path = path + '\\' + filename

## load dataset (separator is escape code \t)
data = pd.read_csv(path, sep = "\t")


## load dataframe

## get file path
path = os.getcwd()
filename = 'factory_20240131.csv'

path = path + '\\' + filename

## load dataset (separator is escape code \t)
data = pd.read_csv(path, sep = "\t")
data['time_batch'] = pd.to_datetime(data['time_batch'], format = '%d-%m-%Y %H:%M')
data.loc[len(data)] = np.NaN
data.loc[len(data)-1,'time_batch'] = data['time_batch'][len(data)-2] + pd.Timedelta(hours=0, minutes=10, seconds=0)
data['hour'] = data['time_batch'].dt.hour
data['date'] = data['time_batch'].dt.date
data['time'] = data['time_batch'].dt.time
cnc_average_kw_median = data['cnc_average_kw'].median()
data.loc[data['cnc_average_kw'] > 175,'cnc_average_kw'] = cnc_average_kw_median
data['consumption'] = data['cnc_average_kw'] + data['work_stations_average_kw']
data.drop(columns={'cnc_average_kw', 'work_stations_average_kw'}, inplace=True)

df = data[['time_batch', 'consumption',
           'temp_CNC_1', 'humidity_CNC_1', 'temp_CNC_2', 'humidity_CNC_2',
           'temp_CNC_3', 'humidity_CNC_3', 'temp_CNC_4', 'humidity_CNC_4',
           'temp_CNC_5', 'humidity_CNC_5', 'temp_CNC_7', 'humidity_CNC_7', 
           'temp_CNC_8', 'humidity_CNC_8', 'temp_CNC_9', 'humidity_CNC_9', 
           # 'temp_CNC_6', 'humidity_CNC_6',
           'temp_outside', 'press_mm_hg_outside', 'humidity_outside', 
           'windspeed_outside', 'visibility_outside', 'dewpoint_outside', 
           'hour'
          ]].sort_values(by = 'time_batch').copy()

## create new variables
df['internal_temp'] = (df['temp_CNC_1'] + df['temp_CNC_2'] + df['temp_CNC_3'] + df['temp_CNC_4'] + df['temp_CNC_5'] + df['temp_CNC_7'] + df['temp_CNC_8'] + df['temp_CNC_9'])/8
df['internal_hum'] = (df['humidity_CNC_1'] + df['humidity_CNC_2'] + df['humidity_CNC_3'] + df['humidity_CNC_4'] + df['humidity_CNC_5'] + df['humidity_CNC_7'] + df['humidity_CNC_8'] + df['humidity_CNC_9'])/8
df['temp_diff'] = df['internal_temp'] - df['temp_outside']
df['hum_diff'] = df['internal_hum'] - df['humidity_outside']


## convert hour into a circular variable
def sin_x(x):
    value = math.sin(math.pi * (x+0.5)/12)
    return value

def cos_x(x):
    value = math.cos(math.pi * (x+0.5)/12)
    return value
    
df['sin_t'] = df['hour'].apply(lambda x: sin_x(x))
df['cos_t'] = df['hour'].apply(lambda x: cos_x(x))


## define function to calculate averages of previous observations.
def roller(frame):
    rol = frame.copy()
    for el in rol.columns:
        ## we don't want to calculate rolling averages on these variables
        if el in ['time_batch','consumption','sin_t','cos_t']:
            continue
        else:
            ## setting window. every hour has 6 periods (one every 10 minutes). We must add one because rolling starts to count from the 
            ## current observation. With closed = 'left' we exclude it.
            rol[el] = rol[el].rolling(min_periods=1, window=7, closed='left').mean()
            
 
    return rol

## create rolling dataframe
rolled_df = roller(df)
rolled_df.reset_index(inplace=True, drop=True)

## drop columns not useful
rolled_df.drop(columns={'time_batch', 'hour'
                        , 'internal_temp', 'internal_hum'
                       }, index = {0}, inplace=True)

## normalization
x = rolled_df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
norm_df = min_max_scaler.fit_transform(x)
norm_df = pd.DataFrame(norm_df, columns = rolled_df.columns)
norm_df.drop(columns = 'consumption', inplace=True)

## Get dependent variable
y = rolled_df['consumption']

## Get indipendent variables
X = norm_df


## Make predictions
filename = 'RandomForest_trained.sav'
rf = pickle.load(open(filename, 'rb'))
predictions = rf.predict(X.values)
data.drop(index = {0}, inplace=True)
data['predicted_consumption'] = predictions.round()


st.set_page_config(layout="wide")
st.title('Factory Consumption Prediction \n')

with st.sidebar:
    date = data['date'][len(data)]
    d = st.date_input("Select date of analysis", date)
    st.write("Date of analysisis selected:", d)
    
    time = data['time'][len(data)]
    t = st.time_input("Select time", time, step = 600)
    st.write("Time selected:", t)

    window = st.slider("Select time window", 1, 4, 1)
    st.write("Time window selected (in hours):", window)




with st.container():
    
    to_plot = data[['time_batch', 'predicted_consumption', 'consumption']]
            
    max_d = dt.datetime.combine(d,t)
    min_d = max_d - pd.Timedelta(hours=window, minutes=0, seconds=0)
            
    to_plot = to_plot[(to_plot['time_batch'] >= min_d) & (to_plot['time_batch'] <= max_d)]
    
    max_v = max(to_plot['predicted_consumption'].max(), to_plot['consumption'].max()) + 10
    min_x = to_plot['time_batch'].min() - pd.Timedelta(hours=0, minutes=5, seconds=0)
    max_x = to_plot['time_batch'].max() + pd.Timedelta(hours=0, minutes=5, seconds=0)
        
            
            
        ## plotting
    fig = go.Figure()
        
    fig.add_trace(go.Scatter(x=to_plot['time_batch'], y=to_plot['consumption'], name='Actual',
                                     line=dict(color='navy', width=2)))
        
    fig.add_trace(go.Scatter(x=to_plot['time_batch'], y=to_plot['predicted_consumption'], name='Predicted',
                                     line=dict(color='firebrick', width=2, dash='dash')))
    
    # fig.update_traces(marker_size=10)
    
            
    fig.update_layout(title='Actual vs Predicted Factory Consumptions',
                          title_x=0.05,
                          title_y=0.9,
                              xaxis_title='Time',
                              yaxis_title='Consumption',
                              yaxis_range=[0,max_v],
                              xaxis_range=[min_x,max_x],
                              font=dict(size=18, color= 'black'),
                              height = 550
                             # template = "simple_white"
                         )

    
    st.plotly_chart(fig, theme=None)
