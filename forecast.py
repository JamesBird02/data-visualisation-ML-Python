from datetime import datetime, timedelta
from functools import lru_cache

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import Dash, State, callback_context, dcc, html
from dash.dependencies import Input, Output
from flask_caching import Cache

from app import app
from ML import DeepModelTS
from navbar import create_navbar

#intitialise components and dataframe
df = pd.read_csv('Data/TotalGas.csv')

fig = {}

nav = create_navbar()

cache = Cache(app.server, config={
    'CACHE_TYPE': 'simple'
})

datasets_type = [
    {'label': 'Electricity (kWh)', 'value': 'AllElec.csv'},
    {'label': 'Gas (Cu Ft)', 'value': 'TotalGas.csv'},
    {'label': 'Water (m3)', 'value': 'AllW.csv'}
]

#Sidebar
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "24rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    "margin-right": "90rem"
    
}

sidebar = html.Div(
    [
        html.H3("GSK Barnard Castle"),
        html.Hr(),
        html.Br(),
        dbc.Nav([dbc.Button("Load Model", id="load-model-button", color="primary", className="mb-3")
],
         vertical=True,
        pills=True,)
 
    ],
    style=SIDEBAR_STYLE,
)

#dash page layout
def forecast_layout():
    layout = html.Div([
        nav,
        dbc.Row([
            dbc.Col(),
            dbc.Col(
                html.Br(),
                width=9,
                style={
                    'margin-left': '7px',
                    'margin-top': '7px'
                }
            )
        ]),
        dbc.Row([
            dbc.Col(sidebar),
            dbc.Col(
                 dcc.Loading(
                    id="loading-forecast",
                    children=dcc.Graph(id='forecast-graph', figure=fig),
                    type="default"
                ),
                width=9,
                style={
                    'margin-left': '15px',
                    'margin-top': '7px',
                    'margin-right': '15px'
                }
            )
        ]),
        dbc.Row([
            dbc.Col(html.Div())
        ])
    ])
    return layout

#creating the forecast graph
@lru_cache(maxsize=None)
def load_forecast():    
     
        # Initiating the class
        deep_learner = DeepModelTS(
            data = df,
            Y_var = df.columns[1],
            lag = 40,
            LSTM_layer_depth = 48,
            epochs = 100,
            batch_size = 16,
            train_test_split = 0.2
        )

        model = deep_learner.LSTModel()

        # Defining the lag that we used for training of the model 
        lag_model = 40
        
        # Getting the last period
        ts = df[df.columns[1]].tail(lag_model).values.tolist()
        
        # Creating the X matrix for the model
        X, _ = deep_learner.create_X_Y(ts, lag=lag_model)
        
        # Getting the forecast
        yhat = model.predict(X)
        yhat = deep_learner.predict()
        
        # Constructing the forecast dataframe
        fc = df.tail(len(yhat)).copy()
        fc.reset_index(inplace=True)
        fc['forecast'] = yhat

        # Creating the model using full data and forecasting n steps 
        aheaddeep_learner = DeepModelTS(
        data=df,
        Y_var=df.columns[1],
        lag=40,
        LSTM_layer_depth=24,
        epochs=100,
        train_test_split=0
        )
        
        # Fitting the model
        deep_learner.LSTModel()
        
        # Forecasting n steps ahead
        n_ahead = 30
        yhat = deep_learner.predict_n_ahead(n_ahead)
        yhat = [y[0][0] for y in yhat]

        # Constructing the forecast dataframe
        fc = df.tail(50).copy() 
        fc['type'] = 'original'
        last_date = max(fc['Date'])
        last_date = datetime.strptime(last_date, '%Y-%m-%d %H:%M:%S') # convert last_date to datetime object
        hat_frame = pd.DataFrame({
            'Date': [last_date + timedelta(days=x + 1) for x in range(n_ahead)], 
            df.columns[1]: yhat,
            'type': 'forecast'
        })

        fc = fc.append(hat_frame)
        fc.reset_index(inplace=True, drop=True)

        fig = px.line(
            fc,
            x='Date',
            y=df.columns[1],
            color='type',
            labels={'Date': 'Date', df.columns[1]: 'Usage kWh'},
            title='GSK Barnard Castle forecasted total gas usage'
        )

        return fig


#callback to load the forecast graph 
@app.callback(
    Output('forecast-graph', 'figure'),
    Input('load-model-button', 'n_clicks'),
)
def update_figure(n_clicks):
    return load_forecast()
    