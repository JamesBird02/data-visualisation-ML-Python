import io
from base64 import b64encode
from datetime import datetime as dt

import dash
import dash_auth
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import Dash, State, dcc, html
from dash.dependencies import Input, Output
from dash_bootstrap_templates import load_figure_template
from flask_caching import Cache

from navbar import create_navbar

# Define navbar
nav = create_navbar()

# Set plotly template
load_figure_template("LUX")

# Load data
df = pd.read_csv('Data/AllElec.csv', parse_dates=['Date'], dayfirst=True)
df2 = pd.read_csv('Data/DegDay5.csv', parse_dates=['Date'], dayfirst=True)

# Convert Date column to datetime object
df['Date'] = pd.to_datetime(df['Date'])
df2['Date'] = pd.to_datetime(df2['Date'])

# Set date range for slider
min_date = dt.fromtimestamp(df['Date'].min().timestamp())
max_date = dt.fromtimestamp(df['Date'].max().timestamp())

# Define dropdown options
datasets_type = [
    {'label': 'Electricity (kWh)', 'value': 'AllElec.csv'},
    {'label': 'Gas', 'value': 'TotalGas.csv'},
    {'label': 'Water (m3)', 'value': 'AllW.csv'}
]

# Define default plots
fig = px.line(df, x='Date', title='Graph of Electricity usage')
fig2 = px.line(df2, x='Date', y='Degree Day Cool 5')

# Initialize app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX], suppress_callback_exceptions=True)


# Define sidebar style
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "24rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    "margin-right": "90rem",
}


# Define sidebar
sidebar = html.Div(
    [
        html.H3("GSK Barnard Castle"),
        html.Hr(),
        html.Br(),
        dbc.Nav(
            [
                html.H1('Type', style={'textAlign': 'left', 'font-size': '85%'}),
                dcc.Dropdown(
                    options=datasets_type,
                    value='AllElec.csv',
                    id='dflist'
                ),

                html.Br(),

                html.H1('Area', style={'textAlign': 'left', 'font-size': '85%'}),
                dcc.Dropdown(
                    value='AllElec.csv',
                    id='area'
                ),

                dcc.Store(
                    id='dropdown-2-options'
                ),

                html.Br(),

                html.H1('Choose date range', style={'textAlign': 'left', 'font-size': '85%'}),

                dcc.RangeSlider(
                    id='date-slider',
                    min=df['Date'].min().timestamp(),
                    max=df['Date'].max().timestamp(),
                    step=10,
                    value=[df['Date'].min().timestamp(), df['Date'].max().timestamp()],
                    marks={
                        int(min_date.timestamp()): {'label': min_date.strftime('%d/%b/%Y'), 'style': {'color': '#77b0b1'}},
                        int(max_date.timestamp()): {'label': max_date.strftime('%d/%b/%Y'), 'style': {'color': '#77b0b1'}}
                    },
                    allowCross=False,
                ),
                
                html.Br(),     
                
            ],
                    
         vertical=True,
        pills=True,
        ),

    ], 
    style=SIDEBAR_STYLE,
)
    



#Dash app page layout
def home_layout():
    layout = html.Div([
        nav,
        dbc.Row([
            dbc.Col(),
            dbc.Col(
                html.Br(),
            )
        ]),
        dbc.Row([
            dbc.Col(sidebar),
            dbc.Col([
                dcc.Graph(id='example-graph',figure=fig),
                dcc.Graph(id='example-graph2',figure=fig2)
            ],
            width=9,
            style={
                'margin-left': '15px',
                'margin-top': '7px',
                'margin-right': '15px'
            }
            )
        ]),
    ])
    return layout

#callback to update degree day graph x axis
@app.callback(
    Output("example-graph2", "figure"),
    [Input("date-slider", "value")])

def update_dataset(date_range):    
    df2 = pd.read_csv("Data/DegDay5.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    start_date = dt.fromtimestamp(date_range[0])
    end_date = dt.fromtimestamp(date_range[1])
    fig2 = update_figure(start_date, end_date, df2)
    return fig2

#function to update figure 1 x axis                                                    
def update_figure(start_date, end_date, df):
    # Convert the Date column to a datetime object
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    # Load the dataset based on the selected date range    
    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    filtered_df = filtered_df.sort_values(by='Date')
    # Build the plotly figure using the filtered dataframe
    fig = px.line(filtered_df, x='Date', y=df.columns[0:])
    return fig
    
#callback to update    
@app.callback(
    Output("example-graph", "figure"),
    [Input("area", "value"),
    Input("date-slider", "value")])  

def update_dataset(area, date_range):    
    df = pd.read_csv("Data/" + area)
    df['Date'] = pd.to_datetime(df['Date'])
    start_date = dt.fromtimestamp(date_range[0])
    end_date = dt.fromtimestamp(date_range[1])
    fig = update_figure(start_date, end_date, df)
    return fig

@app.callback(
    Output('area', 'options'),
    Output('area', 'value'),
    Input('dflist', 'value'),
    State('dropdown-2-options', 'data')
)


def update_dropdown_2(dflist_value, dropdown_2_options):
    
    if dflist_value == 'AllElec.csv':
        dropdown_2_options = [{'label': 'All', 'value': 'AllElec.csv'},
                              {'label': 'C block ', 'value': 'CElec.csv'},
                              {'label': 'D block', 'value': 'DElec.csv'},
                              {'label': 'E block ', 'value': 'EElec.csv'},
                              {'label': 'CHP', 'value': 'CHPElec.csv'},
                              {'label': 'Site incomer ', 'value': 'Elecincomer.csv'},]
        dropdown_2_value = 'AllElec.csv'
        
    elif dflist_value == 'TotalGas.csv':
        dropdown_2_options = [{'label': 'Total (Cu Ft)', 'value': 'TotalGas.csv'},
                              {'label': 'CHP (m3)', 'value': 'CHPGas.csv'}]
        dropdown_2_value = 'TotalGas.csv'
        
    else:
        dropdown_2_options = [{'label': 'All ', 'value': 'AllW.csv'},
                              {'label': 'C block ', 'value': 'CW.csv'},
                              {'label': 'D block', 'value': 'DW.csv'},
                              {'label': 'K block ', 'value': 'KW.csv'},
                              {'label': 'J block', 'value': 'JW.csv'},
                              {'label': 'South site ', 'value': 'SouW.csv'},
                              {'label': 'Main site ', 'value': 'mainW.csv'}
                              ]
        
        dropdown_2_value = 'AllW.csv'
    
    # Update the options for dropdown 2 using the dcc.Store component
    return dropdown_2_options, dropdown_2_value




