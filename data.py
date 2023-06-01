import time
from datetime import datetime as dt

import dash
import dash_auth
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import State, callback_context, dash_table, dcc, html
from dash.dependencies import Input, Output
from dash_bootstrap_templates import load_figure_template
from PIL import Image

from app import app
from navbar import create_navbar

nav = create_navbar()

df = pd.read_csv('Data\AllElec.csv', parse_dates=['Date'], dayfirst=True)

app = dash.Dash(__name__)

data= df.to_dict('records')

datasets_type = [
    {'label': 'Electricity (kWh)', 'value': 'AllElec.csv'},
    {'label': 'Gas', 'value': 'TotalGas.csv'},
    {'label': 'Water (m3)', 'value': 'AllW.csv'}
]

df['Date'] = pd.to_datetime(df['Date'])
min_date = dt.fromtimestamp(df['Date'].min().timestamp())
max_date = dt.fromtimestamp(df['Date'].max().timestamp())

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
                    options= [{'label': 'All', 'value': 'AllElec.csv'},
                              {'label': 'C block ', 'value': 'CElec.csv'},
                              {'label': 'D block', 'value': 'DElec.csv'},
                              {'label': 'E block ', 'value': 'EElec.csv'},
                              {'label': 'CHP', 'value': 'CHPElec.csv'},
                              {'label': 'Site incomer ', 'value': 'Elecincomer.csv'}],
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
            ],
                    
         vertical=True,
        pills=True,),
        
    ], 
    style=SIDEBAR_STYLE,
)

def data_layout():
    layout = html.Div(
        children=[
            dbc.Row(),
            nav,
            dbc.Col(),
            sidebar,
            dbc.Row(
                dbc.Col(
                    dash_table.DataTable(
                        id='datatable',
                        data=df.to_dict('records'),
                        columns=[{"name": i, "id": i} for i in df.columns],
                        style_data={
                            "whiteSpace": "normal",
                            "overflowX": "auto",
                        },
                        css=[{"selector": "table", "rule": "table-layout: fixed"}],
                        style_cell={
                            "width": "{}%".format(len(df.columns) - 0.05),
                            "textOverflow": "ellipsis",
                            "overflow": "hidden",
                        },
                        virtualization=True,
                        fill_width=False,
                    ),
                    width=11,
                ),
                style={"margin-left": "30rem", "margin-top":"5rem"},
            ),
        ]
    )
    return layout



#create the app.callback to update the dropdown based on the value of the dropdown
@app.callback(
    Output('area', 'options'),
    Output('area', 'value'),
    Input('dflist', 'value'),
    State('dropdown-2-options', 'data')
)
#create a callback to update my datatable based on area dropdown value
def update_datatable(area):
    df = pd.read_csv('Data\{}'.format(area))
    data = df.to_dict('records')
    return data


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
        
    
    return dropdown_2_options, dropdown_2_value

if __name__ == '__main__':
    app.run_server(debug=True)
