from dash import dcc, html
from dash.dependencies import Input, Output

from app import app, home_layout
#from data import data_layout
from forecast import forecast_layout

server = app.server
app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/forecast':
        return forecast_layout()
    else:
        return home_layout()

if __name__ == '__main__':
    app.run_server(debug=False)