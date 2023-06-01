import dash_bootstrap_components as dbc

def create_navbar():
    return dbc.NavbarSimple(children=[dbc.DropdownMenu(
                nav=True,
                in_navbar=True,
                label="Menu",
                children=[
                    dbc.DropdownMenuItem("Home", href='/'), # Hyperlink item that appears in the dropdown menu
                    dbc.DropdownMenuItem(divider=True),
                    dbc.DropdownMenuItem("Forecast", href='/forecast'),
                ],
            ),
        ],
    brand="",
    fluid=True,
    sticky="top",
    color= "ffffff"
)