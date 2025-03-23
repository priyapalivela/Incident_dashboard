import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import torch
from datetime import datetime

# Create sample incident data
def create_sample_incidents():
    incidents = [
        {
            'id': 'Incident-1',
            'location': 'Downtown',
            'lat': 41.8781,   # Chicago downtown coordinates
            'lon': -87.6298,
            'timestamp': '2025-03-20 14:30:00',
            'audio_class': 'gun_shot',
            'text_category': 'ROBBERY',
            'text_content': 'Armed robbery at convenience store',
            'severity': 2,  # High
            'confidence': 0.7,
            'measures': [
                "Initiate emergency response protocol",
                "Alert all relevant emergency services",
                "Lockdown facility",
                "Activate all security systems",
                "Secure valuable assets", 
                "Do not confront perpetrators",
                "Preserve evidence",
                "Gather witness statements",
                "Implement active shooter protocol",
                "Notify all personnel to take cover"
            ]
        },
        {
            'id': 'Incident-2',
            'location': 'Suburbia',
            'lat': 41.9200,  # North suburbs
            'lon': -87.7700,
            'timestamp': '2025-03-20 15:00:00',
            'audio_class': 'siren',
            'text_category': 'ASSAULT',
            'text_content': 'Assault reported near the park',
            'severity': 2,  # High
            'confidence': 0.7,
            'measures': [
                "Initiate emergency response protocol",
                "Secure the area",
                "Alert all relevant emergency services",
                "Check for injured individuals",
                "Provide first aid if necessary",
                "Separate involved parties",
                "Document physical injuries",
                "Prepare for emergency vehicle arrival",
                "Clear pathways for emergency vehicles"
            ]
        },
        {
            'id': 'Incident-3',
            'location': 'Industrial Area',
            'lat': 41.8150,  # Industrial district
            'lon': -87.6500,
            'timestamp': '2025-03-20 15:30:00',
            'audio_class': 'drilling',
            'text_category': 'BURGLARY',
            'text_content': 'Burglary at a warehouse',
            'severity': 2,  # High
            'confidence': 0.75,
            'measures': [
                "Initiate emergency response protocol",
                "Secure the immediate area",
                "Contact local law enforcement",
                "Secure entry points",
                "Check for stolen items",
                "Review surveillance footage",
                "Secure potential entry points",
                "Check surveillance footage for intruders",
                "Inspect for property damage"
            ]
        },
        {
            'id': 'Incident-4',
            'location': 'Residential Area',
            'lat': 41.9000,
            'lon': -87.6900,
            'timestamp': '2025-03-20 16:00:00',
            'audio_class': 'dog_bark',
            'text_category': 'PUBLIC PEACE VIOLATION',
            'text_content': 'Loud noise disturbance reported',
            'severity': 1,  # Medium
            'confidence': 0.68,
            'measures': [
                "Dispatch security personnel to the location",
                "Activate alert protocols",
                "Contact local law enforcement",
                "Secure the immediate area",
                "Prepare for potential escalation",
                "Check for injured individuals",
                "Document any suspicious activity"
            ]
        },
        {
            'id': 'Incident-5',
            'location': 'City Center',
            'lat': 41.8825,
            'lon': -87.6250,
            'timestamp': '2025-03-20 16:30:00',
            'audio_class': 'car_horn',
            'text_category': 'CRIMINAL DAMAGE',
            'text_content': 'Vandalism reported on Main Street',
            'severity': 1,  # Medium
            'confidence': 0.65,
            'measures': [
                "Dispatch security personnel to the location",
                "Activate alert protocols",
                "Contact local law enforcement",
                "Document the incident",
                "Investigate the source of the horn",
                "Check for traffic incidents"
            ]
        },
        {
            'id': 'Incident-6',
            'location': 'Shopping Mall',
            'lat': 41.8980,
            'lon': -87.6230,
            'timestamp': '2025-03-20 17:00:00',
            'audio_class': 'children_playing',
            'text_category': 'NON-CRIMINAL',
            'text_content': 'Children playing loudly in the mall',
            'severity': 1,  # Medium
            'confidence': 0.62,
            'measures': [
                "Document the incident",
                "Notify relevant authorities",
                "Increase monitoring in the area",
                "Activate alert protocols if situation escalates"
            ]
        },
        {
            'id': 'Incident-7',
            'location': 'Highway',
            'lat': 41.8330,
            'lon': -87.7520,
            'timestamp': '2025-03-20 17:30:00',
            'audio_class': 'engine_idling',
            'text_category': 'MOTOR VEHICLE THEFT',
            'text_content': 'Stolen vehicle found idling on the highway',
            'severity': 1,  # Medium
            'confidence': 0.64,
            'measures': [
                "Contact local law enforcement",
                "Document vehicle details",
                "Notify local law enforcement",
                "Check for stolen vehicles in the area",
                "Activate alert protocols",
                "Check for suspicious vehicles"
            ]
        },
        {
            'id': 'Incident-8',
            'location': 'Park',
            'lat': 41.8500,
            'lon': -87.6315,
            'timestamp': '2025-03-20 18:00:00',
            'audio_class': 'street_music',
            'text_category': 'PUBLIC INDECENCY',
            'text_content': 'Public indecency reported near the park stage',
            'severity': 0,  # Low
            'confidence': 0.58,
            'measures': [
                "Document the incident",
                "Notify relevant authorities",
                "Increase monitoring in the area"
            ]
        },
        {
            'id': 'Incident-9',
            'location': 'Train Station',
            'lat': 41.8785,
            'lon': -87.6390,
            'timestamp': '2025-03-20 18:30:00',
            'audio_class': 'jackhammer',
            'text_category': 'CRIMINAL TRESPASS',
            'text_content': 'Trespassing reported at construction site',
            'severity': 0,  # Low
            'confidence': 0.55,
            'measures': [
                "Document the incident",
                "Notify relevant authorities",
                "Verify construction activity permits",
                "Ensure safety protocols are followed",
                "Monitor for unauthorized activity"
            ]
        },
        {
            'id': 'Incident-10',
            'location': 'University Campus',
            'lat': 41.8710,
            'lon': -87.6490,
            'timestamp': '2025-03-20 19:00:00',
            'audio_class': 'air_conditioner',
            'text_category': 'OTHER OFFENSE',
            'text_content': 'Noise complaint from university building',
            'severity': 0,  # Low
            'confidence': 0.56,
            'measures': [
                "Document the incident",
                "Notify relevant authorities",
                "Increase monitoring in the area"
            ]
        }
    ]
    
    return incidents

# Convert incidents to DataFrame
def incidents_to_dataframe(incidents):
    df = pd.DataFrame(incidents)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Add readable severity label
    severity_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
    df['severity_label'] = df['severity'].map(severity_labels)
    
    # Add color for mapping
    severity_colors = {0: 'green', 1: 'orange', 2: 'red'}
    df['color'] = df['severity'].map(severity_colors)
    
    return df

# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # For deployment

# Get incident data
incidents = create_sample_incidents()
df = incidents_to_dataframe(incidents)

# Create severity distribution figure
def create_severity_chart(df):
    severity_counts = df['severity_label'].value_counts().reset_index()
    severity_counts.columns = ['Severity', 'Count']
    severity_counts = severity_counts.sort_values(by='Severity', key=lambda x: x.map({'Low': 0, 'Medium': 1, 'High': 2}))
    
    colors = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
    
    fig = px.bar(severity_counts, x='Severity', y='Count', 
                 color='Severity', color_discrete_map=colors,
                 text='Count')
    
    fig.update_layout(
        title='Incident Severity Distribution',
        xaxis_title='Severity Level',
        yaxis_title='Number of Incidents',
        showlegend=False
    )
    
    return fig

# Create map figure
def create_map(df):
    fig = px.scatter_mapbox(
        df, 
        lat='lat', 
        lon='lon', 
        color='severity_label',
        color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'},
        size=[10] * len(df),  # Constant size
        hover_name='id',
        hover_data=['location', 'timestamp', 'text_category', 'severity_label'],
        zoom=10,
        center={"lat": df['lat'].mean(), "lon": df['lon'].mean()},
        mapbox_style="open-street-map",
    )
    
    fig.update_layout(
        title='Geographic Distribution of Incidents',
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        legend_title="Severity",
        height=400
    )
    
    return fig

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Incident Monitoring Dashboard", className="text-center my-4"),
            html.P("Real-time monitoring of security incidents", className="text-center text-muted mb-4")
        ])
    ]),
    
    dbc.Row([
        # Severity Distribution
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='severity-chart', figure=create_severity_chart(df))
                ])
            ])
        ], width=6),
        
        # Map
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Graph(id='incident-map', figure=create_map(df))
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    # High Severity Incidents
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Recent High Severity Incidents"),
                dbc.CardBody([
                    html.Div(id='high-severity-list')
                ])
            ])
        ])
    ], className="mb-4"),
    
    # Incidents Table
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("All Incidents"),
                dbc.CardBody([
                    html.Div(id='incidents-table')
                ])
            ])
        ])
    ]),
    
    # Modal for incident details
    dbc.Modal([
        dbc.ModalHeader(html.H4(id="modal-title")),
        dbc.ModalBody(html.Div(id="modal-content")),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-modal", className="ms-auto")
        ),
    ], id="incident-modal", size="lg"),
    
    # Store intermediate data
    dcc.Store(id='stored-incidents', data=incidents),
    
    # Interval for auto refresh (every 30 seconds)
    dcc.Interval(
        id='interval-component',
        interval=30*1000,  # in milliseconds
        n_intervals=0
    )
], fluid=True)

# Create high severity incidents list
@app.callback(
    Output('high-severity-list', 'children'),
    Input('stored-incidents', 'data')
)
def update_high_severity_list(incidents):
    high_severity = [inc for inc in incidents if inc['severity'] == 2]
    high_severity = sorted(high_severity, key=lambda x: x['timestamp'], reverse=True)
    
    if not high_severity:
        return html.P("No high severity incidents", className="text-center p-3")
    
    cards = []
    for incident in high_severity:
        card = dbc.Card([
            dbc.CardBody([
                html.H5(f"ID: {incident['id']} | Location: {incident['location']}", className="card-title"),
                html.P(f"Time: {incident['timestamp']} | Type: {incident['audio_class']}/{incident['text_category']}"),
                html.P(incident['text_content'], className="font-italic"),
                html.Div([
                    html.P("Recommended Measures:", className="mb-1 font-weight-bold"),
                    html.Ul([html.Li(measure) for measure in incident['measures'][:3]]),
                ]),
                dbc.Button("View Details", id={"type": "high-severity-button", "index": incident['id']}, 
                           color="danger", className="mt-2")
            ])
        ], className="mb-3")
        cards.append(card)
    
    return cards

# Create incidents table
@app.callback(
    Output('incidents-table', 'children'),
    Input('stored-incidents', 'data')
)
def update_incidents_table(incidents):
    # Sort incidents by timestamp (newest first)
    sorted_incidents = sorted(incidents, key=lambda x: x['timestamp'], reverse=True)
    
    # Create table header
    header = html.Thead(html.Tr([
        html.Th("ID"),
        html.Th("Location"),
        html.Th("Time"),
        html.Th("Type"),
        html.Th("Severity"),
        html.Th("Action")
    ]))
    
    # Create table rows
    rows = []
    for incident in sorted_incidents:
        severity_label = {0: 'Low', 1: 'Medium', 2: 'High'}[incident['severity']]
        severity_color = {0: 'success', 1: 'warning', 2: 'danger'}[incident['severity']]
        
        row = html.Tr([
            html.Td(incident['id']),
            html.Td(incident['location']),
            html.Td(incident['timestamp']),
            html.Td(f"{incident['audio_class']}/{incident['text_category']}"),
            html.Td(html.Span(severity_label, className=f"badge bg-{severity_color}")),
            html.Td(dbc.Button("View Details", 
                               id={"type": "table-button", "index": incident['id']},
                               color="primary", size="sm"))
        ])
        rows.append(row)
    
    # Create table body
    body = html.Tbody(rows)
    
    # Create table
    table = dbc.Table([header, body], bordered=True, hover=True, responsive=True, striped=True)
    
    return table

# Callback for opening modal from high severity cards
@app.callback(
    Output('incident-modal', 'is_open', allow_duplicate=True),
    Output('modal-title', 'children', allow_duplicate=True),
    Output('modal-content', 'children', allow_duplicate=True),
    Input({"type": "high-severity-button", "index": dash.ALL}, 'n_clicks'),
    State('stored-incidents', 'data'),
    State('incident-modal', 'is_open'),
    prevent_initial_call=True
)
def open_modal_from_high_severity(n_clicks, incidents, is_open):
    # Check if any button was clicked
    if not any(n_clicks) or all(n is None for n in n_clicks):
        return dash.no_update, dash.no_update, dash.no_update
    
    # Find which button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update
    
    # Get the incident ID
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    incident_id = eval(button_id)['index']
    
    # Find the incident
    incident = next((inc for inc in incidents if inc['id'] == incident_id), None)
    
    if incident:
        # Create modal content
        title = f"Incident Details: {incident['id']}"
        
        content = html.Div([
            html.H5("Basic Information", className="mt-3"),
            dbc.Row([
                dbc.Col([
                    html.P(f"Location: {incident['location']}"),
                    html.P(f"Time: {incident['timestamp']}"),
                ], width=6),
                dbc.Col([
                    html.P(f"Audio Class: {incident['audio_class']}"),
                    html.P(f"Text Category: {incident['text_category']}"),
                ], width=6),
            ]),
            
            html.H5("Incident Description", className="mt-3"),
            html.P(incident['text_content']),
            
            html.H5("Severity Assessment", className="mt-3"),
            dbc.Progress(
                value=incident['confidence'] * 100,
                color={0: "success", 1: "warning", 2: "danger"}[incident['severity']],
                striped=True,
                className="mb-2"
            ),
            html.P(f"Severity: {['Low', 'Medium', 'High'][incident['severity']]} (Confidence: {incident['confidence']:.1%})"),
            
            html.H5("Recommended Measures", className="mt-3"),
            html.Ul([html.Li(measure) for measure in incident['measures']]),
            
            html.H5("Location", className="mt-3"),
            dcc.Graph(
                figure=px.scatter_mapbox(
                    pd.DataFrame([{
                        'lat': incident['lat'],
                        'lon': incident['lon'],
                        'location': incident['location'],
                        'id': incident['id']
                    }]),
                    lat='lat',
                    lon='lon',
                    hover_name='id',
                    hover_data=['location'],
                    zoom=14,
                    height=300,
                    mapbox_style="open-street-map"
                ).update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
            )
        ])
        
        return not is_open, title, content
    
    return is_open, dash.no_update, dash.no_update

# Callback for opening modal from table
@app.callback(
    Output('incident-modal', 'is_open'),
    Output('modal-title', 'children'),
    Output('modal-content', 'children'),
    Input({"type": "table-button", "index": dash.ALL}, 'n_clicks'),
    State('stored-incidents', 'data'),
    State('incident-modal', 'is_open'),
    prevent_initial_call=True
)
def open_modal_from_table(n_clicks, incidents, is_open):
    # Check if any button was clicked
    if not any(n_clicks) or all(n is None for n in n_clicks):
        return dash.no_update, dash.no_update, dash.no_update
    
    # Find which button was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update
    
    # Get the incident ID
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    incident_id = eval(button_id)['index']
    
    # Find the incident
    incident = next((inc for inc in incidents if inc['id'] == incident_id), None)
    
    if incident:
        # Create modal content
        title = f"Incident Details: {incident['id']}"
        
        content = html.Div([
            html.H5("Basic Information", className="mt-3"),
            dbc.Row([
                dbc.Col([
                    html.P(f"Location: {incident['location']}"),
                    html.P(f"Time: {incident['timestamp']}"),
                ], width=6),
                dbc.Col([
                    html.P(f"Audio Class: {incident['audio_class']}"),
                    html.P(f"Text Category: {incident['text_category']}"),
                ], width=6),
            ]),
            
            html.H5("Incident Description", className="mt-3"),
            html.P(incident['text_content']),
            
            html.H5("Severity Assessment", className="mt-3"),
            dbc.Progress(
                value=incident['confidence'] * 100,
                color={0: "success", 1: "warning", 2: "danger"}[incident['severity']],
                striped=True,
                className="mb-2"
            ),
            html.P(f"Severity: {['Low', 'Medium', 'High'][incident['severity']]} (Confidence: {incident['confidence']:.1%})"),
            
            html.H5("Recommended Measures", className="mt-3"),
            html.Ul([html.Li(measure) for measure in incident['measures']]),
            
            html.H5("Location", className="mt-3"),
            dcc.Graph(
                figure=px.scatter_mapbox(
                    pd.DataFrame([{
                        'lat': incident['lat'],
                        'lon': incident['lon'],
                        'location': incident['location'],
                        'id': incident['id']
                    }]),
                    lat='lat',
                    lon='lon',
                    hover_name='id',
                    hover_data=['location'],
                    zoom=14,
                    height=300,
                    mapbox_style="open-street-map"
                ).update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
            )
        ])
        
        return not is_open, title, content
    
    return is_open, dash.no_update, dash.no_update

# Callback for closing modal
@app.callback(
    Output('incident-modal', 'is_open', allow_duplicate=True),
    Input('close-modal', 'n_clicks'),
    State('incident-modal', 'is_open'),
    prevent_initial_call=True
)
def close_modal(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

# Run the app
if __name__ == '__main__':
    app.run(debug=False)