import dash
import dash_table
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output

# Example Data
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
timeslots = ['9:00 - 10:00', '10:00 - 11:00', '11:00 - 12:00', '12:00 - 13:00', '13:00 - 14:00']

# Dummy data for timetable
timetable = [
    [{'course': 'Math', 'room': '101', 'lecturer': 'Dr. A'}, {'course': 'Physics', 'room': '102', 'lecturer': 'Dr. B'}, None, None, None],
    [None, {'course': 'Chemistry', 'room': '101', 'lecturer': 'Dr. C'}, None, None, {'course': 'History', 'room': '102', 'lecturer': 'Dr. D'}],
    [{'course': 'Biology', 'room': '103', 'lecturer': 'Dr. E'}, None, {'course': 'CS', 'room': '105', 'lecturer': 'Dr. F'}, None, None],
    [None, None, None, {'course': 'Math', 'room': '101', 'lecturer': 'Dr. A'}, {'course': 'Physics', 'room': '102', 'lecturer': 'Dr. B'}],
    [{'course': 'Philosophy', 'room': '107', 'lecturer': 'Dr. G'}, None, None, {'course': 'Economics', 'room': '108', 'lecturer': 'Dr. H'}, None],
]

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

def generate_table():
    table_rows = []
    
    # Create header row with weekdays
    header = html.Tr([html.Th("Time")] + [html.Th(day) for day in weekdays])
    table_rows.append(header)
    
    # Create each row for the timeslot
    for i, timeslot in enumerate(timeslots):
        row = [html.Td(timeslot)]  # Start row with timeslot
        
        for j in range(len(weekdays)):
            cell_data = timetable[i][j]
            if cell_data:
                # Tooltip for course name, room, and lecturer on hover
                cell_content = dbc.Tooltip(
                    f"Room: {cell_data['room']}, Lecturer: {cell_data['lecturer']}",
                    target=f"cell-{i}-{j}",
                    placement="top",
                    style={"fontSize": "12px"}
                )
                cell = html.Td(cell_data['course'], id=f"cell-{i}-{j}", style={"background-color": "#a3d4f7", "text-align": "center"})
                row.append(html.Div([cell, cell_content]))  # Wrap cell and tooltip in a div
            else:
                row.append(html.Td("-", style={"background-color": "#f0f0f0", "text-align": "center"}))  # Empty cell

        table_rows.append(html.Tr(row))
    
    return table_rows

# Layout
app.layout = dbc.Container([
    html.H1("Interactive Timetable"),
    dbc.Table(
        children=generate_table(),
        bordered=True,
        hover=True,
        responsive=True,
        striped=True,
        style={"width": "100%", "margin": "0 auto"}
    )
], fluid=True)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
