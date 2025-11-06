import sys
import io

# Force UTF-8 encoding for Windows console - MUST BE FIRST
if sys.platform == 'win32':
    try:
        # Reconfigure stdout/stderr to use UTF-8
        if sys.stdout.encoding != 'utf-8':
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if sys.stderr.encoding != 'utf-8':
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        # Fallback for older Python versions
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)


import dash
from dash import dcc, html, Input, Output, State, ALL, clientside_callback, callback
import dash_bootstrap_components as dbc
import json
import os
import shutil
import time
from datetime import datetime
import requests
import traceback

# Global variables for data management (will be set per session)
all_timetables = []
constraint_details = {}
rooms_data = []
input_data = None
session_has_swaps = False
original_consecutive_violations = []

def create_dash_app(session_file_path):
    """Create a fresh Dash app instance with enhanced error handling"""
    global all_timetables, constraint_details, rooms_data, input_data, session_has_swaps
    
    print(f"📋 Creating Dash app from session file: {session_file_path}")
    
    # Validate session file exists
    if not os.path.exists(session_file_path):
        print(f"❌ Session file does not exist: {session_file_path}")
        return None
    
    # Load and validate session data
    try:
        with open(session_file_path, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        print(f"✅ Session data loaded successfully")
        print(f"📊 Session data keys: {list(session_data.keys())}")
        
        # Load timetables with validation
        all_timetables = session_data.get('timetables', [])
        if not all_timetables:
            print("⚠️ Warning: No timetables found in session data")
            return None
        
        print(f"✅ Loaded {len(all_timetables)} timetables")
        
        # Validate timetable structure
        for i, timetable in enumerate(all_timetables):
            if not isinstance(timetable, dict):
                print(f"❌ Timetable {i} is not a dictionary")
                return None
            
            if 'timetable' not in timetable:
                print(f"❌ Timetable {i} missing 'timetable' field")
                return None
            
            grid = timetable['timetable']
            if not isinstance(grid, list) or not grid:
                print(f"❌ Timetable {i} has invalid grid structure")
                return None
        
        # Load input data with validation
        input_data = session_data.get('input_data', {})
        if not input_data:
            print("⚠️ Warning: No input data found")
            return None
        
        # Load other required data
        constraint_details = session_data.get('constraint_details', {})
        if not constraint_details:
            print("⚠️ Warning: No constraint details found in session data")
            constraint_details = {}
        else:
            print(f"✅ Loaded constraint details: {len(constraint_details)} constraint types")
            # Debug: print constraint types
            for key in constraint_details.keys():
                violation_count = len(constraint_details[key]) if isinstance(constraint_details[key], list) else constraint_details[key]
                print(f"   - {key}: {violation_count} violations")
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        return None
    except Exception as e:
        print(f"❌ Error loading session data: {e}")
        import traceback
        print(f"📋 Full traceback:\n{traceback.format_exc()}")
        return None
    
    # Create Dash app with enhanced configuration
    app = dash.Dash(
        __name__, 
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        update_title=None,  # Prevent "Updating..." in title
        title="Interactive Timetable Editor"
    )
    
    # ADD THE COMPLETE CSS STYLING - This is the fix for Bug #1
    app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
        <style>
            * {
                font-family: 'Poppins', sans-serif;
            }
            .cell {
                padding: 12px 10px;
                border: 1px solid #e0e0e0;
                border-radius: 3px;
                cursor: grab;
                min-height: 45px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: 400;
                font-size: 12px;
                transition: all 0.2s ease;
                user-select: none;
                line-height: 1.2;
                text-align: center;
                background-color: white;
                white-space: pre-line;
                word-wrap: break-word;
                overflow-wrap: break-word;
            }
            .cell:hover {
                transform: translateY(-1px);
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                border-color: #ccc;
            }
            .cell.dragging {
                opacity: 0.6;
                transform: rotate(2deg);
                cursor: grabbing;
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            }
            .cell.drag-over {
                background-color: #fff3cd !important;
                border-color: #ffc107 !important;
                transform: scale(1.02);
                box-shadow: 0 2px 8px rgba(255, 193, 7, 0.6);
            }
            .cell.break-time {
                background-color: #ff5722 !important;
                color: white;
                cursor: not-allowed;
                font-weight: 500;
            }
            .cell.break-time:hover {
                transform: none;
                box-shadow: none;
            }
            .cell.manually-set {
                border: 2px solid #4CAF50;
                background-color: #f1f8f4;
            }
            .cell.room-conflict {
                background-color: #ffebee !important;
                border: 2px solid #ef5350 !important;
                position: relative;
            }
            .cell.lecturer-conflict {
                background-color: #fff3e0 !important;
                border: 2px dashed #ff9800 !important;
            }
            .cell.selected {
                background-color: #e3f2fd !important;
                border: 2px solid #2196F3 !important;
                box-shadow: 0 0 0 3px rgba(33, 150, 243, 0.2);
            }
            .header-cell {
                background-color: #11214D;
                color: white;
                font-weight: 600;
                text-align: center;
                padding: 12px;
                font-size: 13px;
                border: 1px solid #0d1a3d;
            }
            .time-cell {
                background-color: #f5f5f5;
                font-weight: 500;
                text-align: center;
                padding: 12px 8px;
                font-size: 12px;
                color: #555;
                border: 1px solid #e0e0e0;
                white-space: nowrap;
            }
            .modal-overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.5);
                z-index: 999;
                display: none;
            }
            .room-selection-modal {
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: white;
                padding: 0;
                border-radius: 8px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
                z-index: 1000;
                max-width: 600px;
                width: 90%;
                max-height: 80vh;
                overflow-y: auto;
                display: none;
            }
            .modal-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 20px;
                border-bottom: 1px solid #f0f0f0;
                background-color: #fafafa;
                border-radius: 8px 8px 0 0;
            }
            .modal-title {
                margin: 0;
                font-size: 20px;
                font-weight: 600;
                color: #333;
            }
            .modal-close {
                background: none;
                border: none;
                font-size: 28px;
                color: #666;
                cursor: pointer;
                padding: 0;
                width: 30px;
                height: 30px;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 50%;
                transition: all 0.2s ease;
            }
            .modal-close:hover {
                background-color: #f0f0f0;
                color: #333;
            }
            .room-list {
                padding: 15px 20px;
            }
            .room-option {
                padding: 12px;
                margin: 8px 0;
                border: 1px solid #e0e0e0;
                border-radius: 5px;
                cursor: pointer;
                transition: all 0.2s ease;
                background-color: white;
            }
            .room-option:hover {
                background-color: #f5f5f5;
                border-color: #11214D;
                transform: translateX(5px);
            }
            .room-name {
                font-weight: 500;
                font-size: 15px;
                color: #333;
            }
            .room-details {
                font-size: 12px;
                color: #666;
                margin-top: 4px;
            }
            .conflict-warning {
                position: fixed;
                top: 80px;
                right: 20px;
                background: white;
                border-left: 4px solid #ff9800;
                padding: 0;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                z-index: 1001;
                max-width: 400px;
                display: none;
                animation: slideInRight 0.3s ease;
            }
            .conflict-warning-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 12px 16px;
                background-color: #fff3e0;
                border-radius: 8px 8px 0 0;
            }
            .conflict-warning-title {
                font-weight: 600;
                font-size: 14px;
                color: #e65100;
            }
            .conflict-warning-close {
                background: none;
                border: none;
                font-size: 20px;
                color: #e65100;
                cursor: pointer;
                padding: 0;
                width: 24px;
                height: 24px;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 50%;
                transition: all 0.2s ease;
            }
            .conflict-warning-close:hover {
                background-color: rgba(230, 81, 0, 0.1);
            }
            .conflict-warning-content {
                padding: 12px 16px;
                font-size: 13px;
                color: #666;
                line-height: 1.5;
            }
            .save-error {
                position: fixed;
                top: 80px;
                right: 20px;
                background: white;
                border-left: 4px solid #f44336;
                padding: 16px;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                z-index: 1001;
                max-width: 350px;
                display: none;
                animation: slideInRight 0.3s ease;
            }
            .save-error-title {
                font-weight: 600;
                font-size: 14px;
                color: #c62828;
                margin-bottom: 8px;
            }
            .save-error-content {
                font-size: 13px;
                color: #666;
            }
            @keyframes slideInRight {
                from {
                    transform: translateX(400px);
                    opacity: 0;
                }
                to {
                    transform: translateX(0);
                    opacity: 1;
                }
            }
            .help-section {
                padding: 15px 20px;
                border-bottom: 1px solid #f0f0f0;
            }
            .help-section:last-child {
                border-bottom: none;
            }
            .help-section h4 {
                margin: 0 0 10px 0;
                font-size: 16px;
                font-weight: 600;
                color: #11214D;
            }
            .help-section p, .help-section ul {
                margin: 8px 0;
                font-size: 14px;
                line-height: 1.6;
                color: #555;
            }
            .help-section ul {
                padding-left: 20px;
            }
            .help-section li {
                margin: 5px 0;
            }
            .help-note {
                background-color: #fff3cd;
                border: 1px solid #ffc107;
                border-radius: 5px;
                padding: 12px;
                font-size: 13px;
                color: #856404;
                margin-bottom: 15px;
            }
            .constraint-container {
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 5px;
                margin: 10px 0;
                overflow: hidden;
            }
            .constraint-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 12px 16px;
                background-color: #f8f9fa;
                cursor: pointer;
                transition: background-color 0.2s ease;
                user-select: none;
            }
            .constraint-header:hover {
                background-color: #e9ecef;
            }
            .constraint-header-content {
                display: flex;
                align-items: center;
                gap: 12px;
                flex: 1;
            }
            .constraint-badge {
                background-color: #dc3545;
                color: white;
                padding: 4px 12px;
                border-radius: 12px;
                font-size: 12px;
                font-weight: 600;
                min-width: 30px;
                text-align: center;
            }
            .constraint-title {
                font-weight: 500;
                font-size: 14px;
                color: #333;
            }
            .constraint-body {
                max-height: 0;
                overflow: hidden;
                transition: max-height 0.3s ease;
            }
            .constraint-body.expanded {
                max-height: 300px;
                overflow-y: auto;
                border-top: 1px solid #e0e0e0;
            }
            .constraint-item {
                padding: 10px 16px;
                border-bottom: 1px solid #f0f0f0;
                font-size: 13px;
                line-height: 1.4;
                color: #666;
                word-wrap: break-word;
                overflow-wrap: break-word;
            }
            .constraint-item:last-child {
                border-bottom: none;
            }
            .constraint-arrow {
                font-weight: bold;
                transition: transform 0.3s ease;
                font-family: monospace;
                font-size: 16px;
            }
            .constraint-arrow.rotated {
                transform: rotate(180deg);
            }
            .errors-button {
                position: relative;
                background: #dc3545;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s ease;
                font-family: 'Poppins', sans-serif;
                box-shadow: 0 2px 4px rgba(220, 53, 69, 0.3);
            }
            .errors-button:hover {
                background: #c82333;
                transform: translateY(-1px);
                box-shadow: 0 4px 8px rgba(220, 53, 69, 0.4);
            }
            .errors-button:disabled {
                background: #6c757d;
                cursor: not-allowed;
                transform: none;
                box-shadow: none;
            }
            .error-notification {
                position: absolute;
                top: -8px;
                right: -8px;
                background: rgba(255, 255, 255, 0.9);
                color: #dc3545;
                border: 2px solid #dc3545;
                border-radius: 50%;
                width: 24px;
                height: 24px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 12px;
                font-weight: 700;
                font-family: 'Poppins', sans-serif;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''
    
    # Enhanced server configuration
    @app.server.after_request
    def after_request(response):
        response.headers.add('X-Frame-Options', 'SAMEORIGIN')
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        response.headers.add('Cache-Control', 'no-cache, no-store, must-revalidate')
        response.headers.add('Pragma', 'no-cache')
        response.headers.add('Expires', '0')
        return response
    
    # Set the layout with proper data validation
    try:
        app.layout = create_layout()
        print("✅ Layout created successfully")
    except Exception as e:
        print(f"❌ Error creating layout: {e}")
        return None
    
    # Register callbacks with error handling
    try:
        register_callbacks(app)
        print("✅ Callbacks registered successfully")
    except Exception as e:
        print(f"❌ Error registering callbacks: {e}")
        return None
    
    print("✅ Dash app created successfully")
    return app

def create_layout():
    """Create the app layout using loaded global data"""
    global all_timetables, constraint_details, rooms_data
    
    if not all_timetables:
        return html.Div([
            html.Div([
                html.H1("No Timetable Data Available", 
                       style={"color": "#dc3545", "textAlign": "center", "marginTop": "50px"}),
                html.P("The session file did not contain valid timetable data.", 
                      style={"textAlign": "center", "color": "#666", "fontSize": "16px"}),
                html.P("Please check the data conversion process.", 
                      style={"textAlign": "center", "color": "#666", "fontSize": "14px"})
            ])
        ])
    
    return html.Div([
        # Title and dropdown
        html.Div([
            html.H1("Interactive Drag & Drop Timetable - DE Optimization Results", 
                    style={"color": "#11214D", "fontWeight": "600", "fontSize": "24px", 
                          "fontFamily": "Poppins, sans-serif", "margin": "0", "flex": "1"}),
            
            html.Div([
                dcc.Dropdown(
                    id='student-group-dropdown',
                    options=[{'label': timetable_data['student_group']['name'] if isinstance(timetable_data['student_group'], dict) 
                                      else timetable_data['student_group'].name, 'value': idx} 
                            for idx, timetable_data in enumerate(all_timetables)],
                    value=0,
                    searchable=True,
                    clearable=False,
                    style={"width": "280px", "fontSize": "13px", "fontFamily": "Poppins, sans-serif"}
                )
            ], style={"display": "flex", "alignItems": "center"})
        ], style={"display": "flex", "alignItems": "center", "justifyContent": "space-between", 
                 "marginTop": "30px", "marginBottom": "30px", "maxWidth": "1200px", 
                 "margin": "30px auto", "padding": "0 15px"}),
        
        # Stores for data
        dcc.Store(id="all-timetables-store", data=all_timetables),
        dcc.Store(id="rooms-data-store", data=rooms_data),
        dcc.Store(id="original-timetables-store", data=[timetable_data.copy() for timetable_data in all_timetables]),
        dcc.Store(id="constraint-details-store", data=constraint_details),
        dcc.Store(id="swap-data", data=None),
        dcc.Store(id="room-change-data", data=None),
        dcc.Store(id="missing-class-data", data=None),
        dcc.Store(id="manual-cells-store", data=[]),
        
        # Hidden trigger div
        html.Div(id="trigger", style={"display": "none"}),
        
        # Timetable container
        html.Div(id="timetable-container"),
        
        # Room selection modal
        html.Div([
            html.Div(className="modal-overlay", id="modal-overlay", style={"display": "none"}),
            html.Div([
                html.Div([
                    html.H3("Select Classroom", className="modal-title"),
                    html.Button("×", className="modal-close", id="modal-close-btn")
                ], className="modal-header"),
                
                dcc.Input(
                    id="room-search-input",
                    type="text",
                    placeholder="Search classrooms...",
                    className="room-search"
                ),
                
                html.Div(id="room-options-container", className="room-options"),
                
                html.Div([
                    html.Button("Cancel", id="room-cancel-btn", 
                               style={"backgroundColor": "#f5f5f5", "color": "#666", "padding": "8px 16px", 
                                     "border": "1px solid #ddd", "borderRadius": "5px", "marginRight": "10px",
                                     "cursor": "pointer", "fontFamily": "Poppins, sans-serif"}),
                    html.Button("DELETE SCHEDULE", id="room-delete-btn", 
                               style={"backgroundColor": "#dc3545", "color": "white", "padding": "8px 16px", 
                                     "border": "none", "borderRadius": "5px", "cursor": "pointer", "marginRight": "10px",
                                     "fontFamily": "Poppins, sans-serif", "fontWeight": "600", "display": "none"}),
                    html.Button("Confirm", id="room-confirm-btn", 
                               style={"backgroundColor": "#11214D", "color": "white", "padding": "8px 16px", 
                                     "border": "none", "borderRadius": "5px", "cursor": "pointer",
                                     "fontFamily": "Poppins, sans-serif"})
                ], style={"textAlign": "right", "marginTop": "20px", "paddingTop": "15px", 
                         "borderTop": "1px solid #f0f0f0"})
            ], className="room-selection-modal", id="room-selection-modal", style={"display": "none"})
        ]),
        
        # Errors modal
        html.Div([
            html.Div(className="modal-overlay", id="errors-modal-overlay", style={"display": "none"}),
            html.Div([
                html.Div([
                    html.H3("Constraint Violations", className="modal-title"),
                    html.Button("×", className="modal-close", id="errors-modal-close-btn")
                ], className="modal-header"),
                
                html.Div(id="errors-content"),
                
                html.Div([
                    html.Button("Close", id="errors-close-btn", 
                               style={"backgroundColor": "#11214D", "color": "white", "padding": "8px 16px", 
                                     "border": "none", "borderRadius": "5px", "cursor": "pointer",
                                     "fontFamily": "Poppins, sans-serif"})
                ], style={"textAlign": "right", "marginTop": "20px", "paddingTop": "15px", 
                         "borderTop": "1px solid #f0f0f0"})
            ], className="room-selection-modal", id="errors-modal", style={"display": "none"})
        ]),
        
        # Download modal
        html.Div([
            html.Div(className="modal-overlay", id="download-modal-overlay", style={"display": "none"}),
            html.Div([
                html.Div([
                    html.H3("Download Timetables", className="modal-title"),
                    html.Button("×", className="modal-close", id="download-modal-close-btn")
                ], className="modal-header"),
                
                html.Div([
                    html.Div([
                        html.Span("Download SST Timetables", style={"flex": "1", "fontSize": "16px", "fontWeight": "500"}),
                        html.Button("Download", id="download-sst-btn", 
                                   style={"backgroundColor": "#11214D", "color": "white", "padding": "8px 16px", 
                                         "border": "none", "borderRadius": "5px", "cursor": "pointer",
                                         "fontFamily": "Poppins, sans-serif", "fontSize": "14px"})
                    ], style={"display": "flex", "alignItems": "center", "justifyContent": "space-between", 
                             "padding": "15px", "borderBottom": "1px solid #f0f0f0"}),
                    
                    html.Div([
                        html.Span("Download TYD Timetables", style={"flex": "1", "fontSize": "16px", "fontWeight": "500"}),
                        html.Button("Download", id="download-tyd-btn", 
                                   style={"backgroundColor": "#11214D", "color": "white", "padding": "8px 16px", 
                                         "border": "none", "borderRadius": "5px", "cursor": "pointer",
                                         "fontFamily": "Poppins, sans-serif", "fontSize": "14px"})
                    ], style={"display": "flex", "alignItems": "center", "justifyContent": "space-between", 
                             "padding": "15px", "borderBottom": "1px solid #f0f0f0"}),
                    
                    html.Div([
                        html.Span("Download all Lecturer Timetables", style={"flex": "1", "fontSize": "16px", "fontWeight": "500"}),
                        html.Button("Download", id="download-lecturer-btn", 
                                   style={"backgroundColor": "#11214D", "color": "white", "padding": "8px 16px", 
                                         "border": "none", "borderRadius": "5px", "cursor": "pointer",
                                         "fontFamily": "Poppins, sans-serif", "fontSize": "14px"})
                    ], style={"display": "flex", "alignItems": "center", "justifyContent": "space-between", 
                             "padding": "15px"})
                ]),
                
                html.Div([
                    html.Button("Close", id="download-close-btn", 
                               style={"backgroundColor": "#6c757d", "color": "white", "padding": "8px 16px", 
                                     "border": "none", "borderRadius": "5px", "cursor": "pointer",
                                     "fontFamily": "Poppins, sans-serif"})
                ], style={"textAlign": "right", "marginTop": "20px", "paddingTop": "15px", 
                         "borderTop": "1px solid #f0f0f0"})
            ], className="room-selection-modal", id="download-modal", style={"display": "none"})
        ]),
        
        # Conflict warning popup
        html.Div([
            html.Div([
                html.Span("⚠️ Classroom Conflict", className="conflict-warning-title"),
                html.Button("×", className="conflict-warning-close", id="conflict-close-btn")
            ], className="conflict-warning-header"),
            html.Div(id="conflict-warning-text", className="conflict-warning-content")
        ], className="conflict-warning", id="conflict-warning", style={"display": "none"}),
        
        # Help modal
        html.Div([
            html.Div(className="modal-overlay", id="help-modal-overlay", style={"display": "none"}),
            html.Div([
                html.Div([
                    html.H3("Timetable Help Guide"),
                    html.Button("×", className="modal-close", id="help-modal-close-btn")
                ], className="modal-header"),
                
                html.Div([
                    html.Div([
                        html.Strong("NOTE: "),
                        "Ensure all Lecturer names, emails and other details are inputted correctly to prevent errors"
                    ], className="help-note"),
                    
                    html.Div([
                        html.H4("How to Use the Timetable:"),
                        html.P("• Click and drag any class cell to swap it with another cell"),
                        html.P("• Double-click any cell to view and change the classroom for that class"),
                        html.P("• Use the navigation arrows (‹ ›) to switch between different student groups"),
                        html.P("• Click 'View Errors' to see constraint violations and conflicts")
                    ], className="help-section"),
                    
                    html.Div([
                        html.H4("Cell Color Meanings:"),
                        html.Div([
                            html.Div([
                                html.Div(className="color-box normal"),
                                html.Span("Normal class - No conflicts")
                            ], className="color-item"),
                            html.Div([
                                html.Div(className="color-box manual"),
                                html.Span("Manually Scheduled - Class scheduled manually from the 'Missing Classes' list.")
                            ], className="color-item"),
                            html.Div([
                                html.Div(className="color-box break"),
                                html.Span("Break time - Classes cannot be scheduled")
                            ], className="color-item"),
                            html.Div([
                                html.Div(className="color-box room-conflict"),
                                html.Span("Room conflict - Same classroom used by multiple groups")
                            ], className="color-item"),
                            html.Div([
                                html.Div(className="color-box lecturer-conflict"),
                                html.Span("Lecturer conflict - Same lecturer teaching multiple groups")
                            ], className="color-item"),
                            html.Div([
                                html.Div(className="color-box both-conflict"),
                                html.Span("Multiple conflicts - Both room and lecturer issues")
                            ], className="color-item")
                        ], className="color-legend")
                    ], className="help-section")
                ]),
                
                html.Div([
                    html.Button("Close", id="help-close-btn", 
                               style={"backgroundColor": "#11214D", "color": "white", "padding": "10px 20px", 
                                     "border": "none", "borderRadius": "8px", "cursor": "pointer",
                                     "fontFamily": "Poppins, sans-serif", "fontSize": "14px", "fontWeight": "600"})
                ], style={"textAlign": "center", "marginTop": "25px", "paddingTop": "20px", 
                         "borderTop": "2px solid #f0f0f0"})
            ], className="help-modal", id="help-modal", style={"display": "none"})
        ]),

        # Download button
        html.Div([
            html.Button("Download Timetables", id="download-button", 
                       style={"backgroundColor": "#11214D", "color": "white", "padding": "10px 20px", 
                             "border": "none", "borderRadius": "5px", "fontSize": "14px", "cursor": "pointer",
                             "fontWeight": "600", "boxShadow": "0 1px 3px rgba(0,0,0,0.1)",
                             "transition": "all 0.2s ease", "fontFamily": "Poppins, sans-serif"}),
            html.Div(id="download-status", style={"marginTop": "12px", "fontWeight": "600", 
                                             "fontFamily": "Poppins, sans-serif", "fontSize": "12px"})
        ], style={"textAlign": "center", "marginTop": "30px", "maxWidth": "1200px", "margin": "30px auto 0 auto"}),
        
        # Feedback area
        html.Div(id="feedback", style={
            "marginTop": "20px", 
            "textAlign": "center", 
            "fontSize": "16px", 
            "fontWeight": "bold",
            "minHeight": "30px",
            "maxWidth": "1200px",
            "margin": "20px auto 0 auto"
        })
    ])

def register_callbacks(app):
    """Register all callbacks with the specific app instance"""
    
    # Callback to create and update timetable
    @app.callback(
        [Output("timetable-container", "children"),
         Output("trigger", "children")],
        [Input("student-group-dropdown", "value")],
        [State("all-timetables-store", "data"),
         State("manual-cells-store", "data")]
    )
    def create_timetable(selected_group_idx, all_timetables_data, manual_cells):
        if selected_group_idx is None or not all_timetables_data:
            return html.Div("No data available"), "trigger"
        
        if selected_group_idx >= len(all_timetables_data):
            print(f"Selected group index {selected_group_idx} out of bounds (max: {len(all_timetables_data)-1})")
            selected_group_idx = 0
            
        timetable_data = all_timetables_data[selected_group_idx]
        student_group_name = timetable_data['student_group']['name'] if isinstance(timetable_data['student_group'], dict) else timetable_data['student_group'].name
        timetable_rows = timetable_data['timetable']
        
        conflicts = detect_conflicts(all_timetables_data, selected_group_idx)
        
        days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        
        rows = []
        
        # Header row
        header_cells = [html.Th("Time", style={
            "backgroundColor": "#11214D", 
            "color": "white", 
            "padding": "12px 10px",
            "fontWeight": "600",
            "fontSize": "13px",
            "textAlign": "center",
            "border": "1px solid #0d1a3d",
            "fontFamily": "Poppins, sans-serif"
        })]
        
        for day in days_of_week:
            header_cells.append(html.Th(day, style={
                "backgroundColor": "#11214D", 
                "color": "white", 
                "padding": "12px 10px",
                "fontWeight": "600",
                "fontSize": "13px",
                "textAlign": "center",
                "border": "1px solid #0d1a3d",
                "fontFamily": "Poppins, sans-serif"
            }))
        
        rows.append(html.Thead(html.Tr(header_cells)))
        
        # Data rows
        body_rows = []
        
        for row_idx in range(len(timetable_rows)):
            cells = [html.Td(timetable_rows[row_idx][0], className="time-cell")]
            
            for col_idx in range(1, len(timetable_rows[row_idx])):
                cell_content = timetable_rows[row_idx][col_idx] if timetable_rows[row_idx][col_idx] else "FREE"
                cell_id = {"type": "cell", "group": selected_group_idx, "row": row_idx, "col": col_idx-1}
                
                is_break = cell_content == "BREAK"
                
                timeslot_key = f"{row_idx}_{col_idx-1}"
                has_conflict = timeslot_key in conflicts
                conflict_type = conflicts.get(timeslot_key, {}).get('type', 'none') if has_conflict else 'none'
                
                manual_cell_key = f"{selected_group_idx}_{row_idx}_{col_idx-1}"
                is_manual = manual_cells and manual_cell_key in manual_cells
                
                if is_break:
                    cell_class = "cell break-time"
                    draggable = "false"
                elif is_manual:
                    if conflict_type == 'room':
                        cell_class = "cell room-conflict manual-schedule"
                    elif conflict_type == 'lecturer':
                        cell_class = "cell lecturer-conflict manual-schedule"
                    elif conflict_type == 'both':
                        cell_class = "cell both-conflict manual-schedule"
                    else:
                        cell_class = "cell manual-schedule"
                    draggable = "true"
                elif conflict_type == 'room':
                    cell_class = "cell room-conflict"
                    draggable = "true"
                elif conflict_type == 'lecturer':
                    cell_class = "cell lecturer-conflict"
                    draggable = "true"
                elif conflict_type == 'both':
                    cell_class = "cell both-conflict"
                    draggable = "true"
                else:
                    cell_class = "cell"
                    draggable = "true"
                
                cells.append(
                    html.Td(
                        html.Div(
                            cell_content,
                            id=cell_id,
                            className=cell_class,
                            draggable=draggable,
                            n_clicks=0
                        ),
                        style={"padding": "0", "border": "1px solid #e0e0e0"}
                    )
                )
            
            body_rows.append(html.Tr(cells))
        
        rows.append(html.Tbody(body_rows))
        
        table = html.Table(rows, style={
            "width": "100%",
            "borderCollapse": "separate",
            "borderSpacing": "0",
            "backgroundColor": "white",
            "borderRadius": "6px",
            "overflow": "hidden",
            "fontSize": "12px",
            "boxShadow": "0 2px 6px rgba(0,0,0,0.08)",
            "fontFamily": "Poppins, sans-serif"
        })
        
        return html.Div([
            html.Div([
                html.Div([
                    html.H2(f"Timetable for {student_group_name}", 
                           className="timetable-title",
                           style={"color": "#11214D", "fontWeight": "600", "fontSize": "20px", 
                                 "fontFamily": "Poppins, sans-serif", "margin": "0"})
                ], className="timetable-title-container"),
                html.Div([
                    html.Button("‹", className="nav-arrow", id="prev-group-btn",
                               disabled=selected_group_idx == 0),
                    html.Button("›", className="nav-arrow", id="next-group-btn", 
                               disabled=selected_group_idx == len(all_timetables_data) - 1)
                ], className="nav-arrows")
            ], className="timetable-header"),
            
            html.Div([
                html.Button([
                    "View Errors",
                    html.Div(id="error-notification-badge", className="error-notification")
                ], id="errors-btn", className="errors-button"),
                html.Button("Undo All Changes", id="undo-all-btn", 
                           style={"backgroundColor": "#6c757d", "color": "white", "padding": "8px 16px", 
                                 "border": "none", "borderRadius": "5px", "fontSize": "14px", "cursor": "pointer",
                                 "fontWeight": "500", "fontFamily": "Poppins, sans-serif"}),
                html.Button("?", id="help-icon-btn", title="Help", 
                           className="nav-arrow", style={"marginLeft": "auto"})
            ], style={"marginBottom": "15px", "marginRight": "10px", "textAlign": "left", "display": "flex", "gap": "10px", "alignItems": "flex-start"}),
            
            table
        ], className="student-group-container"), "trigger"

    # Callback to handle navigation arrows
    @app.callback(
        Output("student-group-dropdown", "value"),
        [Input("prev-group-btn", "n_clicks"),
         Input("next-group-btn", "n_clicks")],
        [State("student-group-dropdown", "value"),
         State("all-timetables-store", "data")],
        prevent_initial_call=True
    )
    def handle_navigation(prev_clicks, next_clicks, current_value, all_timetables_data):
        ctx = dash.callback_context
        if not ctx.triggered or not all_timetables_data:
            raise dash.exceptions.PreventUpdate
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        click_value = ctx.triggered[0]['value']
        
        if click_value is None or click_value == 0:
            raise dash.exceptions.PreventUpdate
        
        if current_value is None:
            current_value = 0
        
        max_index = len(all_timetables_data) - 1
        current_value = max(0, min(current_value, max_index))
        
        if button_id == "prev-group-btn" and current_value > 0:
            return current_value - 1
        elif button_id == "next-group-btn" and current_value < max_index:
            return current_value + 1
        
        raise dash.exceptions.PreventUpdate

    # Callback to handle swap operations
    @app.callback(
        [Output("all-timetables-store", "data", allow_duplicate=True),
         Output("manual-cells-store", "data", allow_duplicate=True)],
        Input("swap-data", "data"),
        [State("all-timetables-store", "data"),
         State("manual-cells-store", "data")],
        prevent_initial_call=True
    )
    def handle_swap(swap_data, current_timetables, manual_cells):
        global session_has_swaps
        
        if not swap_data or not current_timetables:
            raise dash.exceptions.PreventUpdate
        
        try:
            source = swap_data['source']
            target = swap_data['target']
            
            if not source or not target or 'group' not in source or 'group' not in target:
                raise dash.exceptions.PreventUpdate
            
            if source['group'] != target['group']:
                raise dash.exceptions.PreventUpdate
            
            if source.get('content') == 'BREAK' or target.get('content') == 'BREAK':
                raise dash.exceptions.PreventUpdate
            
            group_idx = source['group']
            
            if group_idx < 0 or group_idx >= len(current_timetables):
                raise dash.exceptions.PreventUpdate
            
            updated_timetables = json.loads(json.dumps(current_timetables))
            timetable_rows = updated_timetables[group_idx]['timetable']
            
            if (source['row'] >= len(timetable_rows) or target['row'] >= len(timetable_rows) or
                source['col'] + 1 >= len(timetable_rows[0]) or target['col'] + 1 >= len(timetable_rows[0])):
                raise dash.exceptions.PreventUpdate
            
            source_content = timetable_rows[source['row']][source['col'] + 1]
            target_content = timetable_rows[target['row']][target['col'] + 1]
            
            timetable_rows[source['row']][source['col'] + 1] = target_content
            timetable_rows[target['row']][target['col'] + 1] = source_content
            
            session_has_swaps = True

            source_is_manual = swap_data.get('sourceIsManual', False)
            target_is_manual = swap_data.get('targetIsManual', False)
            updated_manual_cells = manual_cells.copy() if manual_cells else []

            source_key = f"{source['group']}_{source['row']}_{source['col']}"
            target_key = f"{target['group']}_{target['row']}_{target['col']}"

            if source_is_manual and source_key in updated_manual_cells:
                updated_manual_cells.remove(source_key)
            if target_is_manual and target_key in updated_manual_cells:
                updated_manual_cells.remove(target_key)

            if source_is_manual and target_key not in updated_manual_cells:
                updated_manual_cells.append(target_key)
            if target_is_manual and source_key not in updated_manual_cells:
                updated_manual_cells.append(source_key)
            
            return updated_timetables, updated_manual_cells
            
        except Exception as e:
            print(f"Error in handle_swap: {e}")
            raise dash.exceptions.PreventUpdate

    # Callback to update error notification badge
    @app.callback(
        Output("error-notification-badge", "children"),
        [Input("constraint-details-store", "data"),
         Input("all-timetables-store", "data")],
        prevent_initial_call=False
    )
    def update_error_notification_badge(constraint_details, timetables_data):
        if not constraint_details:
            return "0"
        
        hard_constraint_names = [
            'Same Student Group Overlaps',
            'Different Student Group Overlaps', 
            'Lecturer Clashes',
            'Lecturer Schedule Conflicts (Day/Time)',
            'Lecturer Workload Violations',
            'Consecutive Slot Violations',
            'Missing or Extra Classes',
            'Same Course in Multiple Rooms on Same Day',
            'Room Capacity/Type Conflicts',
            'Classes During Break Time'
        ]
        
        violated_hard_constraints = 0
        for constraint_name in hard_constraint_names:
            violations = constraint_details.get(constraint_name, [])
            if len(violations) > 0:
                violated_hard_constraints += 1
        
        return str(violated_hard_constraints)

    # Callback to handle errors modal
    @app.callback(
        [Output("errors-modal-overlay", "style"),
         Output("errors-modal", "style"),
         Output("errors-content", "children")],
        [Input("errors-btn", "n_clicks"),
         Input("errors-modal-close-btn", "n_clicks"),
         Input("errors-close-btn", "n_clicks"),
         Input("errors-modal-overlay", "n_clicks")],
        State("constraint-details-store", "data"),
        prevent_initial_call=True
    )
    def handle_errors_modal(errors_btn_clicks, close_btn_clicks, close_btn2_clicks, overlay_clicks, constraint_details):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if trigger_id == "errors-btn" and errors_btn_clicks:
            modal_content = create_errors_modal_content(constraint_details)
            return {"display": "block"}, {"display": "block"}, modal_content
        elif trigger_id in ["errors-modal-close-btn", "errors-close-btn", "errors-modal-overlay"]:
            return {"display": "none"}, {"display": "none"}, []
        
        raise dash.exceptions.PreventUpdate

    # Callback to handle help modal
    @app.callback(
        [Output("help-modal-overlay", "style"),
         Output("help-modal", "style")],
        [Input("help-icon-btn", "n_clicks"),
         Input("help-modal-close-btn", "n_clicks"), 
         Input("help-close-btn", "n_clicks"),
         Input("help-modal-overlay", "n_clicks")],
        prevent_initial_call=True
    )
    def handle_help_modal(help_btn_clicks, close_btn_clicks, close_btn2_clicks, overlay_clicks):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if trigger_id == "help-icon-btn" and help_btn_clicks:
            return {"display": "block"}, {"display": "block"}
        elif trigger_id in ["help-modal-close-btn", "help-close-btn", "help-modal-overlay"]:
            return {"display": "none"}, {"display": "none"}
        
        raise dash.exceptions.PreventUpdate

    # Callback to handle download modal
    @app.callback(
        [Output("download-modal-overlay", "style"),
         Output("download-modal", "style")],
        [Input("download-button", "n_clicks"),
         Input("download-modal-close-btn", "n_clicks"),
         Input("download-close-btn", "n_clicks"),
         Input("download-modal-overlay", "n_clicks")],
        prevent_initial_call=True
    )
    def handle_download_modal(download_btn_clicks, close_btn_clicks, close_btn2_clicks, overlay_clicks):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if trigger_id == "download-button" and download_btn_clicks:
            return {"display": "block"}, {"display": "block"}
        elif trigger_id in ["download-modal-close-btn", "download-close-btn", "download-modal-overlay"]:
            return {"display": "none"}, {"display": "none"}
        
        raise dash.exceptions.PreventUpdate

    # Callback to handle download actions
    @app.callback(
        Output("download-status", "children"),
        [Input("download-sst-btn", "n_clicks"),
         Input("download-tyd-btn", "n_clicks"),
         Input("download-lecturer-btn", "n_clicks")],
        [State("all-timetables-store", "data")],
        prevent_initial_call=True
    )
    def handle_download_actions(sst_clicks, tyd_clicks, lecturer_clicks, all_timetables_data):
        """Handle download button clicks by directly calling export functions"""
        global input_data
        
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        try:
            # Import the TimetableExporter class
            from output_data import TimetableExporter
            
            # Initialize exporter with current session data
            exporter = TimetableExporter()
            
            # Convert the stored timetable data to the format expected by exporter
            if all_timetables_data:
                # Save the current timetables to the exporter's data source
                # The exporter will read from this data
                exporter.timetable_data = all_timetables_data
                exporter.input_data = input_data
            
            if trigger_id == "download-sst-btn" and sst_clicks:
                # Call the export function directly
                if all_timetables_data:
                    success, message = exporter.export_sst_timetables(all_timetables_data)
                else:
                    success, message = False, "No timetable data available"
                
                if success:
                    return html.Div([
                        html.Span("✅ ", style={"fontSize": "16px"}),
                        html.Span(message)
                    ], style={"color": "#28a745", "fontWeight": "600"})
                else:
                    return html.Div([
                        html.Span("❌ ", style={"fontSize": "16px"}),
                        html.Span(message)
                    ], style={"color": "#dc3545", "fontWeight": "600"})
                    
            elif trigger_id == "download-tyd-btn" and tyd_clicks:
                # Call the export function directly
                if all_timetables_data:
                    success, message = exporter.export_tyd_timetables(all_timetables_data)
                else:
                    success, message = False, "No timetable data available"
                
                if success:
                    return html.Div([
                        html.Span("✅ ", style={"fontSize": "16px"}),
                        html.Span(message)
                    ], style={"color": "#28a745", "fontWeight": "600"})
                else:
                    return html.Div([
                        html.Span("❌ ", style={"fontSize": "16px"}),
                        html.Span(message)
                    ], style={"color": "#dc3545", "fontWeight": "600"})
                    
            elif trigger_id == "download-lecturer-btn" and lecturer_clicks:
                # Call the export function directly
                if all_timetables_data:
                    success, message = exporter.export_lecturer_timetables(all_timetables_data)
                else:
                    success, message = False, "No timetable data available"
                
                if success:
                    return html.Div([
                        html.Span("✅ ", style={"fontSize": "16px"}),
                        html.Span(message)
                    ], style={"color": "#28a745", "fontWeight": "600"})
                else:
                    return html.Div([
                        html.Span("❌ ", style={"fontSize": "16px"}),
                        html.Span(message)
                    ], style={"color": "#dc3545", "fontWeight": "600"})
        
        except ImportError as ie:
            return html.Div([
                html.Span("❌ ", style={"fontSize": "16px"}),
                html.Span(f"Import Error: {str(ie)}. Ensure output_data.py is accessible.")
            ], style={"color": "#dc3545", "fontWeight": "600"})
        except Exception as e:
            return html.Div([
                html.Span("❌ ", style={"fontSize": "16px"}),
                html.Span(f"Download Error: {str(e)}")
            ], style={"color": "#dc3545", "fontWeight": "600", "wordWrap": "break-word"})
        
        raise dash.exceptions.PreventUpdate
    # Client-side callback for drag and drop functionality
    app.clientside_callback(
        """
        function(trigger) {
            console.log('Setting up drag and drop functionality...');
            
            window.draggedElement = null;
            window.dragStartData = null;
            window.selectedCell = null;
            
            function setupDragAndDrop() {
                const cells = document.querySelectorAll('.cell');
                console.log('Found', cells.length, 'draggable cells');
                
                cells.forEach(function(cell) {
                    cell.ondragstart = null;
                    cell.ondragover = null;
                    cell.ondragenter = null;
                    cell.ondragleave = null;
                    cell.ondrop = null;
                    cell.ondragend = null;
                    cell.ondblclick = null;
                    
                    cell.classList.remove('dragging', 'drag-over');
                    
                    // Double-click handler for room selection
                    cell.ondblclick = function(e) {
                        e.preventDefault();
                        e.stopPropagation();
                        
                        if (this.classList.contains('break-time') || 
                            this.textContent.trim() === 'BREAK') {
                            console.log('Cannot select room for break time');
                            return;
                        }
                        
                        window.selectedCell = this;
                        
                        const modal = document.getElementById('room-selection-modal');
                        const overlay = document.getElementById('modal-overlay');
                        const deleteBtn = document.getElementById('room-delete-btn');
                        
                        if (modal && overlay) {
                            modal.style.display = 'block';
                            overlay.style.display = 'block';
                            
                            if (deleteBtn) {
                                if (this.classList.contains('manual-schedule')) {
                                    deleteBtn.style.display = 'inline-block';
                                } else {
                                    deleteBtn.style.display = 'none';
                                }
                            }
                            
                            window.dash_clientside.set_props("room-change-data", {
                                data: {
                                    action: 'show_modal',
                                    cell_id: this.id,
                                    cell_content: this.textContent.trim(),
                                    is_manual: this.classList.contains('manual-schedule'),
                                    timestamp: Date.now()
                                }
                            });
                        }
                    };
                    
                    // Drag and drop handlers
                    cell.ondragstart = function(e) {
                        if (this.classList.contains('break-time') || this.textContent.trim() === 'BREAK') {
                            e.preventDefault();
                            return false;
                        }
                        
                        window.draggedElement = this;
                        
                        const idStr = this.id;
                        try {
                            const idObj = JSON.parse(idStr);
                            window.dragStartData = {
                                group: idObj.group,
                                row: idObj.row,
                                col: idObj.col,
                                content: this.textContent.trim(),
                                cellId: idStr
                            };
                        } catch (e) {
                            window.draggedElement = null;
                            window.dragStartData = null;
                            return false;
                        }
                        
                        this.classList.add('dragging');
                        e.dataTransfer.effectAllowed = 'move';
                        e.dataTransfer.setData('text/html', this.id);
                    };
                    
                    cell.ondragover = function(e) {
                        e.preventDefault();
                        e.dataTransfer.dropEffect = 'move';
                        return false;
                    };
                    
                    cell.ondragenter = function(e) {
                        e.preventDefault();
                        if (this !== window.draggedElement && 
                            !this.classList.contains('break-time') && 
                            this.textContent.trim() !== 'BREAK') {
                            this.classList.add('drag-over');
                        }
                        return false;
                    };
                    
                    cell.ondragleave = function(e) {
                        this.classList.remove('drag-over');
                    };
                    
                    cell.ondrop = function(e) {
                        e.preventDefault();
                        e.stopPropagation();
                        
                        if (this.classList.contains('break-time') || this.textContent.trim() === 'BREAK') {
                            this.classList.remove('drag-over');
                            return false;
                        }
                        
                        if (!window.draggedElement || !window.dragStartData) {
                            this.classList.remove('drag-over');
                            return false;
                        }
                        
                        if (this === window.draggedElement) {
                            this.classList.remove('drag-over');
                            return false;
                        }
                        
                        const targetIdStr = this.id;
                        try {
                            const targetIdObj = JSON.parse(targetIdStr);
                            const targetData = {
                                group: targetIdObj.group,
                                row: targetIdObj.row,
                                col: targetIdObj.col,
                                content: this.textContent.trim(),
                                cellId: targetIdStr
                            };
                            
                            if (window.dragStartData.group !== targetData.group) {
                                this.classList.remove('drag-over');
                                return false;
                            }
                            
                            const sourceIsManual = window.draggedElement.classList.contains('manual-schedule');
                            const targetIsManual = this.classList.contains('manual-schedule');
                            
                            const tempContent = window.draggedElement.textContent;
                            window.draggedElement.textContent = this.textContent;
                            this.textContent = tempContent;
                            
                            if (sourceIsManual) {
                                this.classList.add('manual-schedule');
                                window.draggedElement.classList.remove('manual-schedule');
                            }
                            if (targetIsManual) {
                                window.draggedElement.classList.add('manual-schedule');
                                this.classList.remove('manual-schedule');
                            }
                            
                            window.dash_clientside.set_props("swap-data", {
                                data: {
                                    source: window.dragStartData,
                                    target: targetData,
                                    sourceIsManual: sourceIsManual,
                                    targetIsManual: targetIsManual,
                                    timestamp: Date.now()
                                }
                            });
                            
                            const feedback = document.getElementById('feedback');
                            if (feedback) {
                                feedback.innerHTML = 'Swapped "' + window.dragStartData.content + '" with "' + targetData.content + '"';
                                feedback.style.color = 'green';
                                feedback.style.backgroundColor = '#e8f5e8';
                                feedback.style.padding = '10px';
                                feedback.style.borderRadius = '5px';
                                feedback.style.border = '2px solid #4caf50';
                            }
                            
                            window.draggedElement = null;
                            window.dragStartData = null;
                            
                        } catch (e) {
                            console.error('Could not parse target ID:', targetIdStr, e);
                        }
                        
                        this.classList.remove('drag-over');
                        return false;
                    };
                    
                    cell.ondragend = function(e) {
                        this.classList.remove('dragging');
                        
                        const allCells = document.querySelectorAll('.cell');
                        allCells.forEach(function(c) {
                            c.classList.remove('drag-over', 'dragging');
                        });
                        
                        window.draggedElement = null;
                        window.dragStartData = null;
                    };
                });
            }
            
            setTimeout(function() {
                setupDragAndDrop();
            }, 100);
            
            const observer = new MutationObserver(function(mutations) {
                let shouldSetup = false;
                mutations.forEach(function(mutation) {
                    if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                        for (let i = 0; i < mutation.addedNodes.length; i++) {
                            const node = mutation.addedNodes[i];
                            if (node.nodeType === 1 && (node.classList.contains('cell') || node.querySelector('.cell'))) {
                                shouldSetup = true;
                                break;
                            }
                        }
                    }
                });
                if (shouldSetup) {
                    setTimeout(function() {
                        if (!window.draggedElement && !window.dragStartData) {
                            setupDragAndDrop();
                        }
                    }, 150);
                }
            });
            
            const container = document.getElementById('timetable-container');
            if (container) {
                observer.observe(container, {
                    childList: true,
                    subtree: true
                });
            }
            
            return window.dash_clientside.no_update;
        }
        """,
        Output("feedback", "style"),
        Input("trigger", "children"),
        prevent_initial_call=False
    )

# Helper functions
def extract_course_and_faculty_from_cell(cell_content):
    """Extract course code and faculty from cell content with new format: Course Code\\nRoom Name\\nFaculty"""
    if not cell_content or cell_content in ["FREE", "BREAK"]:
        return None, None
    
    lines = cell_content.split('\n')
    course_code = lines[0].strip() if len(lines) > 0 and lines[0].strip() else None
    faculty_name = lines[2].strip() if len(lines) > 2 and lines[2].strip() else None
    
    return course_code, faculty_name

def extract_room_from_cell(cell_content):
    """Extract room name from cell content with new format: Course Code\\nRoom Name\\nFaculty"""
    if not cell_content or cell_content in ["FREE", "BREAK"]:
        return None
    
    lines = cell_content.split('\n')
    if len(lines) > 1 and lines[1].strip():
        return lines[1].strip()
    return None

def extract_course_code_from_cell(cell_content):
    """Extract course code from cell content with new format: Course Code\\nRoom Name\\nFaculty"""
    if not cell_content or cell_content in ["FREE", "BREAK"]:
        return None
    
    lines = cell_content.split('\n')
    if lines and lines[0].strip():
        return lines[0].strip()
    return None

def detect_conflicts(all_timetables_data, current_group_idx):
    """Detect both room conflicts and lecturer conflicts across all student groups at each time slot"""
    conflicts = {}
    
    if not all_timetables_data:
        return conflicts
    
    current_timetable = all_timetables_data[current_group_idx]['timetable']
    
    for row_idx in range(len(current_timetable)):
        for col_idx in range(1, len(current_timetable[row_idx])):
            timeslot_key = f"{row_idx}_{col_idx-1}"
            room_usage = {}
            lecturer_usage = {}
            
            for group_idx, timetable_data in enumerate(all_timetables_data):
                timetable_rows = timetable_data['timetable']
                if row_idx < len(timetable_rows) and col_idx < len(timetable_rows[row_idx]):
                    cell_content = timetable_rows[row_idx][col_idx]
                    
                    if cell_content and cell_content not in ["FREE", "BREAK"]:
                        room_name = extract_room_from_cell(cell_content)
                        course_code, faculty_name = extract_course_and_faculty_from_cell(cell_content)
                        
                        group_name = timetable_data['student_group']['name'] if isinstance(timetable_data['student_group'], dict) else timetable_data['student_group'].name
                        
                        if room_name:
                            if room_name not in room_usage:
                                room_usage[room_name] = []
                            room_usage[room_name].append((group_idx, group_name, cell_content))
                        
                        if faculty_name and faculty_name != "Unknown":
                            if faculty_name not in lecturer_usage:
                                lecturer_usage[faculty_name] = []
                            lecturer_usage[faculty_name].append((group_idx, group_name, cell_content))
            
            # Check for room conflicts
            for room_name, usage_list in room_usage.items():
                if len(usage_list) > 1:
                    for group_idx, group_name, cell_content in usage_list:
                        if group_idx == current_group_idx:
                            conflicts[timeslot_key] = {
                                'type': 'room',
                                'resource': room_name,
                                'conflicting_groups': [u for u in usage_list if u[0] != current_group_idx]
                            }
            
            # Check for lecturer conflicts
            for faculty_name, usage_list in lecturer_usage.items():
                if len(usage_list) > 1:
                    for group_idx, group_name, cell_content in usage_list:
                        if group_idx == current_group_idx:
                            if timeslot_key in conflicts:
                                conflicts[timeslot_key]['type'] = 'both'
                                conflicts[timeslot_key]['lecturer'] = faculty_name
                                conflicts[timeslot_key]['lecturer_conflicting_groups'] = [u for u in usage_list if u[0] != current_group_idx]
                            else:
                                conflicts[timeslot_key] = {
                                    'type': 'lecturer',
                                    'resource': faculty_name,
                                    'conflicting_groups': [u for u in usage_list if u[0] != current_group_idx]
                                }
    
    return conflicts

def create_errors_modal_content(constraint_details, expanded_constraint=None, toggle_constraint=None):
    """Create the content for the errors modal with constraint dropdowns"""
    if not constraint_details:
        return [html.Div("No constraint violation data available.", style={"padding": "20px", "textAlign": "center"})]
    
    constraint_mapping = {
        'Same Student Group Overlaps': 'Same Student Group Overlaps',
        'Different Student Group Overlaps': 'Different Student Group Overlaps', 
        'Lecturer Clashes': 'Lecturer Clashes',
        'Lecturer Schedule Conflicts (Day/Time)': 'Lecturer Schedule Conflicts (Day/Time)',
        'Lecturer Workload Violations': 'Lecturer Workload Violations',
        'Consecutive Slot Violations': 'Consecutive Slot Violations',
        'Missing or Extra Classes': 'Missing or Extra Classes',
        'Same Course in Multiple Rooms on Same Day': 'Same Course in Multiple Rooms on Same Day',
        'Room Capacity/Type Conflicts': 'Room Capacity/Type Conflicts',
        'Classes During Break Time': 'Classes During Break Time'
    }
    
    content = []
    
    for display_name, internal_name in constraint_mapping.items():
        violations = constraint_details.get(internal_name, [])
        count = len(violations)
        
        count_class = "constraint-count zero" if count == 0 else "constraint-count non-zero"
        
        header = html.Div([
            html.Span(display_name, style={"flex": "1"}),
            html.Span(f"{count} Occurrence{'s' if count != 1 else ''}", className=count_class)
        ], className="constraint-header", style={
            "backgroundColor": "#f8f9fa",
            "padding": "12px 16px",
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "center",
            "fontWeight": "600",
            "fontSize": "14px",
            "color": "#11214D",
            "borderBottom": "1px solid #e0e0e0"
        })
        
        details_content = []
        if violations:
            for violation in violations:
                if internal_name == 'Same Student Group Overlaps':
                    details_content.append(html.Div(
                        f"Group '{violation['group']}' has clashing courses {', '.join(violation['courses'])} on {violation['location']}",
                        style={"padding": "10px 16px", "borderBottom": "1px solid #f0f0f0", "fontSize": "13px", "lineHeight": "1.4", "color": "#666"}
                    ))
                elif internal_name == 'Different Student Group Overlaps':
                    if 'groups' in violation:
                        details_content.append(html.Div(
                            f"Room conflict in {violation['room']} at {violation['location']}: Groups {', '.join(violation['groups'])} both scheduled",
                            style={"padding": "10px 16px", "borderBottom": "1px solid #f0f0f0", "fontSize": "13px", "lineHeight": "1.4", "color": "#666"}
                        ))
                elif internal_name == 'Lecturer Clashes':
                    if 'groups' in violation and len(violation['groups']) >= 2:
                        details_content.append(html.Div(
                            f"Lecturer '{violation['lecturer']}' has clashing courses {violation['courses'][0]} for group {violation['groups'][0]}, and {violation['courses'][1]} for group {violation['groups'][1]} on {violation['location']}",
                            style={"padding": "10px 16px", "borderBottom": "1px solid #f0f0f0", "fontSize": "13px", "lineHeight": "1.4", "color": "#666"}
                        ))
        else:
            details_content.append(html.Div("No violations found.", style={"padding": "10px 16px", "color": "#28a745", "fontStyle": "italic", "fontSize": "13px"}))
        
        details = html.Div(details_content, style={"background": "white"})
        
        content.append(html.Div([header, details], style={"marginBottom": "15px", "border": "1px solid #e0e0e0", "borderRadius": "8px", "overflow": "hidden"}))
    
    return content

# Test function for standalone execution
if __name__ == "__main__":
    # Create test session data
    test_session_data = {
        "version": "2.0",
        "timetables": [
            {
                "student_group": {"name": "Test Group 1"},
                "timetable": [
                    ["9:00", "Course A\nRoom 101\nDr. Smith", "FREE", "Course B\nRoom 102\nDr. Jones", "FREE", "FREE"],
                    ["10:00", "FREE", "Course A\nRoom 101\nDr. Smith", "FREE", "Course B\nRoom 102\nDr. Jones", "FREE"],
                    ["11:00", "BREAK", "BREAK", "BREAK", "BREAK", "BREAK"],
                    ["12:00", "Course C\nRoom 103\nDr. Brown", "FREE", "Course D\nRoom 104\nDr. Wilson", "FREE", "FREE"]
                ]
            }
        ],
        "input_data": {
            "courses": [{"id": 1, "name": "Course A", "student_groupsID": [1], "facultyId": 1}],
            "rooms": [{"Id": 1, "name": "Room 101", "capacity": 30, "building": "Main", "room_type": "Lecture"}],
            "studentgroups": [{"id": 1, "name": "Test Group 1"}],
            "faculties": [{"id": 1, "name": "Dr. Smith"}]
        },
        "constraint_details": {},
        "upload_id": "test"
    }
    
    with open("test_session.json", "w") as f:
        json.dump(test_session_data, f)
    
    test_app = create_dash_app("test_session.json")
    if test_app:
        print("Test app created successfully - running on http://localhost:8050")
        test_app.run(debug=True, port=8050, host='0.0.0.0')
    else:
        print("Failed to create test app")