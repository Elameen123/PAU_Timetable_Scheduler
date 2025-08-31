import dash
from dash import dcc, html, Input, Output, State, clientside_callback
from dash.dependencies import ALL
import json

# Sample timetable data for demonstration
sample_data = [
    ["MATH101 - Room A", "ENG102 - Room B", "PHY103 - Room C", "CHE104 - Room D", "BIO105 - Room E"],
    ["CSC201 - Room A", "ENG202 - Room B", "PHY203 - Room C", "CHE204 - Room D", "BIO205 - Room E"],
    ["MATH301 - Room A", "ENG302 - Room B", "PHY303 - Room C", "CHE304 - Room D", "BIO305 - Room E"],
    ["CSC401 - Room A", "ENG402 - Room B", "PHY403 - Room C", "CHE404 - Room D", "BIO405 - Room E"],
    ["MATH501 - Room A", "ENG502 - Room B", "PHY503 - Room C", "CHE504 - Room D", "BIO505 - Room E"]
]
days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
hours = ["9:00", "10:00", "11:00", "12:00", "13:00"]

app = dash.Dash(__name__)

# Add CSS styles to the app
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .cell {
                padding: 15px;
                border: 2px solid #ddd;
                border-radius: 8px;
                cursor: grab;
                min-height: 60px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: 500;
                transition: all 0.3s ease;
                user-select: none;
            }
            .cell:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }
            .cell.dragging {
                opacity: 0.5;
                transform: rotate(5deg);
                cursor: grabbing;
            }
            .cell.drag-over {
                background-color: #ffeb3b !important;
                transform: scale(1.05);
                box-shadow: 0 4px 15px rgba(255, 193, 7, 0.8);
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

app.layout = html.Div([
    html.H1("Drag & Drop Timetable", style={"textAlign": "center", "marginBottom": "30px", "color": "#2c3e50"}),
    
    # Store for timetable data
    dcc.Store(id="timetable-store", data=sample_data),
    
    # Store for communicating swaps
    dcc.Store(id="swap-data", data=None),
    
    # Hidden div to trigger the setup
    html.Div(id="trigger", style={"display": "none"}),
    
    # Timetable container
    html.Div(id="timetable-container"),
    
    # Feedback area
    html.Div(id="feedback", style={
        "marginTop": "20px", 
        "textAlign": "center", 
        "fontSize": "16px", 
        "fontWeight": "bold",
        "minHeight": "30px"
    })
])

@app.callback(
    [Output("timetable-container", "children"),
     Output("trigger", "children")],
    Input("timetable-store", "data")
)
def create_timetable(data):
    # Create table rows
    rows = []
    
    # Header row
    header_cells = [html.Th("Time", style={
        "backgroundColor": "#34495e", 
        "color": "white", 
        "padding": "15px",
        "fontWeight": "bold"
    })]
    
    for day in days_of_week:
        header_cells.append(html.Th(day, style={
            "backgroundColor": "#34495e", 
            "color": "white", 
            "padding": "15px",
            "fontWeight": "bold"
        }))
    
    rows.append(html.Thead(html.Tr(header_cells)))
    
    # Data rows
    body_rows = []
    colors = ["#e3f2fd", "#e8f5e8", "#f3e5f5", "#e0f2f1", "#fff3e0"]
    
    for row_idx in range(len(hours)):
        cells = [html.Td(hours[row_idx], style={
            "backgroundColor": "#2c3e50",
            "color": "white",
            "padding": "15px",
            "fontWeight": "bold",
            "textAlign": "center"
        })]
        
        for col_idx in range(len(days_of_week)):
            cell_id = {"type": "cell", "row": row_idx, "col": col_idx}
            cells.append(
                html.Td(
                    html.Div(
                        data[row_idx][col_idx],
                        id=cell_id,
                        className="cell",
                        draggable="true",
                        style={"backgroundColor": colors[col_idx]},
                        n_clicks=0  # Add this to make it interactive
                    ),
                    style={"padding": "5px"}
                )
            )
        
        body_rows.append(html.Tr(cells))
    
    rows.append(html.Tbody(body_rows))
    
    table = html.Table(rows, style={
        "width": "100%",
        "borderCollapse": "collapse",
        "boxShadow": "0 4px 8px rgba(0,0,0,0.1)",
        "borderRadius": "10px",
        "overflow": "hidden"
    })
    
    return table, "trigger"

# Use a clientside callback to handle the drag and drop
clientside_callback(
    """
    function(trigger) {
        console.log('Setting up drag and drop...');
        
        // Global variables for drag state
        window.draggedElement = null;
        window.dragStartData = null;
        
        function setupDragAndDrop() {
            const cells = document.querySelectorAll('.cell');
            console.log('Found', cells.length, 'draggable cells');
            
            cells.forEach(function(cell) {
                // Clear existing listeners
                cell.ondragstart = null;
                cell.ondragover = null;
                cell.ondragenter = null;
                cell.ondragleave = null;
                cell.ondrop = null;
                cell.ondragend = null;
                
                cell.ondragstart = function(e) {
                    console.log('Drag started');
                    window.draggedElement = this;
                    
                    // Get row and col from the element's ID
                    const idStr = this.id;
                    try {
                        const idObj = JSON.parse(idStr);
                        window.dragStartData = {
                            row: idObj.row,
                            col: idObj.col,
                            content: this.textContent.trim()
                        };
                        console.log('Drag data:', window.dragStartData);
                    } catch (e) {
                        console.error('Could not parse ID:', idStr);
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
                    if (this !== window.draggedElement) {
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
                    
                    console.log('Drop detected');
                    
                    if (window.draggedElement && this !== window.draggedElement) {
                        // Get target data
                        const targetIdStr = this.id;
                        try {
                            const targetIdObj = JSON.parse(targetIdStr);
                            const targetData = {
                                row: targetIdObj.row,
                                col: targetIdObj.col,
                                content: this.textContent.trim()
                            };
                            
                            console.log('Swapping:', window.dragStartData, 'with:', targetData);
                            
                            // Perform the swap
                            const tempContent = window.draggedElement.textContent;
                            window.draggedElement.textContent = this.textContent;
                            this.textContent = tempContent;
                            
                            // Update feedback
                            const feedback = document.getElementById('feedback');
                            if (feedback) {
                                feedback.innerHTML = '✅ Swapped "' + window.dragStartData.content + '" with "' + targetData.content + '"';
                                feedback.style.color = 'green';
                                feedback.style.backgroundColor = '#e8f5e8';
                                feedback.style.padding = '10px';
                                feedback.style.borderRadius = '5px';
                                feedback.style.border = '2px solid #4caf50';
                            }
                            
                            console.log('Swap completed successfully');
                            
                        } catch (e) {
                            console.error('Could not parse target ID:', targetIdStr);
                        }
                    }
                    
                    this.classList.remove('drag-over');
                    return false;
                };
                
                cell.ondragend = function(e) {
                    console.log('Drag ended');
                    this.classList.remove('dragging');
                    
                    // Clean up all drag-over classes
                    const cells = document.querySelectorAll('.cell');
                    cells.forEach(function(c) {
                        c.classList.remove('drag-over');
                    });
                    
                    window.draggedElement = null;
                    window.dragStartData = null;
                };
            });
        }
        
        // Setup immediately
        setTimeout(setupDragAndDrop, 100);
        
        // Also setup when DOM changes
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
                setTimeout(setupDragAndDrop, 100);
            }
        });
        
        const container = document.getElementById('timetable-container');
        if (container) {
            observer.observe(container, {
                childList: true,
                subtree: true
            });
        }
        
        console.log('Drag and drop setup complete');
        return window.dash_clientside.no_update;
    }
    """,
    Output("swap-data", "data"),
    Input("trigger", "children"),
    prevent_initial_call=False
)

# Removed the problematic callback that was causing circular dependency
# The drag and drop now works entirely on the client side

if __name__ == "__main__":
    app.run(debug=True)