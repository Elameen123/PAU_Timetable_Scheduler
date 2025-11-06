import os
import json
import io
import dash
import pandas as pd
import shutil
import traceback
import time
import dash.exceptions
import dash.dependencies
from dash import dcc, html, Input, Output, State, clientside_callback, ALL, MATCH

# Optional: import input_data to support lecturer lookups like the original UI
try:
    from input_data import input_data
except Exception:
    input_data = None


def _load_rooms_data():
    try:
        path = os.path.join(os.path.dirname(__file__), 'data', 'rooms-data.json')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return []


def _load_saved_timetable():
    """Prefer fresh_timetable_data.json, else fallback to backend-saved timetable_data.json."""
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    session_path = os.path.join(data_dir, 'timetable_data.json')
    fresh_path = os.path.join(data_dir, 'fresh_timetable_data.json')

    # Prefer fresh results from the latest optimization
    try:
        if os.path.exists(fresh_path):
            with open(fresh_path, 'r', encoding='utf-8') as f:
                saved = json.load(f)
            # fresh is expected to be a list of timetables
            if isinstance(saved, list):
                return saved, []
            # handle legacy dict formats defensively
            if isinstance(saved, dict) and 'timetables' in saved:
                return saved.get('timetables', []), saved.get('manual_cells', []) or []
    except Exception:
        pass

    # Fallback to last session's saved data (includes manual_cells)
    try:
        if os.path.exists(session_path):
            with open(session_path, 'r', encoding='utf-8') as f:
                saved = json.load(f)
            if isinstance(saved, dict) and 'timetables' in saved:
                return saved.get('timetables', []), saved.get('manual_cells', [])
            # old format: list
            if isinstance(saved, list):
                return saved, []
    except Exception:
        pass

    return [], []


def _load_constraint_details():
    """Load constraint details produced by the backend optimizer if available."""
    try:
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        # Backend persists violations here
        path = os.path.join(data_dir, 'constraint_violations.json')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                details = json.load(f)
            if isinstance(details, dict):
                return details
    except Exception:
        pass
    return {}


def _parse_cell(cell_content):
    """Robustly parse a timetable cell into (course, room, lecturer).
    Supports both newline-separated and comma-separated 'Course:/Lecturer:/Room:' formats
    and any ordering of those parts. Returns (None, None, None) for FREE/BREAK/empty."""
    if not cell_content:
        return None, None, None
    text = str(cell_content).strip()
    if text.upper() in ("FREE", "BREAK"):
        return None, None, None
    # Normalize html breaks just in case
    text = text.replace('<br>', ', ').replace('<br/>', ', ').replace('<br />', ', ')

    course = None
    room = None
    lecturer = None

    # Try newline format first
    if "\n" in text and not (', Lecturer:' in text or ', Room:' in text or ', Course:' in text):
        parts = [p.strip() for p in text.split('\n') if p.strip()]
        # Assume line order if prefixes are missing
        if parts:
            # With prefixes
            for p in parts:
                low = p.lower()
                if low.startswith('course:') and not course:
                    course = p.split(':', 1)[1].strip()
                elif low.startswith('room:') and not room:
                    room = p.split(':', 1)[1].strip()
                elif low.startswith('lecturer:') and not lecturer:
                    lecturer = p.split(':', 1)[1].strip()
            # If prefixes not present, fallback to positional
            if course is None:
                course = parts[0] if len(parts) >= 1 else None
            if room is None:
                room = parts[1] if len(parts) >= 2 else None
            if lecturer is None:
                lecturer = parts[2] if len(parts) >= 3 else None
    else:
        # Comma-separated form: "Course: X, Lecturer: Y, Room: Z" (order can vary)
        parts = [p.strip() for p in text.split(',') if p.strip()]
        for p in parts:
            low = p.lower()
            if low.startswith('course:') and not course:
                course = p.split(':', 1)[1].strip()
            elif low.startswith('lecturer:') and not lecturer:
                lecturer = p.split(':', 1)[1].strip()
            elif low.startswith('room:') and not room:
                room = p.split(':', 1)[1].strip()
        # If some parts missing prefixes, try positional heuristic
        if course is None and parts:
            # If first part has no prefix, treat as course
            if ':' not in parts[0]:
                course = parts[0]
        if lecturer is None:
            for p in parts:
                if 'lecturer' in p.lower():
                    try:
                        lecturer = p.split(':', 1)[1].strip()
                        break
                    except Exception:
                        pass
        if room is None:
            for p in parts:
                if 'room' in p.lower():
                    try:
                        room = p.split(':', 1)[1].strip()
                        break
                    except Exception:
                        pass

    # Normalize empty strings to None
    course = course if course and course.lower() != 'unknown' else None
    room = room if room and room.lower() != 'unknown' else None
    lecturer = lecturer if lecturer and lecturer.lower() != 'unknown' else None
    return course, room, lecturer


def _extract_room(cell_content):
    if not cell_content or str(cell_content).strip().upper() in ("FREE", "BREAK"):
        return None
    _, room, _ = _parse_cell(cell_content)
    return room


def _extract_course_and_faculty(cell_content):
    if not cell_content or str(cell_content).strip().upper() in ("FREE", "BREAK"):
        return None, None
    course, room, faculty = _parse_cell(cell_content)
    return course, faculty


def _detect_conflicts(all_timetables, current_group_idx):
    conflicts = {}
    if not all_timetables:
        return conflicts

    current_timetable = all_timetables[current_group_idx]['timetable']
    for row_idx in range(len(current_timetable)):
        # Determine number of day columns from this row
        col_count = max(0, len(current_timetable[row_idx]) - 1)
        for col_idx in range(1, 1 + col_count):
            key = f"{row_idx}_{col_idx-1}"
            room_usage = {}
            lecturer_usage = {}
            for g_idx, tdata in enumerate(all_timetables):
                rows = tdata['timetable']
                if row_idx < len(rows) and col_idx < len(rows[row_idx]):
                    cell = rows[row_idx][col_idx]
                    if cell and str(cell).strip().upper() not in ("FREE", "BREAK"):
                        course, room, faculty = _parse_cell(cell)
                        if room:
                            room_key = str(room).strip().upper()
                            info = room_usage.setdefault(room_key, {"users": [], "display": room})
                            info["users"].append(g_idx)
                        if faculty:
                            lec_key = faculty.strip().upper()
                            info = lecturer_usage.setdefault(lec_key, {"users": [], "display": faculty})
                            info["users"].append(g_idx)
            # room conflicts
            for r_key, info in room_usage.items():
                users = info.get("users", [])
                if len(users) > 1 and current_group_idx in users:
                    conflicts[key] = {'type': 'room', 'resource': info.get("display", r_key)}
            # lecturer conflicts
            for l_key, info in lecturer_usage.items():
                users = info.get("users", [])
                if len(users) > 1 and current_group_idx in users:
                    if key in conflicts:
                        conflicts[key]['type'] = 'both'
                        conflicts[key]['lecturer'] = info.get("display", l_key)
                    else:
                        conflicts[key] = {'type': 'lecturer', 'resource': info.get("display", l_key)}
    return conflicts


# --- OG-compatible helper wrappers/names ---
extract_room_from_cell = _extract_room
extract_course_and_faculty_from_cell = _extract_course_and_faculty

def extract_course_code_from_cell(cell_content):
    if not cell_content or cell_content in ["FREE", "BREAK"]:
        return None
    lines = str(cell_content).split('\n')
    return lines[0].strip() if lines and lines[0].strip() else None

def update_room_in_cell_content(cell_content, new_room):
    """Update the room in cell content, handling both newline and comma-separated formats"""
    if not cell_content or cell_content in ["FREE", "BREAK"]:
        return cell_content
    
    text = str(cell_content).strip()
    
    # Check if it's comma-separated format
    if ', Lecturer:' in text or ', Room:' in text or ', Course:' in text:
        # Parse the comma-separated format
        course, room, lecturer = _parse_cell(text)
        # Return in newline format: Course\nRoom\nLecturer
        course_str = course if course else "Unknown Course"
        lecturer_str = lecturer if lecturer else "Unknown"
        return f"{course_str}\n{new_room}\n{lecturer_str}"
    
    # Handle newline-separated format
    lines = text.split('\n')
    if len(lines) >= 3:
        lines[1] = new_room
        return '\n'.join(lines)
    elif len(lines) == 2:
        return f"{lines[0]}\n{new_room}\n{lines[1]}"
    elif len(lines) == 1:
        # Single line - parse and reconstruct
        course, _, lecturer = _parse_cell(text)
        course_str = course if course else lines[0]
        lecturer_str = lecturer if lecturer else "Unknown"
        return f"{course_str}\n{new_room}\n{lecturer_str}"
    
    return cell_content

def update_consecutive_course_rooms(timetables_data, group_idx, course_code, new_room, current_row, current_col):
    rows = timetables_data[group_idx]['timetable']
    for r in range(len(rows)):
        if r == current_row:
            continue
        cell = rows[r][current_col + 1]
        if extract_course_code_from_cell(cell) == course_code:
            rows[r][current_col + 1] = update_room_in_cell_content(cell, new_room)

def find_lecturer_for_course(course_identifier, group_name):
    """Find the lecturer assigned to teach a specific course for a specific group - SEPT 13 style"""
    try:
        if not input_data:
            return "Unknown"
        
        for student_group in input_data.student_groups:
            if student_group.name == group_name:
                for i, course_id_from_group in enumerate(student_group.courseIDs):
                    course = input_data.getCourse(course_id_from_group)
                    if course and (
                        str(course_id_from_group).strip().lower() == str(course_identifier).strip().lower() or
                        str(getattr(course, 'code', '')).strip().lower() == str(course_identifier).strip().lower() or
                        str(getattr(course, 'name', '')).strip().lower() == str(course_identifier).strip().lower()
                    ):
                        faculty_id = student_group.teacherIDS[i]
                        faculty = input_data.getFaculty(faculty_id)
                        if faculty:
                            return faculty.name if faculty.name else faculty.faculty_id
                        break
        return "Unknown"
    except Exception as e:
        print(f"Error finding lecturer for course {course_identifier}: {e}")
        return "Unknown"


def get_room_usage_at_timeslot(all_timetables_data, row_idx, col_idx):
    usage = {}
    for t in all_timetables_data or []:
        rows = t.get('timetable', [])
        if row_idx < len(rows) and (col_idx + 1) < len(rows[row_idx]):
            cell = rows[row_idx][col_idx + 1]
            _, room, _ = _parse_cell(cell)
            if room:
                name = t['student_group']['name'] if isinstance(t['student_group'], dict) else getattr(t['student_group'], 'name', str(t['student_group']))
                usage.setdefault(room, []).append(name)
    return usage

def save_timetable_to_file(timetables_data, manual_cells_data):
    try:
        save_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(save_dir, exist_ok=True)
        session_path = os.path.join(save_dir, 'timetable_data.json')
        export_path = os.path.join(save_dir, 'fresh_timetable_data.json')
        backup_path = os.path.join(save_dir, 'timetable_data_backup.json')
        if os.path.exists(session_path):
            shutil.copy2(session_path, backup_path)
        data = {'timetables': timetables_data, 'manual_cells': manual_cells_data or []}
        with open(session_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(timetables_data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error auto-saving timetable: {e}")
        traceback.print_exc()
        return False

def recompute_constraint_violations_simplified(timetables_data, rooms_data=None):
    try:
        violations = {
            'Same Student Group Overlaps': [],
            'Different Student Group Overlaps': [],
            'Lecturer Clashes': [],
            'Lecturer Schedule Conflicts (Day/Time)': [],
            'Lecturer Workload Violations': [],
            'Consecutive Slot Violations': [],
            'Missing or Extra Classes': [],
            'Same Course in Multiple Rooms on Same Day': [],
            'Room Capacity/Type Conflicts': [],
            'Classes During Break Time': []
        }

        if not timetables_data:
            return violations

        # Build room lookup by name
        room_lookup = {r['name']: r for r in (rooms_data or []) if isinstance(r, dict) and r.get('name')}
        days_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}

        # Determine grid size dynamically
        max_rows = 0
        max_days = 0
        for t in timetables_data:
            rows = t.get('timetable', []) or []
            if rows:
                max_rows = max(max_rows, len(rows))
                try:
                    row_days = max((len(r) - 1) for r in rows if isinstance(r, list) and len(r) > 1)
                    max_days = max(max_days, row_days)
                except Exception:
                    pass
        if max_rows <= 0 or max_days <= 0:
            return violations

        # Walk the grid and aggregate conflicts
        for time_slot in range(max_rows):
            for day in range(max_days):
                room_usage = {}
                lecturer_usage = {}
                time_label = None

                for timetable in timetables_data:
                    rows = timetable.get('timetable', []) or []
                    if time_slot >= len(rows):
                        continue
                    row = rows[time_slot]
                    # first col is the time label
                    try:
                        time_label = time_label or (row[0] if row and isinstance(row[0], str) else None)
                    except Exception:
                        pass
                    # day+1 is the cell content
                    if (day + 1) >= len(row):
                        continue
                    cell = row[day + 1]
                    if not cell or str(cell).strip().upper() in ['FREE', 'BREAK']:
                        continue

                    course, room, lecturer = _parse_cell(cell)
                    group_name = timetable['student_group']['name'] if isinstance(timetable['student_group'], dict) else getattr(timetable['student_group'], 'name', '')

                    if room:
                        room_usage.setdefault(room, []).append(group_name)
                    if lecturer:
                        lecturer_usage.setdefault(lecturer, []).append({'group': group_name, 'course': course})

                    # Capacity check if we have lookup and input_data
                    try:
                        if room and room in room_lookup and input_data is not None:
                            cap = room_lookup[room].get('capacity', 999999)
                            grp = next((g for g in input_data.student_groups if getattr(g, 'name', '') == group_name), None)
                            if grp and getattr(grp, 'no_students', 0) > cap:
                                violations['Room Capacity/Type Conflicts'].append({
                                    'type': 'Room Capacity Exceeded',
                                    'room': room,
                                    'group': group_name,
                                    'day': days_map.get(day, str(day)),
                                    'time': time_label or f"{time_slot+9}:00",
                                    'students': getattr(grp, 'no_students', 0),
                                    'capacity': cap
                                })
                    except Exception:
                        pass

                # Record room clashes
                for room_name, groups in room_usage.items():
                    if room_name and len(groups) > 1:
                        violations['Different Student Group Overlaps'].append({
                            'room': room_name,
                            'groups': groups,
                            'day': days_map.get(day, str(day)),
                            'time': time_label or f"{time_slot+9}:00",
                            'location': f"{days_map.get(day, str(day))} at {time_label or f'{time_slot+9}:00'}"
                        })

                # Record lecturer clashes
                for lec, sessions in lecturer_usage.items():
                    if len(sessions) > 1:
                        violations['Lecturer Clashes'].append({
                            'lecturer': lec,
                            'day': days_map.get(day, str(day)),
                            'time': time_label or f"{time_slot+9}:00",
                            'courses': [s['course'] for s in sessions],
                            'groups': [s['group'] for s in sessions],
                            'location': f"{days_map.get(day, str(day))} at {time_label or f'{time_slot+9}:00'}"
                        })
        return violations
    except Exception as e:
        print(f"Error computing simplified violations: {e}")
        traceback.print_exc()
        return None


def _make_excel_bytes(all_timetables):
    """Create an in-memory XLSX with one sheet per student group."""
    from openpyxl import Workbook
    wb = Workbook()
    wb.remove(wb.active)
    for t in all_timetables or []:
        name = t['student_group']['name'] if isinstance(t['student_group'], dict) else str(t['student_group'])
        safe_name = (name or 'Group')[:31]
        ws = wb.create_sheet(title=safe_name)
        for row in t.get('timetable', []):
            ws.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.getvalue()


def create_errors_modal_content(constraint_details, expanded_constraint=None, toggle_constraint=None):
    """Create the content for the errors modal with constraint dropdowns - SEPT 13 format"""
    if not constraint_details:
        return [html.Div("No constraint violation data available.", style={"padding": "20px", "textAlign": "center"})]
    
    # Keep track of expanded states (simple approach)
    if not hasattr(create_errors_modal_content, 'expanded_states'):
        create_errors_modal_content.expanded_states = set()
    
    # Toggle the state if a constraint was clicked
    if toggle_constraint:
        if toggle_constraint in create_errors_modal_content.expanded_states:
            create_errors_modal_content.expanded_states.remove(toggle_constraint)
        else:
            create_errors_modal_content.expanded_states.add(toggle_constraint)
    
    # Set initial expanded state if specified
    if expanded_constraint:
        create_errors_modal_content.expanded_states.add(expanded_constraint)
    
    # Mapping from user-friendly names to internal names (SEPT 13 format)
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
        
        # Determine if this dropdown should be expanded
        is_expanded = display_name in create_errors_modal_content.expanded_states
        
        # Determine count badge class
        count_class = "constraint-count zero" if count == 0 else "constraint-count non-zero"
        
        # Create constraint header
        header = html.Div([
            html.Span(display_name, style={"flex": "1"}),
            html.Span(f"{count} Occurrence{'s' if count != 1 else ''}", className=count_class),
            html.Span("^", className=f"constraint-arrow{' rotated' if is_expanded else ''}")
        ], className=f"constraint-header{' active' if is_expanded else ''}", 
           id={"type": "constraint-header", "index": display_name}, n_clicks=0)
        
        # Create constraint details with SEPT 13 formatting
        details_content = []
        if violations:
            for i, violation in enumerate(violations):
                if internal_name == 'Same Student Group Overlaps':
                    details_content.append(html.Div(
                        f"Group '{violation['group']}' has clashing courses {', '.join(violation['courses'])} on {violation['location']}",
                        className="constraint-item"
                    ))
                elif internal_name == 'Different Student Group Overlaps':
                    # Handle both old format (events) and new format (groups)
                    if 'events' in violation:
                        details_content.append(html.Div(
                            f"Room conflict at {violation['location']}: {', '.join(violation['events'])}",
                            className="constraint-item"
                        ))
                    elif 'groups' in violation:
                        details_content.append(html.Div(
                            f"Room conflict in {violation['room']} at {violation['location']}: Groups {', '.join(violation['groups'])} both scheduled",
                            className="constraint-item"
                        ))
                    else:
                        details_content.append(html.Div(
                            f"Room conflict at {violation.get('location', 'Unknown location')}",
                            className="constraint-item"
                        ))
                elif internal_name == 'Lecturer Clashes':
                    # Show both student groups that are clashing
                    if 'groups' in violation and len(violation['groups']) >= 2:
                        details_content.append(html.Div(
                            f"Lecturer '{violation['lecturer']}' has clashing courses {violation['courses'][0]} for group {violation['groups'][0]}, and {violation['courses'][1]} for group {violation['groups'][1]} on {violation['location']}",
                            className="constraint-item"
                        ))
                    else:
                        # Fallback to original format if groups info is missing
                        details_content.append(html.Div(
                            f"Lecturer '{violation['lecturer']}' has clashing courses {', '.join(violation['courses'])} on {violation['location']}",
                            className="constraint-item"
                        ))
                elif internal_name == 'Lecturer Schedule Conflicts (Day/Time)':
                    details_content.append(html.Div(
                        f"Lecturer '{violation['lecturer']}' scheduled for {violation['course']} for group {violation['group']} on {violation['location']} but available: {violation['available_days']} at {violation['available_times']}",
                        className="constraint-item"
                    ))
                elif internal_name == 'Lecturer Workload Violations':
                    if violation['type'] == 'Excessive Daily Hours':
                        courses_text = violation.get('courses', 'Unknown courses')
                        details_content.append(html.Div(
                            f"Lecturer '{violation['lecturer']}' has {violation['hours_scheduled']} hours on {violation['day']} from courses {courses_text}, exceeding maximum of {violation['max_allowed']} hours per day",
                            className="constraint-item"
                        ))
                    elif violation['type'] == 'Excessive Consecutive Hours':
                        courses_text = violation.get('courses', 'Unknown courses')
                        details_content.append(html.Div(
                            f"Lecturer '{violation['lecturer']}' has {violation['consecutive_hours']} consecutive hours on {violation['day']} from courses {courses_text} ({', '.join(violation['hours_times'])}), exceeding maximum of {violation['max_allowed']} consecutive hours",
                            className="constraint-item"
                        ))
                    else:
                        details_content.append(html.Div(
                            f"Lecturer workload violation for {violation['lecturer']} on {violation['day']}: {violation.get('violation', 'Unknown violation')}",
                            className="constraint-item"
                        ))
                elif internal_name == 'Consecutive Slot Violations':
                    details_content.append(html.Div(
                        f"{violation['reason']}: Course '{violation['course']}' ({violation.get('course_name', '')}) for group '{violation['group']}' at times {', '.join(violation['times'])}",
                        className="constraint-item"
                    ))
                elif internal_name == 'Missing or Extra Classes':
                    details_content.append(html.Div(
                        f"{violation['issue']} classes for {violation['location']}: Expected {violation['expected']}, Got {violation['actual']}",
                        className="constraint-item"
                    ))
                elif internal_name == 'Same Course in Multiple Rooms on Same Day':
                    details_content.append(html.Div(
                        f"{violation['location']} in multiple rooms: {', '.join(violation['rooms'])}",
                        className="constraint-item"
                    ))
                elif internal_name == 'Room Capacity/Type Conflicts':
                    if violation['type'] == 'Room Type Mismatch':
                        details_content.append(html.Div(
                            f"Room type mismatch at {violation['location']}: {violation['course']} for group {violation['group']} requires {violation['required_type']} but scheduled in {violation['room']} ({violation['room_type']})",
                            className="constraint-item"
                        ))
                    else:
                        details_content.append(html.Div(
                            f"Room capacity exceeded at {violation['room']} by group {violation['group']} on {violation['day']} at {violation['time']}: {violation['students']} students in {violation['room']} (capacity: {violation['capacity']})",
                            className="constraint-item"
                        ))
                elif internal_name == 'Classes During Break Time':
                    details_content.append(html.Div(
                        f"Class during break time at {violation['location']}: {violation['course']} for {violation['group']}",
                        className="constraint-item"
                    ))
        else:
            details_content.append(html.Div("No violations found.", className="constraint-item", style={"color": "#28a745", "fontStyle": "italic"}))
        
        details = html.Div(details_content, 
                          className=f"constraint-details{' expanded' if is_expanded else ''}", 
                          id={"type": "constraint-details", "index": display_name})
        
        content.append(html.Div([header, details], className="constraint-dropdown"))
    
    return content


def create_app(_ctx: dict | None = None):
    timetables, manual_cells = _load_saved_timetable()
    rooms_data = _load_rooms_data()

    # global-like state for this app instance
    session_state = {'has_swaps': False}

    # Load backend constraint details from DE algorithm (includes Missing or Extra Classes)
    backend_summary = _load_constraint_details()
    
    # Use backend_summary as initial_constraints if available, otherwise compute simplified
    if isinstance(backend_summary, dict) and backend_summary:
        initial_constraints = dict(backend_summary)
    else:
        # Fallback to simplified if constraint_violations.json doesn't exist
        simplified = recompute_constraint_violations_simplified(timetables, rooms_data) or {}
        initial_constraints = dict(simplified)

    # IMPORTANT: No path prefixes here; app is mounted under /interactive by the parent Flask app
    app = dash.Dash(
        __name__,
        assets_url_path="/assets",
        suppress_callback_exceptions=True,
    )

    # Use OG index_string CSS
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
            .cell:hover { transform: translateY(-1px); box-shadow: 0 2px 6px rgba(0,0,0,0.1); border-color: #ccc; }
            .cell.dragging { opacity: 0.6; transform: rotate(2deg); cursor: grabbing; box-shadow: 0 4px 12px rgba(0,0,0,0.2); }
            .cell.drag-over { background-color: #fff3cd !important; border-color: #ffc107 !important; transform: scale(1.02); box-shadow: 0 2px 8px rgba(255, 193, 7, 0.6); }
            .cell.break-time { background-color: #ff5722 !important; color: white; cursor: not-allowed; font-weight: 500; }
            .cell.break-time:hover { transform: none; box-shadow: none; }
            .cell.room-conflict { background-color: #ffebee !important; color: #d32f2f !important; border-color: #f44336 !important; font-weight: 600 !important; }
            .cell.room-conflict:hover { background-color: #ffcdd2 !important; box-shadow: 0 2px 8px rgba(244, 67, 54, 0.4); }
            .cell.lecturer-conflict { background-color: #fff0cc !important; color: #d4942f !important; border-color: #d4942f !important; font-weight: 600 !important; }
            .cell.lecturer-conflict:hover { background-color: #fff8e8 !important; box-shadow: 0 2px 8px rgba(244, 67, 54, 0.4); }
            .cell.both-conflict { background-color: #ffdcec !important; color: #d42fa2 !important; border-color: #d42fa2 !important; font-weight: 600 !important; }
            .cell.both-conflict:hover { background-color: #ffdcec !important; box-shadow: 0 2px 8px rgba(244, 67, 54, 0.4); }
            .cell.manual-schedule { border: 3px solid #72B7F4 !important; box-shadow: 0 0 8px rgba(33, 150, 243, 0.4) !important; background-color: #e3f2fd !important; position: relative; }
            .cell.manual-schedule:hover { box-shadow: 0 2px 12px rgba(33, 150, 243, 0.7) !important; border-color: #1976d2 !important; transform: translateY(-1px); }
            .cell.manual-schedule.room-conflict { background-color: #ffebee !important; border: 3px solid #72B7F4 !important; box-shadow: 0 0 8px rgba(33, 150, 243, 0.5) !important; }
            .cell.manual-schedule.lecturer-conflict { background-color: #fff0cc !important; border: 3px solid #72B7F4 !important; box-shadow: 0 0 8px rgba(33, 150, 243, 0.5) !important; }
            .cell.manual-schedule.both-conflict { background-color: #ffebee !important; border: 3px solid #72B7F4 !important; box-shadow: 0 0 8px rgba(33, 150, 243, 0.5) !important; }
            .room-selection-modal { position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background: white; border-radius: 12px; box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2); padding: 25px; z-index: 1000; max-width: 800px; width: 95%; max-height: 80vh; overflow-y: auto; font-family: 'Poppins', sans-serif; }
            .modal-overlay { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0, 0, 0, 0.5); z-index: 999; }
            .modal-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding-bottom: 15px; border-bottom: 2px solid #f0f0f0; }
            .modal-title { font-size: 18px; font-weight: 600; color: #11214D; margin: 0; }
            .modal-close { background: none; border: none; font-size: 24px; color: #666; cursor: pointer; padding: 5px; border-radius: 50%; transition: all 0.2s ease; }
            .modal-close:hover { background-color: #f5f5f5; color: #333; }
            .room-search { width: calc(100% - 32px); padding: 12px 16px; border: 2px solid #e0e0e0; border-radius: 8px; font-size: 14px; margin-bottom: 15px; transition: border-color 0.2s ease; box-sizing: border-box; }
            .room-search:focus { outline: none; border-color: #11214D; }
            .room-options { max-height: 300px; overflow-y: auto; border: 1px solid #e0e0e0; border-radius: 8px; background: white; }
            .room-option { padding: 12px 16px; border: none; border-bottom: 1px solid #f0f0f0; outline: none; cursor: pointer; transition: background-color 0.2s ease; font-size: 13px; display: flex; justify-content: space-between; align-items: center; }
            .room-option:last-child { border-bottom: none; }
            .room-option:hover { background-color: #f8f9fa; }
            .room-option.available { color: #2e7d32; font-weight: 500; }
            .room-option.occupied { color: #d32f2f; font-weight: 500; }
            .room-option.selected { background-color: #e3f2fd; border-left: 4px solid #11214D; padding-left: 12px; }
            .room-info { font-size: 11px; color: #666; }
            .conflict-warning { position: fixed; top: 20px; left: 20px; background: #ffebee; border: 2px solid #f44336; border-radius: 8px; padding: 15px; max-width: 350px; z-index: 1001; box-shadow: 0 4px 12px rgba(244, 67, 54, 0.3); }
            .conflict-warning-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
            .conflict-warning-title { font-weight: 600; color: #d32f2f; font-size: 14px; }
            .conflict-warning-close { background: none; border: none; font-size: 18px; color: #d32f2f; cursor: pointer; padding: 2px; }
            .conflict-warning-content { color: #b71c1c; font-size: 12px; line-height: 1.4; }
            .student-group-container { margin-bottom: 30px; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; background-color: #fafafa; box-shadow: 0 2px 4px rgba(0,0,0,0.05); max-width: 1200px; margin: 0 auto 30px auto; }
            .dropdown-container { max-width: 1200px; margin: 0 auto; padding: 0 15px; }
            table { font-family: 'Poppins', sans-serif; font-size: 12px; border-collapse: separate; border-spacing: 0; width: 100%; background-color: white; border-radius: 6px; overflow: hidden; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }
            th { background-color: #11214D !important; color: white !important; padding: 12px 10px !important; font-size: 13px !important; font-weight: 600 !important; text-align: center !important; border: 1px solid #0d1a3d !important; }
            td { padding: 0 !important; border: 1px solid #e0e0e0 !important; background-color: white; }
            .time-cell { background-color: #11214D !important; color: white !important; padding: 12px 10px !important; font-weight: 600 !important; text-align: center !important; font-size: 12px !important; border: 1px solid #0d1a3d !important; }
            .timetable-title { color: #11214D; font-weight: 600; margin-bottom: 20px; font-size: 20px; }
            .timetable-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding: 0 10px; }
            .nav-arrows { display: flex; align-items: center; gap: 15px; }
            .nav-arrow { background: #11214D; color: white; border: none; border-radius: 15%; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; cursor: pointer; font-size: 16px; font-weight: bold; transition: all 0.2s ease; }
            .nav-arrow:hover { background: #0d1a3d; transform: scale(1.05); box-shadow: 0 2px 8px rgba(17, 33, 77, 0.3); }
            .nav-arrow:disabled { background: #ccc; cursor: not-allowed; transform: none; box-shadow: none; }
            .save-error { position: fixed; top: 20px; left: 50%; transform: translateX(-50%); background: #ffebee; border: 2px solid #f44336; border-radius: 8px; padding: 15px 20px; max-width: 400px; z-index: 1001; text-align: center; }
            .save-error-title { font-weight: 600; color: #d32f2f; font-size: 14px; margin-bottom: 8px; }
            .save-error-content { color: #b71c1c; font-size: 12px; line-height: 1.4; }
            .constraint-dropdown {
                margin-bottom: 15px;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                overflow: hidden;
            }
            .constraint-header {
                background-color: #f8f9fa;
                padding: 12px 16px;
                cursor: pointer;
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-weight: 600;
                font-size: 14px;
                color: #11214D;
                border-bottom: 1px solid #e0e0e0;
                transition: background-color 0.2s ease;
                gap: 15px;
            }
            .constraint-header:hover {
                background-color: #e9ecef;
            }
            .constraint-header.active {
                background-color: #11214D;
                color: white;
            }
            .constraint-count {
                color: white;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 12px;
                font-weight: 600;
            }
            .constraint-count.zero {
                background-color: #28a745;
            }
            .constraint-count.non-zero {
                background-color: #dc3545;
            }
            .constraint-header.active .constraint-count {
                background-color: rgba(255, 255, 255, 0.2);
            }
            .constraint-details {
                padding: 0;
                max-height: 0;
                overflow: hidden;
                transition: max-height 0.3s ease-out;
                background: white;
            }
            .constraint-details.expanded {
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
            .errors-button { position: relative; background: #dc3545; color: white; border: none; border-radius: 8px; padding: 10px 20px; font-size: 14px; font-weight: 600; cursor: pointer; box-shadow: 0 2px 4px rgba(220, 53, 69, 0.3); }
            .error-notification { position: absolute; top: -8px; right: -8px; background: rgba(255, 255, 255, 0.9); color: #dc3545; border: 2px solid #dc3545; border-radius: 50%; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 700; }
            .help-modal { position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background: white; border-radius: 12px; box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2); padding: 30px; z-index: 1000; max-width: 600px; width: 90%; max-height: 80vh; overflow-y: auto; font-family: 'Poppins', sans-serif; }
            .help-modal h3 { color: #11214D; font-weight: 600; margin-bottom: 20px; font-size: 20px; text-align: center; }
            .help-section { margin-bottom: 20px; }
            .help-section h4 { color: #11214D; font-weight: 600; margin-bottom: 10px; font-size: 16px; }
            .help-section p { color: #555; line-height: 1.6; margin-bottom: 10px; font-size: 14px; }
            .help-note { background-color: #fff3cd; border: 1px solid #ffc107; border-radius: 8px; padding: 15px; margin-bottom: 20px; color: #856404; font-weight: 500; }
            .color-legend { display: flex; flex-direction: column; gap: 8px; }
            .color-item { display: flex; align-items: center; gap: 10px; }
            .color-box { width: 20px; height: 20px; border-radius: 4px; border: 1px solid #ccc; }
            .color-box.manual { background-color: #72b7f4; }
            .color-box.normal { background-color: white; }
            .color-box.break { background-color: #ff5722; }
            .color-box.room-conflict { background-color: #ffebee; border-color: #f44336; }
            .color-box.lecturer-conflict { background-color: #ffd982; border-color: #d4942f; }
            .color-box.both-conflict { background-color: #ffdcec; border-color: #d42fa2; }
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

    # Dropdown options from timetables
    options = [
        {'label': (t['student_group']['name'] if isinstance(t['student_group'], dict) else str(t['student_group'])), 'value': idx}
        for idx, t in enumerate(timetables)
    ]

    # Layout: match OG exactly - title and dropdown on same line
    app.layout = html.Div([
        # Title and dropdown on the same line
        html.Div([
            html.H1("Interactive Drag & Drop Timetable - DE Optimization Results", 
                    style={"color": "#11214D", "fontWeight": "600", "fontSize": "24px", 
                          "fontFamily": "Poppins, sans-serif", "margin": "0", "flex": "1"}),
            
            html.Div([
                dcc.Dropdown(id='student-group-dropdown', options=options, value=0, searchable=True, clearable=False, style={"width": "280px", "fontSize": "13px", "fontFamily": "Poppins, sans-serif"})
            ], style={"display": "flex", "alignItems": "center"})
        ], style={"display": "flex", "alignItems": "center", "justifyContent": "space-between", 
                 "marginTop": "30px", "marginBottom": "30px", "maxWidth": "1200px", 
                 "margin": "30px auto", "padding": "0 15px"}),

        dcc.Store(id='all-timetables-store', data=timetables),
        dcc.Store(id='rooms-data-store', data=rooms_data),
        dcc.Store(id='original-timetables-store', data=[json.loads(json.dumps(t)) for t in timetables]),
        dcc.Store(id='constraint-details-store', data=initial_constraints),
        # Added missing source store to satisfy clientside logging callback inputs
        dcc.Store(id='constraint-source-store', data='initial'),
        dcc.Store(id='swap-data', data=None),
        dcc.Store(id='room-change-data', data=None),
        dcc.Store(id='missing-class-data', data=None),
        dcc.Store(id='selected-missing-class-store', data=None),
        dcc.Store(id='manual-cells-store', data=manual_cells or []),
        # Hidden store to drive room modal open
        dcc.Store(id='room-modal-open', data=None),
        # hidden sink for console logging
        html.Div(id='console-log-sink', style={'display': 'none'}),
        # hidden triggers for clientside -> serverside communication
        html.Button(id='swap-trigger', style={'display': 'none'}),
        html.Button(id='room-change-trigger', style={'display': 'none'}),
        html.Button(id='room-modal-open-trigger', style={'display': 'none'}),
        # missing-classes triggers/stores
        html.Button(id='missing-modal-open-trigger', style={'display': 'none'}),
        dcc.Store(id='add-class-data', data=None),

        # The timetable will be rendered here by callbacks
        html.Div(id='timetable-container'),
        # Hidden helpers used by callbacks/clientside init
        html.Div(id='trigger', style={'display': 'none'}),
        html.Div(id='dnd-init-sink', style={'display': 'none'}),
        html.Div(id='modal-handlers-sink', style={'display': 'none'}),

        # Room selection modal
        html.Div([
            html.Div(className='modal-overlay', id='modal-overlay', style={'display': 'none'}),
            html.Div([
                html.Div([html.H3('Select Classroom', className='modal-title'), html.Button('×', className='modal-close', id='modal-close-btn')], className='modal-header'),
                dcc.Input(id='room-search-input', type='text', placeholder='Search classrooms...', className='room-search'),
                html.Div(id='room-options-container', className='room-options'),
                html.Div([
                    html.Button('Cancel', id='room-cancel-btn', style={"backgroundColor": "#f5f5f5", "color": "#666", "padding": "8px 16px", "border": "1px solid #ddd", "borderRadius": "5px", "cursor": "pointer"}),
                    html.Button('DELETE SCHEDULE', id='room-delete-btn', style={"backgroundColor": "#dc3545", "color": "white", "padding": "8px 16px", "border": "none", "borderRadius": "5px", "cursor": "pointer", "marginRight": "10px", "fontWeight": "600", "display": "none"}),
                    html.Button('Confirm', id='room-confirm-btn', style={"backgroundColor": "#11214D", "color": "white", "padding": "8px 16px", "border": "none", "borderRadius": "5px", "cursor": "pointer"})
                ], style={"textAlign": "right", "marginTop": "20px", "paddingTop": "15px", "borderTop": "1px solid #f0f0f0"})
            ], className='room-selection-modal', id='room-selection-modal', style={'display': 'none'})
        ], style={"position": "relative"}),

        # Missing classes modal
        html.Div([
            html.Div(className='modal-overlay', id='missing-modal-overlay', style={'display': 'none'}),
            html.Div([
                html.Div([html.H3('Missing Classes', className='modal-title'), html.Button('×', className='modal-close', id='missing-modal-close-btn')], className='modal-header'),
                html.Div(id='missing-classes-container', className='room-options'),
                html.Div([
                    html.Button('Cancel', id='missing-cancel-btn', style={"backgroundColor": "#f5f5f5", "color": "#666", "padding": "8px 16px", "border": "1px solid #ddd", "borderRadius": "5px", "cursor": "pointer", "marginRight": "8px"}),
                    html.Button('Add to Slot', id='missing-add-btn', style={"backgroundColor": "#11214D", "color": "white", "padding": "8px 16px", "border": "none", "borderRadius": "5px", "cursor": "pointer"})
                ], style={"textAlign": "right", "marginTop": "20px", "paddingTop": "15px", "borderTop": "1px solid #f0f0f0"})
            ], className='room-selection-modal', id='missing-classes-modal', style={'display': 'none'})
        ], style={"position": "relative"}),

        # Errors modal
        html.Div([
            html.Div(className='modal-overlay', id='errors-modal-overlay', style={'display': 'none'}),
            html.Div([
                html.Div([html.H3('Constraint Violations', className='modal-title'), html.Button('×', className='modal-close', id='errors-modal-close-btn')], className='modal-header'),
                html.Div(id='errors-content'),
                html.Div([html.Button('Close', id='errors-close-btn', style={"backgroundColor": "#11214D", "color": "white", "padding": "8px 16px", "border": "none", "borderRadius": "5px", "cursor": "pointer"})], style={"textAlign": "right", "marginTop": "20px", "paddingTop": "15px", "borderTop": "1px solid #f0f0f0"})
            ], className='room-selection-modal', id='errors-modal', style={'display': 'none'})
        ]),

        # Download modal
        html.Div([
            html.Div(className='modal-overlay', id='download-modal-overlay', style={'display': 'none'}),
            html.Div([
                html.Div([html.H3('Download Timetables', className='modal-title'), html.Button('×', className='modal-close', id='download-modal-close-btn')], className='modal-header'),
                html.Div([
                    html.Div([html.Span('Download SST Timetables', style={"flex": "1", "fontSize": "16px", "fontWeight": "500"}), html.Button('Download', id='download-sst-btn', style={"backgroundColor": "#11214D", "color": "white", "padding": "8px 16px", "border": "none", "borderRadius": "5px", "cursor": "pointer", "fontSize": "14px"})], style={"display": "flex", "alignItems": "center", "justifyContent": "space-between", "padding": "15px", "borderBottom": "1px solid #f0f0f0"}),
                    html.Div([html.Span('Download TYD Timetables', style={"flex": "1", "fontSize": "16px", "fontWeight": "500"}), html.Button('Download', id='download-tyd-btn', style={"backgroundColor": "#11214D", "color": "white", "padding": "8px 16px", "border": "none", "borderRadius": "5px", "cursor": "pointer", "fontSize": "14px"})], style={"display": "flex", "alignItems": "center", "justifyContent": "space-between", "padding": "15px", "borderBottom": "1px solid #f0f0f0"}),
                    html.Div([html.Span('Download all Lecturer Timetables', style={"flex": "1", "fontSize": "16px", "fontWeight": "500"}), html.Button('Download', id='download-lecturer-btn', style={"backgroundColor": "#11214D", "color": "white", "padding": "8px 16px", "border": "none", "borderRadius": "5px", "cursor": "pointer", "fontSize": "14px"})], style={"display": "flex", "alignItems": "center", "justifyContent": "space-between", "padding": "15px"})
                ]),
                html.Div([html.Button('Close', id='download-close-btn', style={"backgroundColor": "#6c757d", "color": "white", "padding": "8px 16px", "border": "none", "borderRadius": "5px", "cursor": "pointer"})], style={"textAlign": "right", "marginTop": "20px", "paddingTop": "15px", "borderTop": "1px solid #f0f0f0"})
            ], className='room-selection-modal', id='download-modal', style={'display': 'none'})
        ]),

        # Conflict warning and save error popups
        html.Div([html.Div([html.Span('⚠️ Classroom Conflict', className='conflict-warning-title'), html.Button('×', className='conflict-warning-close', id='conflict-close-btn')], className='conflict-warning-header'), html.Div(id='conflict-warning-text', className='conflict-warning-content')], className='conflict-warning', id='conflict-warning', style={'display': 'none'}),
        html.Div([html.Div('❌ Cannot Save Timetable', className='save-error-title'), html.Div('Please resolve all classroom conflicts before saving the timetable.', className='save-error-content')], className='save-error', id='save-error', style={'display': 'none'}),

        # Bottom controls: download button and status
        html.Div([
            html.Button('Download Timetables', id='download-button', style={"backgroundColor": "#11214D", "color": "white", "padding": "10px 20px", "border": "none", "borderRadius": "5px", "fontSize": "14px", "cursor": "pointer", "fontWeight": "600", "boxShadow": "0 1px 3px rgba(0,0,0,0.1)", "transition": "all 0.2s ease"}),
            html.Div(id='download-status', style={"marginTop": "12px", "fontWeight": "600", "fontSize": "12px"})
        ], style={"textAlign": "center", "marginTop": "30px", "maxWidth": "1200px", "margin": "30px auto 0 auto"}),

        # Add Download component for client-side downloads
        dcc.Download(id='download-data'),

        html.Div(id='feedback', style={"marginTop": "20px", "textAlign": "center", "fontSize": "16px", "fontWeight": "bold", "minHeight": "30px", "maxWidth": "1200px", "margin": "20px auto 0 auto"}),

        # Help modal
        html.Div([
            html.Div(className='modal-overlay', id='help-modal-overlay', style={'display': 'none'}),
            html.Div([
                html.Div([
                    html.H3('Timetable Help Guide', className='modal-title', style={"color": "#11214D", "fontWeight": "600", "marginBottom": "0", "fontSize": "20px"}),
                    html.Button('×', className='modal-close', id='help-modal-close-btn')
                ], className='modal-header'),
                html.Div([
                    html.Div([
                        html.Strong('NOTE: '),
                        'Ensure all Lecturer names, emails and other details are inputted correctly to prevent errors'
                    ], className='help-note'),
                    html.Div([
                        html.H4('How to Use the Timetable:', style={"color": "#11214D", "fontWeight": "600", "marginBottom": "10px", "fontSize": "16px"}),
                        html.P('• Click and drag any class cell to swap it with another cell'),
                        html.P('• Double-click any cell to view and change the classroom for that class'),
                        html.P('• Use the navigation arrows (‹ ›) to switch between different student groups'),
                        html.P('• Click \'View Errors\' to see constraint violations and conflicts')
                    ], className='help-section'),
                    html.Div([
                        html.H4('Cell Color Meanings:', style={"color": "#11214D", "fontWeight": "600", "marginBottom": "10px", "fontSize": "16px"}),
                        html.Div([
                            html.Div([
                                html.Div(className='color-box normal'),
                                html.Span('Normal class - No conflicts')
                            ], className='color-item'),
                            html.Div([
                                html.Div(className='color-box manual'),
                                html.Span('Manually Scheduled - Class scheduled manually from the \'Missing Classes\' list.')
                            ], className='color-item'),
                            html.Div([
                                html.Div(className='color-box break'),
                                html.Span('Break time - Classes cannot be scheduled')
                            ], className='color-item'),
                            html.Div([
                                html.Div(className='color-box.room-conflict'),
                                html.Span('Room conflict - Same classroom used by multiple groups')
                            ], className='color-item'),
                            html.Div([
                                html.Div(className='color-box lecturer-conflict'),
                                html.Span('Lecturer conflict - Same lecturer teaching multiple groups')
                            ], className='color-item'),
                            html.Div([
                                html.Div(className='color-box both-conflict'),
                                html.Span('Multiple conflicts - Both room and lecturer issues')
                            ], className='color-item')
                        ], className='color-legend')
                    ], className='help-section')
                ]),
                
                html.Div([
                    html.Button('Close', id='help-close-btn', style={
                        "backgroundColor": "#11214D", "color": "white", "padding": "10px 20px",
                        "border": "none", "borderRadius": "8px", "cursor": "pointer",
                        "fontFamily": "Poppins, sans-serif", "fontSize": "14px", "fontWeight": "600"
                    })
                ], style={"textAlign": "center", "marginTop": "25px", "paddingTop": "20px", 
                         "borderTop": "2px solid #f0f0f0"})
            ], className='help-modal', id='help-modal', style={'display': 'none'})
        ])
    ])

    # Add Location component to detect page loads/refreshes
    app.layout.children.insert(0, dcc.Location(id='url', refresh=False))

    # Callback to reload fresh timetable data when page loads
    @app.callback(
        [Output('all-timetables-store', 'data', allow_duplicate=True),
         Output('manual-cells-store', 'data', allow_duplicate=True),
         Output('student-group-dropdown', 'options', allow_duplicate=True),
         Output('student-group-dropdown', 'value', allow_duplicate=True)],
        [Input('url', 'pathname')],
        prevent_initial_call='initial_duplicate'
    )
    def reload_data_on_page_load(pathname):
        """Reload fresh timetable data from JSON files when page is accessed"""
        print(f"[Dash UI] Page loaded, reloading timetable data from files...")
        fresh_timetables, fresh_manual_cells = _load_saved_timetable()
        
        if fresh_timetables:
            print(f"[Dash UI] Loaded {len(fresh_timetables)} timetables from JSON files")
            # Rebuild dropdown options
            new_options = [
                {'label': (t['student_group']['name'] if isinstance(t['student_group'], dict) else str(t['student_group'])), 'value': idx}
                for idx, t in enumerate(fresh_timetables)
            ]
            return fresh_timetables, fresh_manual_cells, new_options, 0
        else:
            print(f"[Dash UI] No fresh data found, keeping existing data")
            # Keep existing data
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    # Create timetable for selected group (matches OG: returns children and trigger)
    @app.callback(
        [Output('timetable-container', 'children'), Output('trigger', 'children')],
        [Input('student-group-dropdown', 'value')],
        [State('all-timetables-store', 'data'), State('manual-cells-store', 'data'), State('constraint-details-store', 'data')]
    )
    def create_timetable(selected_group_idx, all_timetables_data, manual_cells_state, constraint_details):
        if selected_group_idx is None or not all_timetables_data:
            return html.Div('No data available'), 'trigger'
        if selected_group_idx >= len(all_timetables_data):
            selected_group_idx = 0
        tdata = all_timetables_data[selected_group_idx]
        name = tdata['student_group']['name'] if isinstance(tdata['student_group'], dict) else getattr(tdata['student_group'], 'name', str(tdata['student_group']))
        rows = tdata['timetable']
        # Build overlay conflicts from constraints (day/time -> row/col)
        overlay = {}
        try:
            day_map = {'Mon':0,'Tue':1,'Wed':2,'Thu':3,'Fri':4,'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4}
            def add_overlay(day_idx, time_label, typ):
                try:
                    # Expect time_label like '9:00' or '09:00'
                    hour = None
                    if isinstance(time_label, str) and ':' in time_label:
                        hour = int(time_label.split(':')[0])
                    elif isinstance(time_label, (int, float)):
                        hour = int(time_label)
                    if hour is None:
                        return
                    r = hour - 9
                    if 0 <= r < len(rows) and 0 <= int(day_idx) <= 4:
                        key = f"{r}_{int(day_idx)}"
                        if key not in overlay:
                            overlay[key] = typ
                        else:
                            if overlay[key] != typ:
                                overlay[key] = 'both'
                except Exception:
                    return
            if isinstance(constraint_details, dict):
                for v in constraint_details.get('Different Student Group Overlaps', []) or []:
                    if isinstance(v, dict):
                        d = v.get('day', None)
                        t = v.get('time', None)
                        # Fallback: parse from 'location' like "Mon at 9:00"
                        if (d is None or t is None) and isinstance(v.get('location'), str):
                            try:
                                loc = v.get('location')
                                # extract day word and hour
                                varr = loc.replace(' at ', ' ').replace(',', ' ').split()
                                # find first day-like token and first token with ':''
                                dd = None; tt = None
                                for token in varr:
                                    if token[:3] in ['Mon','Tue','Wed','Thu','Fri','Sat','Sun','Mon.', 'Tue.', 'Wed.', 'Thu.', 'Fri.']:
                                        dd = token[:3]
                                    if ':' in token:
                                        tt = token
                                if dd and not d:
                                    d = dd
                                if tt and not t:
                                    t = tt
                            except Exception:
                                pass
                        if isinstance(d, str):
                            d = day_map.get(d, None)
                        add_overlay(d, t, 'room')
                for v in constraint_details.get('Lecturer Clashes', []) or []:
                    if isinstance(v, dict):
                        d = v.get('day', None)
                        t = v.get('time', None)
                        if (d is None or t is None) and isinstance(v.get('location'), str):
                            try:
                                loc = v.get('location')
                                varr = loc.replace(' at ', ' ').replace(',', ' ').split()
                                dd = None; tt = None
                                for token in varr:
                                    if token[:3] in ['Mon','Tue','Wed','Thu','Fri','Sat','Sun','Mon.', 'Tue.', 'Wed.', 'Thu.', 'Fri.']:
                                        dd = token[:3]
                                    if ':' in token:
                                        tt = token
                                if dd and not d:
                                    d = dd
                                if tt and not t:
                                    t = tt
                            except Exception:
                                pass
                        if isinstance(d, str):
                            d = day_map.get(d, None)
                        add_overlay(d, t, 'lecturer')
        except Exception:
            overlay = {}
        # Runtime detection across groups
        conflicts = _detect_conflicts(all_timetables_data, selected_group_idx)
        header_cells = [html.Th('Time', style={"backgroundColor": "#11214D", "color": "white", "padding": "12px 10px", "fontWeight": "600", "fontSize": "13px", "textAlign": "center", "border": "1px solid #0d1a3d"})]
        for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
            header_cells.append(html.Th(day, style={"backgroundColor": "#11214D", "color": "white", "padding": "12px 10px", "fontWeight": "600", "fontSize": "13px", "textAlign": "center", "border": "1px solid #0d1a3d"}))
        body_rows = []
        for r in range(len(rows)):
            cells = [html.Td(rows[r][0], className='time-cell')]
            for c in range(1, len(rows[r])):
                raw = rows[r][c] if rows[r][c] else 'FREE'
                text = str(raw).strip()
                text_upper = text.upper()
                is_break = text_upper == 'BREAK'
                key = f"{r}_{c-1}"
                ctype = conflicts.get(key, {}).get('type', 'none') if key in conflicts else 'none'
                # Do not overlay conflicts on FREE cells
                if text_upper != 'FREE':
                    # overlay from constraints file only if not FREE
                    if key in overlay:
                        if ctype == 'none':
                            ctype = overlay[key]
                        elif ctype != overlay[key]:
                            ctype = 'both'
                else:
                    ctype = 'none'
                manual_key = f"{selected_group_idx}_{r}_{c-1}"
                is_manual = bool(manual_cells_state) and manual_key in (manual_cells_state or [])
                if is_break:
                    cls = 'cell break-time'; draggable = 'false'
                elif is_manual:
                    # Manual cells get blue outline regardless of conflicts
                    if ctype == 'room':
                        cls = 'cell room-conflict manual-schedule'
                    elif ctype == 'lecturer':
                        cls = 'cell lecturer-conflict manual-schedule'
                    elif ctype == 'both':
                        cls = 'cell both-conflict manual-schedule'
                    else:
                        cls = 'cell manual-schedule'
                    draggable = 'true'
                elif ctype == 'room':
                    cls = 'cell room-conflict'
                    draggable = 'true'
                elif ctype == 'lecturer':
                    cls = 'cell lecturer-conflict'
                    draggable = 'true'
                elif ctype == 'both':
                    cls = 'cell both-conflict'
                    draggable = 'true'
                else:
                    cls = 'cell'
                    draggable = 'true'
                cell_id = {"type": "cell", "group": selected_group_idx, "row": r, "col": c-1}
                cells.append(html.Td(html.Div(text, id=cell_id, className=cls, draggable=draggable, n_clicks=0), style={"padding": "0", "border": "1px solid #e0e0e0"}))
            body_rows.append(html.Tr(cells))
        table = html.Table([html.Thead(html.Tr(header_cells)), html.Tbody(body_rows)], style={"width": "100%", "borderCollapse": "separate", "borderSpacing": "0", "backgroundColor": "white", "borderRadius": "6px", "overflow": "hidden", "fontSize": "12px", "boxShadow": "0 2px 6px rgba(0,0,0,0.08)", "fontFamily": "Poppins, sans-serif"})
        # Header with title and navigation arrows
        header = html.Div([
            html.Div([
                html.H2(f"Timetable for {name}", className='timetable-title', style={"color": "#11214D", "fontWeight": "600", "fontSize": "20px", "fontFamily": "Poppins, sans-serif", "margin": "0"})
            ], className='timetable-title-container'),
            html.Div([
                html.Button('‹', className='nav-arrow', id='prev-group-btn', disabled=selected_group_idx == 0),
                html.Button('›', className='nav-arrow', id='next-group-btn', disabled=selected_group_idx == len(all_timetables_data) - 1)
            ], className='nav-arrows')
        ], className='timetable-header')
        # Buttons row with Help button aligned to the right
        btns = html.Div([
            html.Button(['View Errors', html.Div(id='error-notification-badge', className='error-notification')], id='errors-btn', className='errors-button'),
            html.Button('Undo All Changes', id='undo-all-btn', style={"backgroundColor": "#6c757d", "color": "white", "padding": "8px 16px", "border": "none", "borderRadius": "5px", "fontSize": "14px", "cursor": "pointer", "fontWeight": "500", "fontFamily": "Poppins, sans-serif"}),
            html.Button('?', id='help-icon-btn', title='Help', className='nav-arrow', style={"marginLeft": "auto"})
        ], style={"marginBottom": "15px", "marginRight": "10px", "textAlign": "left", "display": "flex", "gap": "10px", "alignItems": "flex-start"})
        return html.Div([header, btns, table], className='student-group-container'), 'trigger'

    # Update timetable content after swaps
    @app.callback(
        Output('timetable-container', 'children', allow_duplicate=True),
        Input('all-timetables-store', 'data'),
        [State('student-group-dropdown', 'value'), State('manual-cells-store', 'data'), State('constraint-details-store', 'data')],
        prevent_initial_call=True
    )
    def update_timetable_content(all_timetables_data, selected_group_idx, manual_cells_state, constraint_details):
        if selected_group_idx is None or not all_timetables_data or selected_group_idx >= len(all_timetables_data):
            raise dash.exceptions.PreventUpdate
        # reuse create_timetable body to render container only
        children, _ = create_timetable(selected_group_idx, all_timetables_data, manual_cells_state, constraint_details)
        return children

    # Navigation arrows
    @app.callback(
        Output('student-group-dropdown', 'value'),
        [Input('prev-group-btn', 'n_clicks'), Input('next-group-btn', 'n_clicks')],
        [State('student-group-dropdown', 'value'), State('all-timetables-store', 'data')],
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
        if button_id == 'prev-group-btn' and current_value > 0:
            return current_value - 1
        if button_id == 'next-group-btn' and current_value < max_index:
            return current_value + 1
        raise dash.exceptions.PreventUpdate

    # Handle swaps: update data and manual cell states, auto-save
    @app.callback(
        [Output('all-timetables-store', 'data', allow_duplicate=True), Output('manual-cells-store', 'data', allow_duplicate=True)],
        Input('swap-data', 'data'),
        [State('all-timetables-store', 'data'), State('manual-cells-store', 'data')],
        prevent_initial_call=True
    )
    def handle_swap(swap_data, current_timetables, manual_cells_state):
        print(f"[SWAP] Callback triggered with swap_data: {swap_data}")
        if not swap_data or not current_timetables:
            print("[SWAP] No swap data or timetables - preventing update")
            raise dash.exceptions.PreventUpdate
        try:
            source = swap_data['source']; target = swap_data['target']
            print(f"[SWAP] Source: group={source['group']}, row={source['row']}, col={source['col']}")
            print(f"[SWAP] Target: group={target['group']}, row={target['row']}, col={target['col']}")
            
            if source['group'] != target['group']:
                print("[SWAP] Different groups - preventing update")
                raise dash.exceptions.PreventUpdate
            if str(source.get('content','')).upper() == 'BREAK' or str(target.get('content','')).upper() == 'BREAK':
                print("[SWAP] BREAK cell detected - preventing update")
                raise dash.exceptions.PreventUpdate
            g = int(source['group'])
            updated = json.loads(json.dumps(current_timetables))
            rows = updated[g]['timetable']
            if (source['row'] >= len(rows) or target['row'] >= len(rows) or source['col'] + 1 >= len(rows[0]) or target['col'] + 1 >= len(rows[0])):
                print("[SWAP] Out of bounds - preventing update")
                raise dash.exceptions.PreventUpdate
            
            s_content = rows[source['row']][source['col'] + 1]
            t_content = rows[target['row']][target['col'] + 1]
            print(f"[SWAP] Swapping '{s_content[:50] if s_content else 'FREE'}...' <-> '{t_content[:50] if t_content else 'FREE'}...'")
            
            rows[source['row']][source['col'] + 1] = t_content
            rows[target['row']][target['col'] + 1] = s_content
            session_state['has_swaps'] = True
            source_is_manual = swap_data.get('sourceIsManual', False) or swap_data.get('source', {}).get('sourceIsManual', False)
            target_is_manual = swap_data.get('targetIsManual', False) or swap_data.get('target', {}).get('targetIsManual', False)
            updated_manual = manual_cells_state.copy() if manual_cells_state else []
            s_key = f"{g}_{source['row']}_{source['col']}"; t_key = f"{g}_{target['row']}_{target['col']}"
            if source_is_manual and s_key in updated_manual: updated_manual.remove(s_key)
            if target_is_manual and t_key in updated_manual: updated_manual.remove(t_key)
            if source_is_manual and t_key not in updated_manual: updated_manual.append(t_key)
            if target_is_manual and s_key not in updated_manual: updated_manual.append(s_key)
            save_timetable_to_file(updated, updated_manual)
            print(f"[SWAP] Successfully swapped cells and saved to file")
            return updated, updated_manual
        except Exception as e:
            print(f"[SWAP] ERROR: {e}")
            traceback.print_exc()
            raise dash.exceptions.PreventUpdate

    # Update constraint violations after user changes
    @app.callback(
        Output('constraint-details-store', 'data', allow_duplicate=True),
        [Input('all-timetables-store', 'data')],
        [State('rooms-data-store', 'data'), State('constraint-details-store', 'data')],
        prevent_initial_call=True
    )
    def update_constraints_rt(timetables_data, rooms, current_constraints):
        if not timetables_data:
            raise dash.exceptions.PreventUpdate
        if not session_state.get('has_swaps'):
            return dash.no_update
        
        # Recompute dynamic constraints (room conflicts, lecturer clashes, etc.)
        updated = recompute_constraint_violations_simplified(timetables_data, rooms)
        
        if updated and current_constraints:
            # CRITICAL: Preserve constraints from original DE algorithm that don't change with swaps
            # These are calculated based on course allocations, not cell positions
            constraints_to_preserve = ['Missing or Extra Classes', 'Classes During Break Time']
            for constraint_type in constraints_to_preserve:
                if constraint_type in current_constraints:
                    updated[constraint_type] = current_constraints[constraint_type]
        
        try:
            print("[Dash_UI] Updated constraint details (preserving original algorithm constraints):")
            if updated:
                for constraint_type in ['Missing or Extra Classes', 'Classes During Break Time']:
                    if constraint_type in updated:
                        print(f"  - {constraint_type}: {len(updated[constraint_type])} violations preserved")
        except Exception:
            pass
        return updated or current_constraints

    # Re-render timetable when constraints change so overlays/colors refresh
    @app.callback(
        Output('timetable-container', 'children', allow_duplicate=True),
        Input('constraint-details-store', 'data'),
        [State('student-group-dropdown', 'value'), State('all-timetables-store', 'data'), State('manual-cells-store', 'data')],
        prevent_initial_call=True
    )
    def refresh_timetable_on_constraints(constraint_details, selected_group_idx, all_timetables_data, manual_cells_state):
        if selected_group_idx is None or not all_timetables_data or selected_group_idx >= len(all_timetables_data):
            raise dash.exceptions.PreventUpdate
        children, _ = create_timetable(selected_group_idx, all_timetables_data, manual_cells_state, constraint_details)
        return children

    # Errors modal open/close and content
    @app.callback(
        [Output('errors-modal-overlay', 'style'), Output('errors-modal', 'style'), Output('errors-content', 'children')],
        [Input('errors-btn', 'n_clicks'), Input('errors-modal-close-btn', 'n_clicks'), Input('errors-close-btn', 'n_clicks'), Input('errors-modal-overlay', 'n_clicks')],
        State('constraint-details-store', 'data'),
        prevent_initial_call=True
   
    )
    def handle_errors_modal(errors_btn_clicks, close1, close2, overlay, constraint_details):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        # Reset expanded states when modal is opened
        if trigger_id == 'errors-btn' and errors_btn_clicks:
            if hasattr(create_errors_modal_content, 'expanded_states'):
                create_errors_modal_content.expanded_states.clear()
            content = create_errors_modal_content(constraint_details)
            return {"display": "block"}, {"display": "block"}, content
        
        if trigger_id in ['errors-modal-close-btn', 'errors-close-btn', 'errors-modal-overlay']:
            return {"display": "none"}, {"display": "none"}, dash.no_update
        
        raise dash.exceptions.PreventUpdate

    # Handle expanding/collapsing constraint details in the errors modal
    @app.callback(
        Output('errors-content', 'children', allow_duplicate=True),
        Input({'type': 'constraint-header', 'index': dash.dependencies.ALL}, 'n_clicks'),
        [State('constraint-details-store', 'data'),
         State({'type': 'constraint-header', 'index': dash.dependencies.ALL}, 'id')],
        prevent_initial_call=True
    )
    def toggle_constraint_details(n_clicks, constraint_details, ids):
        ctx = dash.callback_context
        if not ctx.triggered or not any(n_clicks):
            raise dash.exceptions.PreventUpdate
        
        triggering_id = ctx.triggered[0]['prop_id']
        # The ID is a JSON string like '{"index":"Lecturer Clashes","type":"constraint-header"}'
        import json
        try:
            index_to_toggle = json.loads(triggering_id.split('.')[0])['index']
        except (json.JSONDecodeError, KeyError):
            raise dash.exceptions.PreventUpdate

        # The create_errors_modal_content function uses a stateful set to track expanded items
        content = create_errors_modal_content(constraint_details, toggle_constraint=index_to_toggle)
        return content

    # Handle help button to show/hide help modal and overlay (matches Sept 13)
    @app.callback(
        [Output('help-modal-overlay', 'style'),
         Output('help-modal', 'style')],
        [Input('help-icon-btn', 'n_clicks'), 
         Input('help-modal-close-btn', 'n_clicks'),
         Input('help-close-btn', 'n_clicks'),
         Input('help-modal-overlay', 'n_clicks')],
        prevent_initial_call=True
    )
    def handle_help_modal(help_btn_clicks, close_btn_clicks, close_btn2_clicks, overlay_clicks):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if trigger_id == 'help-icon-btn' and help_btn_clicks:
            # Show modal and overlay
            return {'display': 'block'}, {'display': 'block'}
        elif trigger_id in ['help-modal-close-btn', 'help-close-btn', 'help-modal-overlay']:
            # Hide modal and overlay
            return {'display': 'none'}, {'display': 'none'}
        
        raise dash.exceptions.PreventUpdate

    # Log full constraints to browser console whenever they change
    clientside_callback(
        """
        function(data, source){
            try {
                if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
                    console.debug('[Dash] Constraint source:', source);
                    console.debug('[Dash] Constraint violations updated:', data);
                }
            } catch(e){}
            return '';
        }
        """,
        Output('console-log-sink', 'children'),
        [Input('constraint-details-store', 'data'), Input('constraint-source-store', 'data')],
        prevent_initial_call=False
    )

    # Initialize drag/drop and dblclick to bridge to hidden triggers; also FREE dblclick -> missing modal
    clientside_callback(
        """
        function(children){
            if(!children){ return ''; }
            setTimeout(function(){
                window.dash_clientside = window.dash_clientside || {};
                const cells = document.querySelectorAll('.cell');
                const getCtx = (el)=>{ try { return JSON.parse(el.id); } catch(e){ return null; } };
                
                cells.forEach(function(cell){
                    // Clear existing listeners first (Sept 13 approach)
                    cell.ondragstart = null;
                    cell.ondragover = null;
                    cell.ondragenter = null;
                    cell.ondragleave = null;
                    cell.ondrop = null;
                    cell.ondragend = null;
                    cell.ondblclick = null;
                    
                    // Remove any existing drag-related classes
                    cell.classList.remove('dragging', 'drag-over');
                    
                    const txt = (cell.innerText||'').trim();
                    const isBreak = txt.toUpperCase() === 'BREAK';
                    cell.setAttribute('draggable', isBreak ? 'false' : 'true');
                    cell.ondragstart = function(e){
                        if (isBreak) { e.preventDefault(); return; }
                        const ctx = getCtx(cell); if(!ctx){ return; }
                        cell.classList.add('dragging');
                        window.dash_clientside._drag_src = { group: ctx.group, row: ctx.row, col: ctx.col, content: txt, sourceIsManual: cell.classList.contains('manual-schedule') };
                        try { e.dataTransfer.setData('text/plain', 'drag'); } catch(err){}
                    };
                    cell.ondragend = function(){ cell.classList.remove('dragging'); };
                    cell.ondragover = function(e){
                        if (isBreak) return;
                        try{ e.preventDefault(); }catch(err){}
                        cell.classList.add('drag-over');
                    };
                    cell.ondragleave = function(){ cell.classList.remove('drag-over'); };
                    cell.ondrop = function(e){
                        try{ e.preventDefault(); e.stopPropagation(); }catch(err){}
                        cell.classList.remove('drag-over');
                        
                        // Prevent dropping on break times
                        if (isBreak || txt.toUpperCase() === 'BREAK') {
                            console.log('Cannot drop on break time');
                            return false;
                        }
                        
                        const tgt = getCtx(cell); 
                        const src = window.dash_clientside._drag_src;
                        
                        if(!tgt || !src){ 
                            console.log('No valid drag operation');
                            return; 
                        }
                        if(src.group !== tgt.group){ 
                            console.log('Cannot swap between different groups');
                            return; 
                        }
                        
                        const ttxt = (cell.innerText||'').trim();
                        if((src.content && src.content.toUpperCase() === 'BREAK') || ttxt.toUpperCase() === 'BREAK'){ 
                            console.log('Cannot swap with BREAK cell');
                            return; 
                        }
                        
                        const targetIsManual = cell.classList.contains('manual-schedule');
                        const payload = { 
                            source: src, 
                            target: { 
                                group: tgt.group, 
                                row: tgt.row, 
                                col: tgt.col, 
                                content: ttxt, 
                                targetIsManual: targetIsManual 
                            },
                            timestamp: Date.now()
                        };
                        
                        console.log('Swapping cells:', payload);
                        
                        // Use Sept 13 approach: window.dash_clientside.set_props
                        try {
                            window.dash_clientside.set_props('swap-data', { data: payload });
                            console.log('✅ Swap data sent to callback');
                        } catch(err) {
                            console.error('Failed to send swap data:', err);
                        }
                    };
                    // Double-click handler for room selection and FREE cell scheduling
                    cell.ondblclick = function(e) {
                        e.preventDefault();
                        e.stopPropagation();
                        
                        console.log('Double-click detected on cell');
                        
                        const ctx = getCtx(cell); 
                        if(!ctx){ return; }
                        
                        const currentText = (cell.innerText||'').trim().toUpperCase();
                        
                        // Handle FREE cell double-click for manual scheduling
                        if(currentText === 'FREE'){
                            console.log('FREE cell double-clicked - showing missing classes');
                            window.selectedCell = cell;
                            
                            // Show missing classes modal
                            const modal = document.getElementById('missing-classes-modal');
                            const overlay = document.getElementById('missing-modal-overlay');
                            
                            if (modal && overlay) {
                                modal.style.display = 'block';
                                overlay.style.display = 'block';
                                
                                // Trigger missing classes loading
                                try {
                                    window.dash_clientside.set_props('missing-class-data', {
                                        data: {
                                            action: 'show_missing',
                                            cell_id: cell.id,
                                            timestamp: Date.now()
                                        }
                                    });
                                    console.log('✅ Missing class data sent to callback');
                                } catch(err) {
                                    console.error('Failed to send missing class data:', err);
                                }
                            }
                            return;
                        }
                        
                        // Don't allow room selection for break times
                        if (currentText === 'BREAK' || cell.classList.contains('break-time')) {
                            console.log('Cannot select room for break time');
                            return;
                        }
                        
                        // Store the selected cell
                        window.selectedCell = cell;
                        
                        // Show room selection modal
                        const modal = document.getElementById('room-selection-modal');
                        const overlay = document.getElementById('modal-overlay');
                        const deleteBtn = document.getElementById('room-delete-btn');
                        
                        if (modal && overlay) {
                            modal.style.display = 'block';
                            overlay.style.display = 'block';
                            
                            // Show delete button only for manually scheduled cells
                            if (deleteBtn) {
                                if (cell.classList.contains('manual-schedule')) {
                                    deleteBtn.style.display = 'inline-block';
                                    deleteBtn.textContent = 'DELETE SCHEDULE';
                                    console.log('Showing DELETE SCHEDULE button for manual cell');
                                } else {
                                    deleteBtn.style.display = 'none';
                                    console.log('Hiding DELETE SCHEDULE button for non-manual cell');
                                }
                            }
                            
                            // Trigger room options loading
                            try {
                                window.dash_clientside.set_props('room-change-data', {
                                    data: {
                                        action: 'show_modal',
                                        cell_id: cell.id,
                                        cell_content: cell.textContent.trim(),
                                        is_manual: cell.classList.contains('manual-schedule'),
                                        timestamp: Date.now()
                                    }
                                });
                                console.log('✅ Room change data sent to callback');
                            } catch(err) {
                                console.error('Failed to send room change data:', err);
                            }
                            
                            // Focus on search input
                            setTimeout(function() {
                                const searchInput = document.getElementById('room-search-input');
                                if (searchInput) {
                                    searchInput.focus();
                                }
                            }, 100);
                        }
                    };
                });
            }, 50); // A small delay can help ensure all elements are in the DOM
            return '';
        }
        """,
        Output('dnd-init-sink', 'children'),
        Input('timetable-container', 'children'),
        prevent_initial_call=False
    )

    # Setup modal close handlers - FROM SEPT 13
    clientside_callback(
        """
        function(children){
            if(!children){ return ''; }
            setTimeout(function(){
                // Function to close room selection modal
                function closeModal() {
                    const modal = document.getElementById('room-selection-modal');
                    const overlay = document.getElementById('modal-overlay');
                    if (modal && overlay) {
                        modal.style.display = 'none';
                        overlay.style.display = 'none';
                    }
                    window.selectedCell = null;
                }
                
                // Attach close handlers to room modal
                const modalCloseBtn = document.getElementById('modal-close-btn');
                const roomCancelBtn = document.getElementById('room-cancel-btn');
                const overlay = document.getElementById('modal-overlay');
                
                if (modalCloseBtn) {
                    modalCloseBtn.onclick = closeModal;
                }
                
                if (roomCancelBtn) {
                    roomCancelBtn.onclick = closeModal;
                }
                
                if (overlay) {
                    overlay.onclick = closeModal;
                }
                
                console.log('✅ Modal close handlers attached');
            }, 100);
            return '';
        }
        """,
        Output('modal-handlers-sink', 'children'),
        Input('timetable-container', 'children'),
        prevent_initial_call=False
    )

    # Bridge swap trigger to store (this becomes a fallback if direct store update fails)
    clientside_callback(
        """
        function(n, current_data){
            if(!n){ return window.dash_clientside.no_update; }
            const payload = (window.dash_clientside||{}).swap_payload;
            if (payload && payload.ts > ((current_data||{}).ts || 0)) {
                return payload;
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output('swap-data', 'data'),
        Input('swap-trigger', 'n_clicks'),
        State('swap-data', 'data'),
        prevent_initial_call=True
    )

    # Bridge room modal open trigger to store (fallback)
    clientside_callback(
        """
        function(n, current_data){
            if(!n){ return window.dash_clientside.no_update; }
            const payload = (window.dash_clientside||{}).room_modal_payload;
            if (payload && payload.ts > ((current_data||{}).ts || 0)) {
                return payload;
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output('room-modal-open', 'data'),
        Input('room-modal-open-trigger', 'n_clicks'),
        State('room-modal-open', 'data'),
        prevent_initial_call=True
    )

    # Callback to handle room selection modal and search
    @app.callback(
        [Output('room-options-container', 'children'),
         Output('room-selection-modal', 'style'),
         Output('modal-overlay', 'style'),
         Output('room-delete-btn', 'style')],
        [Input('room-change-data', 'data'),
         Input('room-search-input', 'value')],
        [State('all-timetables-store', 'data'),
         State('rooms-data-store', 'data'),
         State('student-group-dropdown', 'value'),
         State('manual-cells-store', 'data')],
        prevent_initial_call=True
    )
    def open_room_modal(room_change_data, search_value, timetables_state, rooms_data, selected_group_idx, manual_cells):
        if not room_change_data or room_change_data.get('action') != 'show_modal':
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        if not rooms_data or not timetables_state:
            return html.Div("No rooms data available"), {"display": "block"}, {"display": "block"}, {"display": "none"}
        
        # Parse cell information - FROM SEPT 13
        try:
            cell_id_str = room_change_data['cell_id']
            cell_id = json.loads(cell_id_str)
            row_idx = cell_id['row']
            col_idx = cell_id['col']
            group_idx = cell_id['group']
        except Exception as e:
            print(f"Error parsing cell ID: {e}")
            print(f"cell_id_str: {room_change_data.get('cell_id')}")
            return html.Div("Error parsing cell data"), {"display": "block"}, {"display": "block"}, {"display": "none"}
        
        # Get current room usage for this time slot across all groups
        current_room_usage = get_room_usage_at_timeslot(timetables_state, row_idx, col_idx)
        
        # Filter rooms based on search
        filtered_rooms = rooms_data
        if search_value:
            search_lower = search_value.lower()
            filtered_rooms = [room for room in rooms_data 
                             if search_lower in room['name'].lower() or 
                                search_lower in room.get('building', '').lower() or
                                search_lower in room.get('room_type', '').lower()]
        
        # Create room options
        room_options = []
        for room in filtered_rooms:
            room_name = room['name']
            is_available = room_name not in current_room_usage
            
            # Determine styling
            if is_available:
                option_class = "room-option available"
            else:
                option_class = "room-option occupied"
            
            # Create conflict info if room is occupied
            conflict_info = ""
            if not is_available:
                conflicting_groups = current_room_usage[room_name]
                conflict_info = f" (Used by: {', '.join(conflicting_groups)})"
            
            room_options.append(
                html.Div([
                    html.Div([
                        html.Span(room_name, style={"fontWeight": "600"}),
                        html.Span(conflict_info, style={"fontSize": "11px", "color": "#666", "marginLeft": "8px"})
                    ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap"}),
                    html.Div([
                        html.Span(f"Cap: {room['capacity']}", className="room-info"),
                        html.Span(f" | {room['building']}", className="room-info"),
                        html.Span(f" | {room.get('room_type', 'N/A')}", className="room-info")
                    ])
                ], 
                className=option_class,
                id={"type": "room-option", "room_id": room['Id'], "room_name": room_name},
                n_clicks=0)
            )
        
        # Determine if DELETE SCHEDULE button should be shown
        manual_cell_key = f"{group_idx}_{row_idx}_{col_idx}"
        is_manual_cell = manual_cells and manual_cell_key in manual_cells
        
        delete_btn_style = {
            "backgroundColor": "#dc3545", 
            "color": "white", 
            "padding": "8px 16px", 
            "border": "none", 
            "borderRadius": "5px", 
            "cursor": "pointer", 
            "marginRight": "10px",
            "fontFamily": "Poppins, sans-serif", 
            "fontWeight": "600", 
            "display": "inline-block" if is_manual_cell else "none"
        }
        
        return room_options, {"display": "block"}, {"display": "block"}, delete_btn_style

    # Callback to handle room option selection
    @app.callback(
        Output("room-change-data", "data", allow_duplicate=True),
        [Input({"type": "room-option", "room_id": ALL, "room_name": ALL}, "n_clicks")],
        [State({"type": "room-option", "room_id": ALL, "room_name": ALL}, "id"),
         State("room-change-data", "data")],
        prevent_initial_call=True
    )
    def handle_room_selection(n_clicks_list, room_ids, current_room_data):
        if not any(n_clicks_list) or not current_room_data:
            raise dash.exceptions.PreventUpdate
        
        # Find which room was clicked using callback context
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        try:
            triggered_room = json.loads(triggered_id)
            selected_room_name = triggered_room['room_name']
        except:
            raise dash.exceptions.PreventUpdate
        
        # Update room_change_data with selected room - use 'room_selected' action
        return {
            **current_room_data,
            'action': 'room_selected',
            'selected_room': selected_room_name,
            'timestamp': current_room_data.get('timestamp', 0) + 1
        }

    # Callback to handle room confirmation and update timetable
    @app.callback(
        [Output('all-timetables-store', 'data', allow_duplicate=True),
         Output('manual-cells-store', 'data', allow_duplicate=True),
         Output('room-selection-modal', 'style', allow_duplicate=True),
         Output('modal-overlay', 'style', allow_duplicate=True)],
        Input('room-confirm-btn', 'n_clicks'),
        [State('room-change-data', 'data'),
         State('all-timetables-store', 'data'),
         State('student-group-dropdown', 'value'),
         State('manual-cells-store', 'data')],
        prevent_initial_call=True
    )
    def confirm_room_change(confirm_clicks, room_change_data, all_timetables_data, selected_group_idx, manual_cells):
        if not confirm_clicks or not room_change_data or room_change_data.get('action') != 'room_selected':
            raise dash.exceptions.PreventUpdate
        
        session_state['has_swaps'] = True
        
        try:
            # Parse cell information
            cell_id_str = room_change_data['cell_id']
            cell_id = json.loads(cell_id_str)
            row_idx = cell_id['row']
            col_idx = cell_id['col']
            selected_room = room_change_data['selected_room']
            
            # Update the current group's timetable
            updated_timetables = json.loads(json.dumps(all_timetables_data))
            current_group_timetable = updated_timetables[selected_group_idx]['timetable']
            
            # Get the original cell content and extract course info
            original_content = current_group_timetable[row_idx][col_idx + 1]
            
            # Update room in the cell content
            updated_content = update_room_in_cell_content(original_content, selected_room)
            current_group_timetable[row_idx][col_idx + 1] = updated_content
            
            # Find and update consecutive classes of the same course
            course_code = extract_course_code_from_cell(original_content)
            if course_code:
                update_consecutive_course_rooms(updated_timetables, selected_group_idx, course_code, selected_room, row_idx, col_idx)
            
            # Save changes
            save_timetable_to_file(updated_timetables, manual_cells)
            
            # Hide the modal
            return updated_timetables, manual_cells, {"display": "none"}, {"display": "none"}
            
        except Exception as e:
            print(f"Error confirming room change: {e}")
            import traceback
            traceback.print_exc()
            raise dash.exceptions.PreventUpdate

    # Callback to handle DELETE SCHEDULE button
    @app.callback(
        [Output('all-timetables-store', 'data', allow_duplicate=True),
         Output('manual-cells-store', 'data', allow_duplicate=True),
         Output('room-selection-modal', 'style', allow_duplicate=True),
         Output('modal-overlay', 'style', allow_duplicate=True)],
        Input('room-delete-btn', 'n_clicks'),
        [State('room-change-data', 'data'),
         State('all-timetables-store', 'data'),
         State('manual-cells-store', 'data')],
        prevent_initial_call=True
    )
    def delete_manual_schedule(delete_clicks, room_data, timetables_state, manual_cells):
        if not delete_clicks or not room_data or not timetables_state:
            raise dash.exceptions.PreventUpdate
        
        try:
            # Parse cell ID - FROM SEPT 13
            cell_id_str = room_data.get('cell_id', '')
            cell_id = json.loads(cell_id_str)
            g = cell_id['group']
            r = cell_id['row']
            c = cell_id['col']
            
            # Update timetable to FREE
            updated = json.loads(json.dumps(timetables_state))
            updated[g]['timetable'][r][c + 1] = "FREE"
            
            # Remove from manual cells tracking
            updated_manual_cells = manual_cells.copy() if manual_cells else []
            manual_cell_key = f"{g}_{r}_{c}"
            if manual_cell_key in updated_manual_cells:
                updated_manual_cells.remove(manual_cell_key)
            
            # Save changes
            session_state['has_swaps'] = True
            save_timetable_to_file(updated, updated_manual_cells)
            
            # Hide the modal
            return updated, updated_manual_cells, {"display": "none"}, {"display": "none"}
            
        except Exception as e:
            print(f"Error deleting schedule: {e}")
            import traceback
            traceback.print_exc()
            raise dash.exceptions.PreventUpdate

    # Highlight selected room option in modal (clientside for performance)
    clientside_callback(
        """
        function(children){ 
            if(!children) return window.dash_clientside.no_update; 
            setTimeout(function(){ 
                const roomOptions = document.querySelectorAll('.room-option'); 
                roomOptions.forEach(function(option){ 
                    option.onclick=function(){ 
                        roomOptions.forEach(function(opt){ opt.classList.remove('selected'); }); 
                        this.classList.add('selected'); 
                    }; 
                }); 
            },100); 
            return window.dash_clientside.no_update; 
        }
        """,
        Output('room-options-container', 'style'),
        Input('room-options-container', 'children'),
        prevent_initial_call=True
    )

    # Highlight selected missing class option in modal (clientside for performance)
    clientside_callback(
        """
        function(children){ 
            if(!children) return window.dash_clientside.no_update; 
            setTimeout(function(){ 
                const roomOptions = document.querySelectorAll('.room-option'); 
                roomOptions.forEach(function(option){ 
                    option.onclick=function(){ 
                        roomOptions.forEach(function(opt){ opt.classList.remove('selected'); }); 
                        this.classList.add('selected'); 
                    }; 
                }); 
            },100); 
            return window.dash_clientside.no_update; 
        }
        """,
        Output('missing-classes-container', 'style'),
        Input('missing-classes-container', 'children'),
        prevent_initial_call=True
    )

    # Callback to handle missing classes modal - FROM SEPT 13
    @app.callback(
        [Output("missing-classes-container", "children"),
         Output("missing-classes-modal", "style"),
         Output("missing-modal-overlay", "style")],
        [Input("missing-class-data", "data")],
        [State("constraint-details-store", "data"),
         State("student-group-dropdown", "value"),
         State("all-timetables-store", "data")],
        prevent_initial_call=True
    )
    def handle_missing_classes_modal(missing_data, constraint_details, selected_group_idx, all_timetables_data):
        """Handle opening the missing classes modal and populating it with missing classes"""
        if not missing_data or missing_data.get('action') != 'show_missing':
            return dash.no_update, dash.no_update, dash.no_update
        
        if not constraint_details or not all_timetables_data:
            return html.Div("No constraint data available"), {"display": "block"}, {"display": "block"}
        
        # Get current student group
        current_group = all_timetables_data[selected_group_idx]['student_group']
        group_name = current_group['name'] if isinstance(current_group, dict) else current_group.name
        
        # Filter missing classes for current student group
        missing_classes = []
        
        # Only check the 3 specific constraint types
        target_constraints = ['Missing or Extra Classes', 'Same Student Group Overlaps', 'Classes During Break Time']
        
        for constraint_type in target_constraints:
            if constraint_type not in constraint_details:
                continue
                
            for violation in constraint_details[constraint_type]:
                if not isinstance(violation, dict):
                    continue
                    
                # Check if this violation involves the current group
                if constraint_type == 'Missing or Extra Classes':
                    if violation.get('group') == group_name:
                        course_code = violation.get('course', 'Unknown Course')
                        faculty_name = find_lecturer_for_course(course_code, group_name)
                        if faculty_name == "Unknown":
                            faculty_name = violation.get('faculty', 'Unknown Faculty')
                        
                        missing_classes.append({
                            'course': course_code,
                            'faculty': faculty_name,
                            'type': 'Missing Class',
                            'reason': violation.get('reason', 'Missing from schedule')
                        })
                
                elif constraint_type == 'Same Student Group Overlaps':
                    # Based on constraints.py, the violation structure has 'group', 'courses' array, and 'location'
                    if violation.get('group') == group_name and 'courses' in violation:
                        courses = violation.get('courses', [])
                        # Take the first course from the clashing courses for this group
                        course_code = courses[0] if courses else 'Unknown Course'
                        faculty_name = find_lecturer_for_course(course_code, group_name)
                        if faculty_name == "Unknown":
                            faculty_name = violation.get('lecturer', 'Unknown Faculty')
                        
                        missing_classes.append({
                            'course': course_code,
                            'faculty': faculty_name,
                            'type': 'Student Group Clash',
                            'reason': violation.get('location', 'Clash detected')
                        })
                    # Handle case where violation has 'groups' and 'courses' arrays (multiple groups)
                    elif 'groups' in violation and 'courses' in violation:
                        groups = violation.get('groups', [])
                        courses = violation.get('courses', [])
                        if group_name in groups:
                            try:
                                idx = groups.index(group_name)
                                course_code = courses[idx] if idx < len(courses) else courses[0] if courses else 'Unknown Course'
                                faculty_name = find_lecturer_for_course(course_code, group_name)
                                if faculty_name == "Unknown":
                                    faculty_name = violation.get('lecturer', 'Unknown Faculty')
                            except (ValueError, IndexError):
                                course_code = courses[0] if courses else 'Unknown Course'
                                faculty_name = find_lecturer_for_course(course_code, group_name)
                            
                            missing_classes.append({
                                'course': course_code,
                                'faculty': faculty_name,
                                'type': 'Student Group Clash',
                                'reason': violation.get('location', 'Clash detected')
                            })
                
                elif constraint_type == 'Classes During Break Time':
                    if violation.get('group') == group_name:
                        course_code = violation.get('course', 'Unknown Course')
                        # Find the correct lecturer for this course and group
                        faculty_name = find_lecturer_for_course(course_code, group_name)
                        if faculty_name == "Unknown":
                            faculty_name = violation.get('faculty', 'Unknown Faculty')
                        
                        missing_classes.append({
                            'course': course_code,
                            'faculty': faculty_name,
                            'type': 'Break Time Violation',
                            'reason': 'Scheduled during break time'
                        })
        
        if not missing_classes:
            return html.Div("No missing classes found for this student group", 
                           style={"padding": "20px", "textAlign": "center"}), {"display": "block"}, {"display": "block"}
        
        # Create missing class options
        class_options = []
        for idx, missing_class in enumerate(missing_classes):
            # Use actual course and faculty names
            display_course = missing_class['course']
            display_faculty = missing_class.get('faculty', 'Unknown Faculty')
            
            class_options.append(
                html.Button([
                    html.Div([
                        html.Span(display_course, style={"fontWeight": "600"}),
                        html.Span(f" ({missing_class['type']})", style={"fontSize": "11px", "color": "#666", "marginLeft": "8px"})
                    ]),
                    html.Div(
                        display_faculty,
                        style={"fontSize": "12px", "color": "#666", "marginTop": "4px"}
                    )
                ], 
                className="room-option available",
                id={"type": "missing-class-option", "index": idx, "course": display_course, "faculty": display_faculty},
                n_clicks=0,
                style={"width": "100%", "textAlign": "left", "background": "none", "padding": "12px 16px", "cursor": "pointer"})
            )
        
        return class_options, {"display": "block"}, {"display": "block"}

    # Callback to handle missing classes modal close - FROM SEPT 13
    @app.callback(
        [Output("missing-modal-overlay", "style", allow_duplicate=True),
         Output("missing-classes-modal", "style", allow_duplicate=True)],
        [Input("missing-modal-close-btn", "n_clicks"),
         Input("missing-cancel-btn", "n_clicks"),
         Input("missing-modal-overlay", "n_clicks")],
        prevent_initial_call=True
    )
    def handle_missing_modal_close(close_btn_clicks, cancel_btn_clicks, overlay_clicks):
        """Handle closing the missing classes modal"""
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        
        # Hide modal and overlay
        return {"display": "none"}, {"display": "none"}

    # Callback to handle download button click and show download modal - FROM SEPT 13
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
        """Handle showing/hiding the download modal"""
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if trigger_id == "download-button" and download_btn_clicks:
            # Show modal and overlay
            return {"display": "block"}, {"display": "block"}
        elif trigger_id in ["download-modal-close-btn", "download-close-btn", "download-modal-overlay"]:
            # Hide modal and overlay
            return {"display": "none"}, {"display": "none"}
        
        raise dash.exceptions.PreventUpdate

    # Callback to handle download actions - Triggers browser downloads
    @app.callback(
        [Output("download-data", "data"),
         Output("download-status", "children")],
        [Input("download-sst-btn", "n_clicks"),
         Input("download-tyd-btn", "n_clicks"),
         Input("download-lecturer-btn", "n_clicks")],
        [State("all-timetables-store", "data")],
        prevent_initial_call=True
    )
    def handle_download_actions(sst_clicks, tyd_clicks, lecturer_clicks, all_timetables_data):
        """Handle download button clicks - triggers browser downloads with current Dash UI data"""
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
        
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        try:
            # Import output_data functions that return bytes
            from output_data import export_sst_timetables_bytes_from_data, export_tyd_timetables_bytes_from_data, export_lecturer_timetables_bytes_from_data
            
            if trigger_id == "download-sst-btn" and sst_clicks:
                file_bytes, filename_or_error = export_sst_timetables_bytes_from_data(all_timetables_data)
                if file_bytes:
                    return dcc.send_bytes(file_bytes, filename_or_error), html.Div(f"✅ Download started: {filename_or_error}", style={"color": "green", "fontWeight": "600"})
                else:
                    return dash.no_update, html.Div(f"❌ {filename_or_error}", style={"color": "red", "fontWeight": "600"})
                    
            elif trigger_id == "download-tyd-btn" and tyd_clicks:
                file_bytes, filename_or_error = export_tyd_timetables_bytes_from_data(all_timetables_data)
                if file_bytes:
                    return dcc.send_bytes(file_bytes, filename_or_error), html.Div(f"✅ Download started: {filename_or_error}", style={"color": "green", "fontWeight": "600"})
                else:
                    return dash.no_update, html.Div(f"❌ {filename_or_error}", style={"color": "red", "fontWeight": "600"})
                    
            elif trigger_id == "download-lecturer-btn" and lecturer_clicks:
                file_bytes, filename_or_error = export_lecturer_timetables_bytes_from_data(all_timetables_data)
                if file_bytes:
                    return dcc.send_bytes(file_bytes, filename_or_error), html.Div(f"✅ Download started: {filename_or_error}", style={"color": "green", "fontWeight": "600"})
                else:
                    return dash.no_update, html.Div(f"❌ {filename_or_error}", style={"color": "red", "fontWeight": "600"})
        
        except Exception as e:
            return dash.no_update, html.Div(f"❌ Error: {str(e)}", style={"color": "red", "fontWeight": "600"})
        
        raise dash.exceptions.PreventUpdate

    # Callback to store which missing class option is selected AND update visual selection
    @app.callback(
        [Output('selected-missing-class-store', 'data'),
         Output({"type": "missing-class-option", "index": ALL, "course": ALL, "faculty": ALL}, "className")],
        [Input({"type": "missing-class-option", "index": ALL, "course": ALL, "faculty": ALL}, "n_clicks")],
        [State({"type": "missing-class-option", "index": ALL, "course": ALL, "faculty": ALL}, "id"),
         State("missing-class-data", "data")],
        prevent_initial_call=True
    )
    def store_selected_missing_class(n_clicks_list, class_ids, missing_data):
        """Store which missing class option was selected (for visual feedback)"""
        if not any(n_clicks_list) or not missing_data:
            raise dash.exceptions.PreventUpdate

        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        # Find which class was clicked by checking which n_clicks increased
        try:
            # Find the index of the button that was clicked
            clicked_index = None
            for idx, clicks in enumerate(n_clicks_list):
                if clicks > 0:
                    # Check if this is the one that just changed
                    clicked_index = idx
                    break
            
            if clicked_index is None:
                raise dash.exceptions.PreventUpdate
            
            # Get the corresponding class_id
            triggered_class = class_ids[clicked_index]
            selected_course = triggered_class['course']
            selected_faculty = triggered_class.get('faculty', 'Unknown Faculty')
            selected_index = triggered_class.get('index')
            
            result = {
                'course': selected_course,
                'faculty': selected_faculty,
                'cell_id': missing_data['cell_id']
            }
            
            # Update className for all buttons - selected one gets 'selected' class
            classnames = []
            for idx, class_id in enumerate(class_ids):
                if idx == clicked_index:
                    classnames.append("room-option available selected")
                else:
                    classnames.append("room-option available")
            
            # Store the selection and update visual feedback
            return result, classnames
        except Exception as e:
            raise dash.exceptions.PreventUpdate

    # Callback to handle "Add to Slot" button and schedule the selected missing class
    @app.callback(
        [Output("all-timetables-store", "data", allow_duplicate=True),
         Output("manual-cells-store", "data", allow_duplicate=True),
         Output("missing-classes-modal", "style", allow_duplicate=True),
         Output("missing-modal-overlay", "style", allow_duplicate=True),
         Output('selected-missing-class-store', 'data', allow_duplicate=True)],
        Input('missing-add-btn', 'n_clicks'),
        [State('selected-missing-class-store', 'data'),
         State("all-timetables-store", "data"),
         State("student-group-dropdown", "value"),
         State("manual-cells-store", "data")],
        prevent_initial_call=True
    )
    def handle_add_to_slot(add_btn_clicks, selected_class_data, all_timetables_data, selected_group_idx, manual_cells):
        """Schedule the selected missing class when 'Add to Slot' is clicked"""
        if not add_btn_clicks or not selected_class_data:
            raise dash.exceptions.PreventUpdate

        # Extract data from stored selection
        try:
            selected_course = selected_class_data['course']
            selected_faculty_from_id = selected_class_data.get('faculty', 'Unknown Faculty')
            cell_id_str = selected_class_data['cell_id']
            cell_id = json.loads(cell_id_str)
            row_idx = cell_id['row']
            col_idx = cell_id['col']
            group_idx = cell_id['group']
        except Exception as e:
            raise dash.exceptions.PreventUpdate

        # Update timetable with manual scheduling
        updated_timetables = json.loads(json.dumps(all_timetables_data))

        # Use "Unknown" as the room for missing classes
        available_room = "Unknown"

        # Re-fetch the lecturer name to ensure correctness
        current_group_name = all_timetables_data[group_idx]['student_group']['name']
        faculty_name = find_lecturer_for_course(selected_course, current_group_name)

        # If lookup fails, fallback to the name from the button ID
        if faculty_name == "Unknown":
            faculty_name = selected_faculty_from_id
        if faculty_name == "Unknown" or not faculty_name:
            faculty_name = 'Unknown Faculty'

        # Preserve the actual course name
        course_display_name = selected_course
        if course_display_name == 'Unknown Course' or not course_display_name:
            course_display_name = 'MANUAL-CLASS'

        manual_class_content = f"{course_display_name}\n{available_room}\n{faculty_name}"
        updated_timetables[group_idx]['timetable'][row_idx][col_idx + 1] = manual_class_content

        # Update manual cells tracking
        updated_manual_cells = manual_cells.copy() if manual_cells else []
        manual_cell_key = f"{group_idx}_{row_idx}_{col_idx}"
        if manual_cell_key not in updated_manual_cells:
            updated_manual_cells.append(manual_cell_key)

        # Mark that we have swaps
        session_state['has_swaps'] = True

        # Save the updated timetable
        save_timetable_to_file(updated_timetables, updated_manual_cells)

        # Close the modal by hiding it and clear the selection
        return updated_timetables, updated_manual_cells, {"display": "none"}, {"display": "none"}, None

    # Return the configured Dash app
    return app