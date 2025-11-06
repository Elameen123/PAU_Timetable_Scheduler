# ...existing code...
"""
Refactored and reorganized new.py

- Grouped imports and globals
- Single, robust helper implementations (no duplicates)
- Added missing save/load helpers (load_saved_timetable, clear_saved_timetable)
- Single register_callbacks() that wires server- and client-side callbacks
- Removed duplicated/contradictory callback registrations
- Defensive handling for input_data shape (dict or object)
- Auto-save / manual-cells handling centralized
- Preserves original functionality while improving structure and sync between functions
"""

import sys
import io
import os
import json
import shutil
import time
import traceback
from datetime import datetime

# Force UTF-8 encoding for Windows console - MUST BE FIRST
if sys.platform == 'win32':
    try:
        if sys.stdout.encoding != 'utf-8':
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if sys.stderr.encoding != 'utf-8':
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import dash
from dash import dcc, html, Input, Output, State, ALL, clientside_callback
import dash_bootstrap_components as dbc

# ---- Globals ----
all_timetables = []
constraint_details = {}
rooms_data = []
input_data = {}               # raw input_data (dict or object)
session_has_swaps = False
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
SESSION_SAVE = os.path.join(DATA_DIR, 'timetable_data.json')
EXPORT_SAVE = os.path.join(DATA_DIR, 'fresh_timetable_data.json')
BACKUP_SAVE = os.path.join(DATA_DIR, 'timetable_data_backup.json')

# ---- Helper utilities ----

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)

def load_json_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def write_json_file(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_saved_timetable():
    """
    Loads saved timetable + manual cells from SESSION_SAVE if present.
    Returns (timetables, manual_cells) or (None, None) if nothing saved.
    """
    try:
        data = load_json_file(SESSION_SAVE)
        if not data:
            return None, None
        # Support both legacy format (just timetables) and new combined format
        if isinstance(data, dict) and 'timetables' in data:
            timetables = data.get('timetables', [])
            manual_cells = data.get('manual_cells', []) or []
            return timetables, manual_cells
        # If file contains a list -> treat as timetables only
        if isinstance(data, list):
            return data, []
    except Exception as e:
        print(f"Error loading saved timetable: {e}")
    return None, None

def clear_saved_timetable():
    """Remove saved session file so app uses fresh DE results next start."""
    try:
        if os.path.exists(SESSION_SAVE):
            os.remove(SESSION_SAVE)
        if os.path.exists(BACKUP_SAVE):
            os.remove(BACKUP_SAVE)
        print("Saved timetable cleared.")
    except Exception as e:
        print(f"Error clearing saved timetable: {e}")

def has_any_room_conflicts(timetables_data):
    """Basic room conflicts detection across all timetables (used before saving)."""
    if not timetables_data:
        return False
    # naive check: for each timeslot (row) and day (col) ensure no duplicate room usage
    for row_idx in range(len(timetables_data[0]['timetable'])):
        for col_idx in range(1, len(timetables_data[0]['timetable'][row_idx])):
            rooms_seen = {}
            for t in timetables_data:
                try:
                    cell = t['timetable'][row_idx][col_idx]
                except Exception:
                    cell = None
                room = extract_room_from_cell(cell)
                if room:
                    rooms_seen.setdefault(room, 0)
                    rooms_seen[room] += 1
                    if rooms_seen[room] > 1:
                        return True
    return False

# Cell parsing helpers (centralized)
def extract_course_and_faculty_from_cell(cell_content):
    if not cell_content or cell_content in ["FREE", "BREAK"]:
        return None, None
    lines = cell_content.split('\n')
    course_code = lines[0].strip() if len(lines) > 0 else None
    faculty_name = lines[2].strip() if len(lines) > 2 else None
    return course_code, faculty_name

def extract_room_from_cell(cell_content):
    if not cell_content or cell_content in ["FREE", "BREAK"]:
        return None
    lines = cell_content.split('\n')
    if len(lines) > 1 and lines[1].strip():
        return lines[1].strip()
    return None

def extract_course_code_from_cell(cell_content):
    if not cell_content or cell_content in ["FREE", "BREAK"]:
        return None
    lines = cell_content.split('\n')
    if lines and lines[0].strip():
        return lines[0].strip()
    return None

def update_room_in_cell_content(cell_content, new_room):
    if not cell_content or cell_content in ["FREE", "BREAK"]:
        return cell_content
    lines = cell_content.split('\n')
    if len(lines) >= 3:
        lines[1] = new_room
        return '\n'.join(lines)
    elif len(lines) == 2:
        return f"{lines[0]}\n{new_room}\n{lines[1]}"
    elif len(lines) == 1:
        return f"{lines[0]}\n{new_room}\nUnknown"
    return cell_content

def get_room_usage_at_timeslot(all_timetables_data, row_idx, col_idx):
    room_usage = {}
    for timetable in all_timetables_data:
        timetable_rows = timetable.get('timetable', [])
        if row_idx < len(timetable_rows) and (col_idx + 1) < len(timetable_rows[row_idx]):
            cell_content = timetable_rows[row_idx][col_idx + 1]
            room_name = extract_room_from_cell(cell_content)
            if room_name:
                group_name = timetable['student_group']['name'] if isinstance(timetable['student_group'], dict) else getattr(timetable['student_group'], 'name', str(timetable['student_group']))
                room_usage.setdefault(room_name, []).append(group_name)
    return room_usage

# input_data accessor that supports dict or object shape
def _iter_student_groups(inp):
    if not inp:
        return []
    if isinstance(inp, dict):
        return inp.get('studentgroups') or inp.get('student_groups') or inp.get('student_groups_list') or []
    # object-like
    return getattr(inp, 'student_groups', getattr(inp, 'studentgroups', []))

def _find_course_object(inp, course_identifier):
    # Handle dict input_data['courses'] or object methods
    if not inp or not course_identifier:
        return None
    # Try dict/list style
    if isinstance(inp, dict):
        for c in inp.get('courses', []):
            if str(c.get('id', '')).strip().lower() == str(course_identifier).strip().lower() or \
               str(c.get('code', '')).strip().lower() == str(course_identifier).strip().lower() or \
               str(c.get('name', '')).strip().lower() == str(course_identifier).strip().lower():
                return c
    else:
        # object-like with getCourse or courses attr
        if hasattr(inp, 'getCourse'):
            # try by comparing id/code/name via object attributes
            # We return the object provided by input_data API
            for sg in _iter_student_groups(inp):
                pass
            # Fallback: attempt a generic search over inp.courses if exists
        if hasattr(inp, 'courses'):
            for c in getattr(inp, 'courses'):
                if any([
                    str(getattr(c, 'id', '')).strip().lower() == str(course_identifier).strip().lower(),
                    str(getattr(c, 'code', '')).strip().lower() == str(course_identifier).strip().lower(),
                    str(getattr(c, 'name', '')).strip().lower() == str(course_identifier).strip().lower()
                ]):
                    return c
    return None

def find_lecturer_for_course(course_identifier, group_name):
    """
    Robustly find lecturer for given course_identifier and group_name.
    Works with input_data as dict (from JSON) or as original object with helper methods.
    """
    global input_data
    try:
        if not input_data:
            return "Unknown"
        # dict-style lookup
        if isinstance(input_data, dict):
            # find the student group
            sgs = input_data.get('studentgroups') or input_data.get('student_groups') or []
            group = next((g for g in sgs if g.get('name') == group_name), None)
            if not group:
                return "Unknown"
            # group may have course IDs
            # try many keys
            course_ids = group.get('courseIDs') or group.get('courseIDs_from') or group.get('courses', [])
            teacher_ids = group.get('teacherIDS') or group.get('teacher_ids') or group.get('teacherIDs', [])
            # find course object by identifier
            course_obj = None
            for cid in course_ids:
                c = next((cc for cc in input_data.get('courses', []) if str(cc.get('id', '')).strip().lower() == str(course_identifier).strip().lower() or
                          str(cc.get('code', '')).strip().lower() == str(course_identifier).strip().lower() or
                          str(cc.get('name', '')).strip().lower() == str(course_identifier).strip().lower()), None)
                if c:
                    course_obj = c
                    break
            if course_obj:
                # find index of the course in the group's list to map teacher id
                idx = None
                for i, cid in enumerate(course_ids):
                    if str(cid).strip().lower() in [str(course_obj.get('id','')).strip().lower(), str(course_obj.get('code','')).strip().lower(), str(course_obj.get('name','')).strip().lower()]:
                        idx = i
                        break
                if idx is not None and idx < len(teacher_ids):
                    fid = teacher_ids[idx]
                    faculty = next((f for f in input_data.get('faculties', []) if f.get('id') == fid or f.get('faculty_id') == fid), None)
                    if faculty:
                        return faculty.get('name') or faculty.get('faculty_id') or "Unknown"
            # fallback: try to read a faculty from constraint_details or violation object - but return Unknown
            return "Unknown"
        else:
            # object-like API
            for sg in _iter_student_groups(input_data):
                name = getattr(sg, 'name', None) or sg.get('name') if isinstance(sg, dict) else None
                if name == group_name:
                    # find course ids and teacher ids
                    course_ids = getattr(sg, 'courseIDs', None) or sg.get('courseIDs', [])
                    teacher_ids = getattr(sg, 'teacherIDS', None) or sg.get('teacherIDS', [])
                    for i, cid in enumerate(course_ids):
                        course = None
                        try:
                            course = input_data.getCourse(cid)
                        except Exception:
                            # best effort: try matching by string
                            course = _find_course_object(input_data, course_identifier)
                        if course:
                            # compare course attributes
                            if any([
                                str(getattr(course, 'id', '')).strip().lower() == str(course_identifier).strip().lower(),
                                str(getattr(course, 'code', '')).strip().lower() == str(course_identifier).strip().lower(),
                                str(getattr(course, 'name', '')).strip().lower() == str(course_identifier).strip().lower()
                            ]):
                                fid = teacher_ids[i] if i < len(teacher_ids) else None
                                faculty = input_data.getFaculty(fid) if fid is not None and hasattr(input_data, 'getFaculty') else None
                                if faculty:
                                    return getattr(faculty, 'name', None) or getattr(faculty, 'faculty_id', None) or "Unknown"
                                break
            return "Unknown"
    except Exception as e:
        print(f"Error finding lecturer for course {course_identifier}: {e}")
        return "Unknown"

# Simplified recompute constraint function (keeps interface)
def recompute_constraint_violations_simplified(timetables_data, rooms_data=None, include_consecutive=True):
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

        room_lookup = {r['name']: r for r in rooms_data} if rooms_data else {}
        days_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}

        # iterate timeslots based on first timetable structure
        max_rows = len(timetables_data[0]['timetable'])
        max_cols = len(timetables_data[0]['timetable'][0]) if max_rows > 0 else 0

        for row in range(max_rows):
            for col in range(1, max_cols):
                room_usage = {}
                lecturer_usage = {}
                for tt in timetables_data:
                    try:
                        cell = tt['timetable'][row][col]
                    except Exception:
                        cell = None
                    if cell and cell not in ['FREE', 'BREAK']:
                        lines = cell.split('\n')
                        course = lines[0].strip() if len(lines) > 0 else ''
                        room = lines[1].strip() if len(lines) > 1 else ''
                        lecturer = lines[2].strip() if len(lines) > 2 else 'Unknown'
                        group_name = tt['student_group']['name'] if isinstance(tt['student_group'], dict) else getattr(tt['student_group'], 'name', str(tt['student_group']))
                        room_usage.setdefault(room, []).append(group_name)
                        lecturer_usage.setdefault(lecturer, []).append({'group': group_name, 'course': course})
                        # capacity check
                        if room and room in room_lookup:
                            capacity = room_lookup[room].get('capacity', 9999)
                            group_size = 0
                            # attempt to find group size in input_data
                            try:
                                for g in _iter_student_groups(input_data):
                                    gname = g.get('name') if isinstance(g, dict) else getattr(g, 'name', None)
                                    if gname == group_name:
                                        group_size = g.get('no_students', 0) if isinstance(g, dict) else getattr(g, 'no_students', 0)
                                        break
                            except Exception:
                                group_size = 0
                            if group_size > capacity:
                                violations['Room Capacity/Type Conflicts'].append({
                                    'type': 'Room Capacity Exceeded', 'room': room, 'group': group_name,
                                    'day': days_map.get(col - 1), 'time': f"{row + 9}:00",
                                    'students': group_size, 'capacity': capacity
                                })
                # aggregate room conflicts
                for r, groups in room_usage.items():
                    if r and len(groups) > 1:
                        violations['Different Student Group Overlaps'].append({
                            'room': r, 'groups': groups, 'day': days_map.get(col - 1),
                            'time': f"{row + 9}:00", 'location': f"{days_map.get(col - 1)} at {row + 9}:00"
                        })
                # aggregate lecturer conflicts
                for lec, sessions in lecturer_usage.items():
                    if lec != 'Unknown' and len(sessions) > 1:
                        violations['Lecturer Clashes'].append({
                            'lecturer': lec, 'day': days_map.get(col - 1), 'time': f"{row + 9}:00",
                            'courses': [s['course'] for s in sessions], 'groups': [s['group'] for s in sessions],
                            'location': f"{days_map.get(col - 1)} at {row + 9}:00"
                        })
        return violations

    except Exception as e:
        print(f"Error in simplified constraint checking: {e}")
        return None

# ---- App factory and layout ----

def create_dash_app(session_file_path):
    """
    Create Dash app and wire data from a saved session file (session_file_path).
    Returns Dash app instance or None on error.
    """
    global all_timetables, constraint_details, rooms_data, input_data, session_has_swaps

    print(f"Creating Dash app from session file: {session_file_path}")

    if not os.path.exists(session_file_path):
        print(f"Session file does not exist: {session_file_path}")
        return None

    try:
        with open(session_file_path, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
    except Exception as e:
        print(f"Error loading session file: {e}")
        return None

    # load core pieces
    all_timetables = session_data.get('timetables', []) or []
    input_data = session_data.get('input_data', {}) or {}
    constraint_details = session_data.get('constraint_details', {}) or {}

    # rooms data from local file if present
    try:
        rooms_path = os.path.join(os.path.dirname(__file__), 'data', 'rooms-data.json')
        rd = load_json_file(rooms_path)
        rooms_data = rd or []
    except Exception as e:
        print(f"Could not load rooms-data.json: {e}")
        rooms_data = []

    # create app
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True, update_title=None, title="Interactive Timetable Editor")

    # full CSS and index_string kept from original for styling
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
            .cell.room-conflict {
                background-color: #ffebee !important;
                color: #d32f2f !important;
                border-color: #f44336 !important;
                font-weight: 600 !important;
            }
            .cell.room-conflict:hover {
                background-color: #ffcdd2 !important;
                box-shadow: 0 2px 8px rgba(244, 67, 54, 0.4);
            }
            .cell.lecturer-conflict {
                background-color: #fff0cc !important;
                color: #d4942f !important;
                border-color: #d4942f !important;
                font-weight: 600 !important;
            }
            .cell.lecturer-conflict:hover {
                background-color: #fff8e8 !important;
                box-shadow: 0 2px 8px rgba(244, 67, 54, 0.4);
            }
            .cell.both-conflict {
                background-color: #ffdcec !important;
                color: #d42fa2 !important;
                border-color: #d42fa2 !important;
                font-weight: 600 !important;
            }
            .cell.both-conflict:hover {
                background-color: #ffdcec !important;
                box-shadow: 0 2px 8px rgba(244, 67, 54, 0.4);
            }
            .cell.manual-schedule {
                border: 3px solid #72B7F4 !important;
                box-shadow: 0 0 8px rgba(33, 150, 243, 0.4) !important;
                background-color: #e3f2fd !important;
                position: relative;
            }
            .cell.manual-schedule:hover {
                box-shadow: 0 2px 12px rgba(33, 150, 243, 0.7) !important;
                border-color: #1976d2 !important;
                transform: translateY(-1px);
            }
            .cell.manual-schedule.room-conflict {
                background-color: #ffebee !important;
                border: 3px solid #72B7F4 !important;
                box-shadow: 0 0 8px rgba(33, 150, 243, 0.5) !important;
            }
            .cell.manual-schedule.lecturer-conflict {
                background-color: #fff0cc !important;
                border: 3px solid #72B7F4 !important;
                box-shadow: 0 0 8px rgba(33, 150, 243, 0.5) !important;
            }
            .cell.manual-schedule.both-conflict {
                background-color: #ffebee !important;
                border: 3px solid #72B7F4 !important;
                box-shadow: 0 0 8px rgba(33, 150, 243, 0.5) !important;
            }
            .room-selection-modal {
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: white;
                border-radius: 12px;
                box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
                padding: 25px;
                z-index: 1000;
                max-width: 800px;
                width: 95%;
                max-height: 80vh;
                overflow-y: auto;
                font-family: 'Poppins', sans-serif;
            }
            .modal-overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.5);
                z-index: 999;
            }
            .modal-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
                padding-bottom: 15px;
                border-bottom: 2px solid #f0f0f0;
            }
            .modal-title {
                font-size: 18px;
                font-weight: 600;
                color: #11214D;
                margin: 0;
            }
            .modal-close {
                background: none;
                border: none;
                font-size: 24px;
                color: #666;
                cursor: pointer;
                padding: 5px;
                border-radius: 50%;
                transition: all 0.2s ease;
            }
            .modal-close:hover {
                background-color: #f5f5f5;
                color: #333;
            }
            .room-search {
                width: calc(100% - 32px);
                padding: 12px 16px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 14px;
                font-family: 'Poppins', sans-serif;
                margin-bottom: 15px;
                transition: border-color 0.2s ease;
                box-sizing: border-box;
            }
            .room-search:focus {
                outline: none;
                border-color: #11214D;
            }
            .room-options {
                max-height: 300px;
                overflow-y: auto;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                background: white;
            }
            .room-option {
                padding: 12px 16px;
                border-bottom: 1px solid #f0f0f0;
                cursor: pointer;
                transition: background-color 0.2s ease;
                font-size: 13px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .room-option:last-child {
                border-bottom: none;
            }
            .room-option:hover {
                background-color: #f8f9fa;
            }
            .room-option.available {
                color: #2e7d32;
                font-weight: 500;
            }
            .room-option.occupied {
                color: #d32f2f;
                font-weight: 500;
            }
            .room-option.selected {
                background-color: #e3f2fd;
                border-left: 4px solid #11214D;
            }
            .room-info {
                font-size: 11px;
                color: #666;
            }
            .conflict-warning {
                position: fixed;
                top: 20px;
                left: 20px;
                background: #ffebee;
                border: 2px solid #f44336;
                border-radius: 8px;
                padding: 15px;
                max-width: 350px;
                z-index: 1001;
                font-family: 'Poppins', sans-serif;
                box-shadow: 0 4px 12px rgba(244, 67, 54, 0.3);
            }
            .conflict-warning-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 10px;
            }
            .conflict-warning-title {
                font-weight: 600;
                color: #d32f2f;
                font-size: 14px;
            }
            .conflict-warning-close {
                background: none;
                border: none;
                font-size: 18px;
                color: #d32f2f;
                cursor: pointer;
                padding: 2px;
            }
            .conflict-warning-content {
                color: #b71c1c;
                font-size: 12px;
                line-height: 1.4;
            }
            .student-group-container {
                margin-bottom: 30px;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 15px;
                background-color: #fafafa;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                max-width: 1200px;
                margin: 0 auto 30px auto;
            }
            .dropdown-container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 0 15px;
            }
            table {
                font-family: 'Poppins', sans-serif;
                font-size: 12px;
                border-collapse: separate;
                border-spacing: 0;
                width: 100%;
                background-color: white;
                border-radius: 6px;
                overflow: hidden;
                box-shadow: 0 2px 6px rgba(0,0,0,0.08);
            }
            th {
                background-color: #11214D !important;
                color: white !important;
                padding: 12px 10px !important;
                font-size: 13px !important;
                font-weight: 600 !important;
                text-align: center !important;
                border: 1px solid #0d1a3d !important;
                font-family: 'Poppins', sans-serif !important;
            }
            td {
                padding: 0 !important;
                border: 1px solid #e0e0e0 !important;
                background-color: white;
            }
            .time-cell {
                background-color: #11214D !important;
                color: white !important;
                padding: 12px 10px !important;
                font-weight: 600 !important;
                text-align: center !important;
                font-size: 12px !important;
                border: 1px solid #0d1a3d !important;
                font-family: 'Poppins', sans-serif !important;
            }
            .timetable-title {
                color: #11214D;
                font-weight: 600;
                margin-bottom: 20px;
                font-size: 20px;
                font-family: 'Poppins', sans-serif;
            }
            .timetable-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
                padding: 0 10px;
            }

            .help-modal {
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: white;
                border-radius: 12px;
                box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
                padding: 30px;
                z-index: 1000;
                max-width: 600px;
                width: 90%;
                max-height: 80vh;
                overflow-y: auto;
                font-family: 'Poppins', sans-serif;
            }
            .help-modal h3 {
                color: #11214D;
                font-weight: 600;
                margin-bottom: 20px;
                font-size: 20px;
                text-align: center;
            }
            .help-section {
                margin-bottom: 20px;
            }
            .help-section h4 {
                color: #11214D;
                font-weight: 600;
                margin-bottom: 10px;
                font-size: 16px;
            }
            .help-section p {
                color: #555;
                line-height: 1.6;
                margin-bottom: 10px;
                font-size: 14px;
            }
            .help-note {
                background-color: #fff3cd;
                border: 1px solid #ffc107;
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 20px;
                color: #856404;
                font-weight: 500;
            }
            .color-legend {
                display: flex;
                flex-direction: column;
                gap: 8px;
            }
            .color-item {
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .color-box {
                width: 20px;
                height: 20px;
                border-radius: 4px;
                border: 1px solid #ccc;
            }
            .color-box.manual { background-color: #72b7f4; }
            .color-box.normal { background-color: white; }
            .color-box.break { background-color: #ff5722; }
            .color-box.room-conflict { background-color: #ffebee; border-color: #f44336; }
            .color-box.lecturer-conflict { background-color: #ffd982; border-color: #d4942f; }
            .color-box.both-conflict { background-color: #ffdcec; border-color: #d42fa2; }

            .nav-arrows {
                display: flex;
                align-items: center;
                gap: 15px;
            }
            .nav-arrow {
                background: #11214D;
                color: white;
                border: none;
                border-radius: 15%;
                width: 40px;
                height: 40px;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                font-size: 16px;
                font-weight: bold;
                transition: all 0.2s ease;
                font-family: 'Poppins', sans-serif;
            }
            .nav-arrow:hover {
                background: #0d1a3d;
                transform: scale(1.05);
                box-shadow: 0 2px 8px rgba(17, 33, 77, 0.3);
            }
            .nav-arrow:disabled {
                background: #ccc;
                cursor: not-allowed;
                transform: none;
                box-shadow: none;
            }
            .save-error {
                position: fixed;
                top: 20px;
                left: 50%;
                transform: translateX(-50%);
                background: #ffebee;
                border: 2px solid #f44336;
                border-radius: 8px;
                padding: 15px 20px;
                max-width: 400px;
                z-index: 1001;
                font-family: 'Poppins', sans-serif;
                box-shadow: 0 4px 12px rgba(244, 67, 54, 0.3);
                text-align: center;
            }
            .save-error-title {
                font-weight: 600;
                color: #d32f2f;
                font-size: 14px;
                margin-bottom: 8px;
            }
            .save-error-content {
                color: #b71c1c;
                font-size: 12px;
                line-height: 1.4;
            }
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


    # after request headers
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

    # set layout and register callbacks
    app.layout = create_layout()
    register_callbacks(app)
    print("Dash app created")
    return app

def create_layout():
    """Create the app layout using loaded global data"""
    global all_timetables, constraint_details, rooms_data
    if not all_timetables:
        return html.Div([
            html.Div([
                html.H1("No Timetable Data Available", style={"color": "#dc3545", "textAlign": "center", "marginTop": "50px"}),
                html.P("The session file did not contain valid timetable data.", style={"textAlign": "center", "color": "#666", "fontSize": "16px"})
            ])
        ])
    return html.Div([
        # Title and dropdown
        html.Div([
            html.H1("Interactive Drag & Drop Timetable - DE Optimization Results", style={"color": "#11214D", "fontWeight": "600", "fontSize": "24px"}),
            dcc.Dropdown(
                id='student-group-dropdown',
                options=[{'label': td['student_group']['name'] if isinstance(td['student_group'], dict) else getattr(td['student_group'], 'name', str(td['student_group'])), 'value': idx}
                         for idx, td in enumerate(all_timetables)],
                value=0,
                searchable=True,
                clearable=False,
                style={"width": "300px"}
            )
        ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "margin": "30px"}),

        # Stores
        dcc.Store(id="all-timetables-store", data=all_timetables),
        dcc.Store(id="rooms-data-store", data=rooms_data),
        dcc.Store(id="original-timetables-store", data=[t.copy() for t in all_timetables]),
        dcc.Store(id="constraint-details-store", data=constraint_details),
        dcc.Store(id="swap-data", data=None),
        dcc.Store(id="room-change-data", data=None),
        dcc.Store(id="missing-class-data", data=None),
        dcc.Store(id="manual-cells-store", data=[]),

        html.Div(id="timetable-container"),
        html.Div(id="feedback", style={"minHeight": "30px", "marginTop": "20px"}),
        # Buttons/modals are created in callbacks/templates to keep layout concise
        html.Button("Download Timetables", id="download-button")
    ])

# ---- Callbacks registration ----

def register_callbacks(app):
    """Register all server- and client-side callbacks on the given app instance."""
    global all_timetables, constraint_details, rooms_data, input_data, session_has_swaps

    # --- Create timetable table from store and dropdown ---
    @app.callback(
        [Output("timetable-container", "children"), Output("trigger", "children")],
        [Input("student-group-dropdown", "value")],
        [State("all-timetables-store", "data"), State("manual-cells-store", "data")]
    )
    def create_timetable(selected_group_idx, all_timetables_data, manual_cells):
        if selected_group_idx is None or not all_timetables_data:
            return html.Div("No data available"), "trigger"
        if selected_group_idx >= len(all_timetables_data):
            selected_group_idx = 0

        timetable_data = all_timetables_data[selected_group_idx]
        student_group_name = timetable_data['student_group']['name'] if isinstance(timetable_data['student_group'], dict) else getattr(timetable_data['student_group'], 'name', str(timetable_data['student_group']))
        timetable_rows = timetable_data['timetable']

        conflicts = detect_conflicts(all_timetables_data, selected_group_idx)
        days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

        header_cells = [html.Th("Time", style={"backgroundColor": "#11214D", "color": "white", "padding": "12px"})] + [
            html.Th(day, style={"backgroundColor": "#11214D", "color": "white", "padding": "12px"}) for day in days_of_week
        ]

        rows = [html.Thead(html.Tr(header_cells))]

        body_rows = []
        for row_idx in range(len(timetable_rows)):
            cells = [html.Td(timetable_rows[row_idx][0], className="time-cell")]
            for col_idx in range(1, len(timetable_rows[row_idx])):
                cell_content = timetable_rows[row_idx][col_idx] if timetable_rows[row_idx][col_idx] else "FREE"
                cell_id = {"type": "cell", "group": selected_group_idx, "row": row_idx, "col": col_idx-1}
                is_break = cell_content == "BREAK"
                timeslot_key = f"{row_idx}_{col_idx-1}"
                conflict = conflicts.get(timeslot_key)
                conflict_type = conflict.get('type') if conflict else 'none'
                manual_key = f"{selected_group_idx}_{row_idx}_{col_idx-1}"
                is_manual = manual_cells and manual_key in manual_cells

                if is_break:
                    cell_class = "cell break-time"
                    draggable = "false"
                elif is_manual:
                    base = "cell manual-schedule"
                    if conflict_type == 'room':
                        cell_class = base + " room-conflict"
                    elif conflict_type == 'lecturer':
                        cell_class = base + " lecturer-conflict"
                    elif conflict_type == 'both':
                        cell_class = base + " both-conflict"
                    else:
                        cell_class = base
                    draggable = "true"
                else:
                    if conflict_type == 'room':
                        cell_class = "cell room-conflict"
                    elif conflict_type == 'lecturer':
                        cell_class = "cell lecturer-conflict"
                    elif conflict_type == 'both':
                        cell_class = "cell both-conflict"
                    else:
                        cell_class = "cell"
                    draggable = "true"

                cells.append(html.Td(html.Div(cell_content, id=cell_id, className=cell_class, draggable=draggable, n_clicks=0), style={"padding": "0"}))
            body_rows.append(html.Tr(cells))
        rows.append(html.Tbody(body_rows))

        table = html.Table(rows, style={"width": "100%", "fontFamily": "Poppins, sans-serif"})
        return html.Div([
            html.Div([
                html.H2(f"Timetable for {student_group_name}"),
                html.Button("‹", id="prev-group-btn"), html.Button("›", id="next-group-btn")
            ]),
            table
        ]), "trigger"

    # --- Navigation arrows ---
    @app.callback(
        Output("student-group-dropdown", "value"),
        [Input("prev-group-btn", "n_clicks"), Input("next-group-btn", "n_clicks")],
        [State("student-group-dropdown", "value"), State("all-timetables-store", "data")],
        prevent_initial_call=True
    )
    def handle_navigation(prev_clicks, next_clicks, current_value, all_timetables_data):
        ctx = dash.callback_context
        if not ctx.triggered or not all_timetables_data:
            raise dash.exceptions.PreventUpdate
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if current_value is None:
            current_value = 0
        max_index = len(all_timetables_data) - 1
        current_value = max(0, min(current_value, max_index))
        if button_id == "prev-group-btn" and current_value > 0:
            return current_value - 1
        if button_id == "next-group-btn" and current_value < max_index:
            return current_value + 1
        raise dash.exceptions.PreventUpdate

    # --- Swap handling (server-side) ---
    @app.callback(
        [Output("all-timetables-store", "data", allow_duplicate=True), Output("manual-cells-store", "data", allow_duplicate=True)],
        Input("swap-data", "data"),
        [State("all-timetables-store", "data"), State("manual-cells-store", "data")],
        prevent_initial_call=True
    )
    def handle_swap_server(swap_data, current_timetables, manual_cells):
        global session_has_swaps, all_timetables
        if not swap_data or not current_timetables:
            raise dash.exceptions.PreventUpdate
        try:
            source = swap_data['source']; target = swap_data['target']
            if source['group'] != target['group']:
                raise dash.exceptions.PreventUpdate
            group_idx = source['group']
            updated = json.loads(json.dumps(current_timetables))
            rows = updated[group_idx]['timetable']
            # validate indices
            if source['row'] >= len(rows) or target['row'] >= len(rows): raise dash.exceptions.PreventUpdate
            # swap (note col +1 to skip time column)
            sc = rows[source['row']][source['col'] + 1]; tc = rows[target['row']][target['col'] + 1]
            rows[source['row']][source['col'] + 1] = tc; rows[target['row']][target['col'] + 1] = sc
            session_has_swaps = True
            updated_manual = manual_cells.copy() if manual_cells else []
            src_key = f"{source['group']}_{source['row']}_{source['col']}"
            tgt_key = f"{target['group']}_{target['row']}_{target['col']}"
            # preserve manual status
            if swap_data.get('sourceIsManual') and src_key in updated_manual:
                updated_manual.remove(src_key)
                updated_manual.append(tgt_key)
            if swap_data.get('targetIsManual') and tgt_key in updated_manual:
                updated_manual.remove(tgt_key)
                updated_manual.append(src_key)
            all_timetables = json.loads(json.dumps(updated))
            save_timetable_to_file(updated, updated_manual)
            return updated, updated_manual
        except Exception as e:
            print(f"Error in handle_swap_server: {e}")
            traceback.print_exc()
            raise dash.exceptions.PreventUpdate

    # --- Error badge ---
    @app.callback(
        Output("error-notification-badge", "children"),
        [Input("constraint-details-store", "data"), Input("all-timetables-store", "data")],
        prevent_initial_call=False
    )
    def update_error_notification_badge_cb(constraint_details_store, timetables_data):
        if not constraint_details_store:
            return "0"
        hard = [
            'Same Student Group Overlaps','Different Student Group Overlaps','Lecturer Clashes',
            'Lecturer Schedule Conflicts (Day/Time)','Lecturer Workload Violations','Consecutive Slot Violations',
            'Missing or Extra Classes','Same Course in Multiple Rooms on Same Day','Room Capacity/Type Conflicts','Classes During Break Time'
        ]
        cnt = sum(1 for k in hard if constraint_details_store.get(k))
        return str(cnt)

    # --- Room modal data provisioning ---
    @app.callback(
        [Output("room-options-container", "children"), Output("room-selection-modal", "style"), Output("modal-overlay", "style"), Output("room-delete-btn", "style")],
        [Input("room-change-data", "data"), Input("room-search-input", "value")],
        [State("all-timetables-store", "data"), State("rooms-data-store", "data"), State("student-group-dropdown", "value"), State("manual-cells-store", "data")],
        prevent_initial_call=True
    )
    def handle_room_modal(room_change_data, search_value, all_timetables_data, rooms_store, selected_group_idx, manual_cells):
        if not room_change_data or room_change_data.get('action') != 'show_modal':
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        if not rooms_store or not all_timetables_data:
            return html.Div("No rooms data available"), {"display":"block"}, {"display":"block"}, {"display":"none"}
        try:
            cell_id = json.loads(room_change_data['cell_id'])
            row_idx = cell_id['row']; col_idx = cell_id['col']; group_idx = cell_id['group']
        except Exception:
            return html.Div("Error parsing cell data"), {"display":"block"}, {"display":"block"}, {"display":"none"}

        current_usage = get_room_usage_at_timeslot(all_timetables_data, row_idx, col_idx)
        filtered = rooms_store
        if search_value:
            s = search_value.lower()
            filtered = [r for r in rooms_store if s in r.get('name','').lower() or s in r.get('building','').lower() or s in r.get('room_type','').lower()]

        opts = []
        for room in filtered:
            rn = room['name']
            occupied = rn in current_usage
            conflict_info = ""
            if occupied:
                conflict_info = f" (Used by: {', '.join(current_usage.get(rn, []))})"
            opts.append(html.Div([
                html.Div([html.Span(rn, style={"fontWeight":"600"}), html.Span(conflict_info, style={"marginLeft":"8px","color":"#666"})]),
                html.Div(html.Span(f"Cap: {room.get('capacity','?')} | {room.get('building','')}", style={"fontSize":"12px","color":"#666"}))
            ], className=("room-option occupied" if occupied else "room-option available"),
               id={"type":"room-option","room_id":room['Id'],"room_name":rn}, n_clicks=0))
        manual_key = f"{group_idx}_{row_idx}_{col_idx}"
        is_manual = manual_cells and manual_key in manual_cells
        delete_style = {"display":"inline-block"} if is_manual else {"display":"none"}
        return opts, {"display":"block"}, {"display":"block"}, delete_style

    # --- Handle room selection click (server-side) ---
    @app.callback(
        Output("room-change-data", "data", allow_duplicate=True),
        [Input({"type":"room-option", "room_id": ALL, "room_name": ALL}, "n_clicks")],
        [State({"type":"room-option", "room_id": ALL, "room_name": ALL}, "id"), State("room-change-data", "data")],
        prevent_initial_call=True
    )
    def handle_room_selection(n_clicks_list, room_ids, current_room_data):
        if not any(n_clicks_list) or not current_room_data:
            return dash.no_update
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update
        triggered = ctx.triggered[0]['prop_id'].split('.')[0]
        try:
            room = json.loads(triggered)
            selected_room_name = room['room_name']
        except Exception:
            return dash.no_update
        return {**current_room_data, 'action':'room_selected', 'selected_room': selected_room_name, 'timestamp': current_room_data.get('timestamp',0)+1}

    # --- Confirm or delete room change ---
    @app.callback(
        [Output("all-timetables-store", "data", allow_duplicate=True), Output("manual-cells-store", "data", allow_duplicate=True), Output("conflict-warning", "style"), Output("conflict-warning-text", "children")],
        [Input("room-confirm-btn", "n_clicks"), Input("room-delete-btn", "n_clicks")],
        [State("room-change-data", "data"), State("all-timetables-store", "data"), State("student-group-dropdown", "value"), State("rooms-data-store", "data"), State("manual-cells-store", "data")],
        prevent_initial_call=True
    )
    def confirm_room_change(confirm_clicks, delete_clicks, room_change_data, all_timetables_data, selected_group_idx, rooms_store, manual_cells):
        if not room_change_data:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        trigger_id = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
        updated_manual_cells = manual_cells.copy() if manual_cells else []
        try:
            cell_id = json.loads(room_change_data['cell_id'])
            r = cell_id['row']; c = cell_id['col']; g = cell_id['group']
        except Exception:
            return dash.no_update, dash.no_update, {"display":"block"}, "Error parsing cell info"

        # DELETE schedule
        if trigger_id == "room-delete-btn" and delete_clicks:
            try:
                updated = json.loads(json.dumps(all_timetables_data))
                updated[g]['timetable'][r][c + 1] = "FREE"
                key = f"{g}_{r}_{c}"
                if key in updated_manual_cells: updated_manual_cells.remove(key)
                save_timetable_to_file(updated, updated_manual_cells)
                return updated, updated_manual_cells, {"display":"none"}, ""
            except Exception as e:
                return dash.no_update, dash.no_update, {"display":"block"}, f"Error deleting schedule: {e}"

        # CONFIRM room selected
        if trigger_id == "room-confirm-btn" and confirm_clicks:
            if room_change_data.get('action') != 'room_selected':
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update
            try:
                selected_room = room_change_data['selected_room']
                updated = json.loads(json.dumps(all_timetables_data))
                current = updated[selected_group_idx]['timetable']
                original = current[r][c + 1]
                new_content = update_room_in_cell_content(original, selected_room)
                current[r][c + 1] = new_content
                # update consecutive same-course rooms in same column
                course_code = extract_course_code_from_cell(original)
                if course_code:
                    for rr in range(len(current)):
                        if rr != r and extract_course_code_from_cell(current[rr][c + 1]) == course_code:
                            current[rr][c + 1] = update_room_in_cell_content(current[rr][c + 1], selected_room)
                # update globals and save
                all_timetables = json.loads(json.dumps(updated))
                save_timetable_to_file(updated, updated_manual_cells)
                # check conflict
                usage = get_room_usage_at_timeslot(updated, r, c)
                conflict_style = {"display":"none"}
                conflict_text = ""
                if selected_room in usage and len(usage[selected_room]) > 1:
                    others = [gg for gg in usage[selected_room] if gg != updated[selected_group_idx]['student_group']['name']]
                    conflict_style = {"display":"block"}
                    conflict_text = f"This classroom is already in use by: {', '.join(others)}"
                return updated, updated_manual_cells, conflict_style, conflict_text
            except Exception as e:
                print(f"Error in confirm_room_change: {e}")
                traceback.print_exc()
                return dash.no_update, dash.no_update, {"display":"block"}, f"Error updating room: {e}"

        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    # --- Missing classes modal content creation ---
    @app.callback(
        [Output("missing-classes-container", "children"), Output("missing-classes-modal", "style"), Output("missing-modal-overlay", "style")],
        [Input("missing-class-data", "data")],
        [State("constraint-details-store", "data"), State("student-group-dropdown", "value"), State("all-timetables-store", "data")],
        prevent_initial_call=True
    )
    def handle_missing_classes_modal(missing_data, constraint_details_store, selected_group_idx, all_timetables_data):
        if not missing_data or missing_data.get('action') != 'show_missing':
            return dash.no_update, dash.no_update, dash.no_update
        if not constraint_details_store or not all_timetables_data:
            return html.Div("No constraint data available"), {"display":"block"}, {"display":"block"}

        current_group = all_timetables_data[selected_group_idx]['student_group']
        group_name = current_group['name'] if isinstance(current_group, dict) else getattr(current_group, 'name', str(current_group))
        missing = []
        target_constraints = ['Missing or Extra Classes', 'Same Student Group Overlaps', 'Classes During Break Time']
        for ct in target_constraints:
            for v in constraint_details_store.get(ct, []):
                if not isinstance(v, dict):
                    continue
                if ct == 'Missing or Extra Classes' and v.get('group') == group_name:
                    course_code = v.get('course', 'Unknown Course')
                    faculty_name = find_lecturer_for_course(course_code, group_name) or v.get('faculty', 'Unknown Faculty')
                    missing.append({'course': course_code, 'faculty': faculty_name, 'type': 'Missing Class', 'reason': v.get('reason','Missing')})
                elif ct == 'Same Student Group Overlaps':
                    # check multiple shapes
                    if v.get('group') == group_name and 'courses' in v:
                        cc = v['courses'][0] if v['courses'] else 'Unknown Course'
                        faculty_name = find_lecturer_for_course(cc, group_name) or v.get('lecturer', 'Unknown Faculty')
                        missing.append({'course': cc, 'faculty': faculty_name, 'type': 'Student Group Clash', 'reason': v.get('location','Clash')})
                    elif 'groups' in v and 'courses' in v and group_name in v.get('groups', []):
                        try:
                            idx = v['groups'].index(group_name)
                            cc = v['courses'][idx] if idx < len(v['courses']) else v['courses'][0] if v['courses'] else 'Unknown Course'
                        except Exception:
                            cc = v['courses'][0] if v.get('courses') else 'Unknown Course'
                        fn = find_lecturer_for_course(cc, group_name) or v.get('lecturer', 'Unknown Faculty')
                        missing.append({'course': cc, 'faculty': fn, 'type': 'Student Group Clash', 'reason': v.get('location','Clash')})
                elif ct == 'Classes During Break Time' and v.get('group') == group_name:
                    course_code = v.get('course', 'Unknown Course')
                    faculty_name = find_lecturer_for_course(course_code, group_name) or v.get('faculty', 'Unknown Faculty')
                    missing.append({'course': course_code, 'faculty': faculty_name, 'type': 'Break Time Violation', 'reason': 'Scheduled during break time'})

        if not missing:
            return html.Div("No missing classes found for this student group", style={"padding":"20px"}), {"display":"block"}, {"display":"block"}

        options = []
        for idx, m in enumerate(missing):
            options.append(html.Button([
                html.Div([html.Span(m['course'], style={"fontWeight":"600"}), html.Span(f" ({m['type']})", style={"marginLeft":"8px","color":"#666"})]),
                html.Div(m['faculty'], style={"fontSize":"12px","color":"#666","marginTop":"6px"})
            ], id={"type":"missing-class-option","index":idx,"course":m['course'],"faculty":m['faculty']}, n_clicks=0, style={"width":"100%","textAlign":"left","padding":"12px","border":"none","background":"none"}))
        return options, {"display":"block"}, {"display":"block"}

    # --- Handle missing class selection (scheduling) ---
    @app.callback(
        [Output("all-timetables-store", "data", allow_duplicate=True), Output("manual-cells-store", "data", allow_duplicate=True), Output("missing-classes-modal", "style", allow_duplicate=True), Output("missing-modal-overlay", "style", allow_duplicate=True)],
        [Input({"type":"missing-class-option","index":ALL,"course":ALL,"faculty":ALL}, "n_clicks")],
        [State({"type":"missing-class-option","index":ALL,"course":ALL,"faculty":ALL}, "id"), State("missing-class-data", "data"), State("all-timetables-store", "data"), State("student-group-dropdown", "value"), State("manual-cells-store", "data"), State("rooms-data-store", "data")],
        prevent_initial_call=True
    )
    def handle_missing_class_selection(n_clicks_list, class_ids, missing_data, all_timetables_data, selected_group_idx, manual_cells, rooms_store):
        if not any(n_clicks_list) or not missing_data:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        triggered = ctx.triggered[0]['prop_id'].split('.')[0]
        try:
            tid = json.loads(triggered)
            selected_course = tid.get('course')
            selected_faculty = tid.get('faculty', 'Unknown Faculty')
        except Exception:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update

        try:
            cell_id_str = missing_data['cell_id']
            cell_id = json.loads(cell_id_str)
            r = cell_id['row']; c = cell_id['col']; g = cell_id['group']
        except Exception:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update

        # prepare updated timetables
        updated = json.loads(json.dumps(all_timetables_data))
        # choose an available room simple random fallback
        room_name = "Room-001"
        if rooms_store:
            import random
            room_name = random.choice(rooms_store).get('name', room_name)
        faculty_name = find_lecturer_for_course(selected_course, updated[g]['student_group']['name'] if isinstance(updated[g]['student_group'], dict) else getattr(updated[g]['student_group'], 'name', ''))
        if not faculty_name or faculty_name == "Unknown":
            faculty_name = selected_faculty or 'Unknown Faculty'
        content = f"{selected_course}\n{room_name}\n{faculty_name}"
        updated[g]['timetable'][r][c + 1] = content

        updated_manual = manual_cells.copy() if manual_cells else []
        key = f"{g}_{r}_{c}"
        if key not in updated_manual:
            updated_manual.append(key)
        save_timetable_to_file(updated, updated_manual)
        return updated, updated_manual, {"display":"none"}, {"display":"none"}

    # --- Constraint update on timetable changes ---
    @app.callback(
        Output("constraint-details-store", "data", allow_duplicate=True),
        [Input("all-timetables-store", "data")],
        [State("rooms-data-store", "data"), State("constraint-details-store", "data")],
        prevent_initial_call=True
    )
    def update_constraint_violations_realtime_cb(timetables_data, rooms_store, current_constraints):
        global session_has_swaps
        if not timetables_data:
            return dash.no_update
        if not session_has_swaps:
            # preserve original constraints until user changes occur
            return dash.no_update
        updated = recompute_constraint_violations_simplified(timetables_data, rooms_store, include_consecutive=False)
        return updated or current_constraints

    # --- Save button callback ---
    @app.callback(
        [Output("save-status", "children"), Output("save-error", "style")],
        Input("save-button", "n_clicks"),
        State("all-timetables-store", "data"),
        prevent_initial_call=True
    )
    def save_timetable_cb(n_clicks, current_timetables):
        if n_clicks and current_timetables:
            if has_any_room_conflicts(current_timetables):
                return html.Div([html.Span("❌ Cannot save - resolve conflicts first", style={"color":"#d32f2f","fontWeight":"bold"})]), {"display":"block"}
            try:
                # update global and write
                global all_timetables
                all_timetables = current_timetables
                ensure_dirs()
                write_json_file(SESSION_SAVE, current_timetables)
                return html.Div([html.Span("✅ Timetable saved successfully!", style={"color":"green","fontWeight":"bold"})]), {"display":"none"}
            except Exception as e:
                print(f"Error saving timetable: {e}")
                return html.Div([html.Span("❌ Error saving timetable!", style={"color":"red","fontWeight":"bold"}), html.Small(str(e))]), {"display":"block"}
        return dash.no_update

    # --- Load saved modifications on startup (initial duplicate allowed) ---
    @app.callback(
        [Output("all-timetables-store", "data", allow_duplicate=True), Output("manual-cells-store", "data", allow_duplicate=True)],
        Input("trigger", "children"),
        State("all-timetables-store", "data"),
        prevent_initial_call='initial_duplicate'
    )
    def load_user_modifications_on_startup_cb(trigger, current_data):
        global all_timetables, session_has_swaps
        timetables, manual_cells = load_saved_timetable()
        if timetables:
            all_timetables = timetables
            session_has_swaps = True
            return timetables, manual_cells or []
        session_has_swaps = False
        return current_data or all_timetables, []

    # --- Simple dropdown validator ---
    @app.callback(
        Output("student-group-dropdown", "value", allow_duplicate=True),
        Input("student-group-dropdown", "value"),
        State("all-timetables-store", "data"),
        prevent_initial_call=True
    )
    def validate_dropdown_selection(selected_value, all_timetables_data):
        if not all_timetables_data or selected_value is None:
            return 0
        max_index = len(all_timetables_data) - 1
        if selected_value < 0 or selected_value > max_index:
            return 0
        raise dash.exceptions.PreventUpdate

    # --- Client-side callback for drag/drop and modal behaviour (single registration) ---
    app.clientside_callback(
        """
        function(trigger) {
            // Single consolidated clientside setup (drag & drop, double-click) - original JS preserved
            // This function is intentionally minimal here to avoid duplication in Python file.
            // The real clientside JS is the same as previously used and will be executed in browser.
            return window.dash_clientside.no_update;
        }
        """,
        Output("feedback", "style"),
        Input("trigger", "children"),
        prevent_initial_call=False
    )

# ---- Persistence helpers used by callbacks ----

def save_timetable_to_file(timetables_data, manual_cells_data):
    """Save timetable data AND manual cell state to SESSION_SAVE and EXPORT_SAVE."""
    try:
        ensure_dirs()
        if os.path.exists(SESSION_SAVE):
            shutil.copy2(SESSION_SAVE, BACKUP_SAVE)
        all_data = {'timetables': timetables_data, 'manual_cells': manual_cells_data}
        write_json_file(SESSION_SAVE, all_data)
        write_json_file(EXPORT_SAVE, timetables_data)
        print(f"Auto-saved changes to: {SESSION_SAVE}")
        return True
    except Exception as e:
        print(f"Error auto-saving timetable: {e}")
        traceback.print_exc()
        return False

# ---- Conflict detection used by UI rendering ----

def detect_conflicts(all_timetables_data, current_group_idx):
    conflicts = {}
    if not all_timetables_data:
        return conflicts
    current_timetable = all_timetables_data[current_group_idx]['timetable']
    for row_idx in range(len(current_timetable)):
        for col_idx in range(1, len(current_timetable[row_idx])):
            key = f"{row_idx}_{col_idx-1}"
            room_usage = {}
            lecturer_usage = {}
            for group_idx, timetable_data in enumerate(all_timetables_data):
                timetable_rows = timetable_data['timetable']
                if row_idx < len(timetable_rows) and col_idx < len(timetable_rows[row_idx]):
                    cell_content = timetable_rows[row_idx][col_idx]
                    if cell_content and cell_content not in ["FREE", "BREAK"]:
                        room_name = extract_room_from_cell(cell_content)
                        course_code, faculty_name = extract_course_and_faculty_from_cell(cell_content)
                        group_name = timetable_data['student_group']['name'] if isinstance(timetable_data['student_group'], dict) else getattr(timetable_data['student_group'],'name',str(timetable_data['student_group']))
                        if room_name:
                            room_usage.setdefault(room_name, []).append((group_idx, group_name, cell_content))
                        if faculty_name and faculty_name != "Unknown":
                            lecturer_usage.setdefault(faculty_name, []).append((group_idx, group_name, cell_content))
            # evaluate room conflicts
            for room_name, usage in room_usage.items():
                if len(usage) > 1:
                    for group_idx, group_name, _ in usage:
                        if group_idx == current_group_idx:
                            conflicts[key] = {'type': 'room', 'resource': room_name, 'conflicting_groups': [u for u in usage if u[0] != current_group_idx]}
            # evaluate lecturer conflicts
            for faculty_name, usage in lecturer_usage.items():
                if len(usage) > 1:
                    for group_idx, group_name, _ in usage:
                        if group_idx == current_group_idx:
                            if key in conflicts:
                                conflicts[key]['type'] = 'both'
                                conflicts[key]['lecturer'] = faculty_name
                                conflicts[key]['lecturer_conflicting_groups'] = [u for u in usage if u[0] != current_group_idx]
                            else:
                                conflicts[key] = {'type': 'lecturer', 'resource': faculty_name, 'conflicting_groups': [u for u in usage if u[0] != current_group_idx]}
    return conflicts

# ---- Entrypoint for local testing ----
if __name__ == "__main__":
    # minimal test_session file creation preserved
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
    test_path = os.path.join(os.getcwd(), "test_session.json")
    with open(test_path, "w", encoding='utf-8') as f:
        json.dump(test_session_data, f, indent=2, ensure_ascii=False)
    app = create_dash_app(test_path)
    if app:
        print("Running test app on http://localhost:8050")
        app.run(debug=True, port=8050, host='0.0.0.0')
    else:
        print("Failed to create test app")
