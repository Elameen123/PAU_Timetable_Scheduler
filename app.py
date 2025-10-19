#!/usr/bin/env python3
import os
import sys
import uuid
import tempfile
import threading
import subprocess
import time
import socket
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
import traceback

# Project imports
from transformer_api import transform_excel_to_json, validate_excel_structure
from input_data_api import initialize_input_data_from_json
from differential_evolution import DifferentialEvolution
from export_service import create_export_service
from data_converter import TimetableDataConverter

app = Flask(__name__)

# Simple, single CORS configuration
CORS(app, 
     origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization", "Accept"],
     supports_credentials=False
)

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'change-this-in-prod')
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

# Thread-safe job storage
processing_jobs = {}
generated_timetables = {}
job_locks = {}
dash_sessions = {}

export_service = create_export_service()

# Continue with your functions...


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_job_lock(upload_id):
    if upload_id not in job_locks:
        job_locks[upload_id] = threading.Lock()
    return job_locks[upload_id]


def make_json_serializable(obj):
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(i) for i in obj]
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    try:
        if hasattr(obj, '__dict__'):
            result = {}
            for attr in dir(obj):
                if attr.startswith('_'):
                    continue
                val = getattr(obj, attr, None)
                if callable(val):
                    continue
                try:
                    result[attr] = make_json_serializable(val)
                except Exception:
                    continue
            # Try a few common attributes
            for common in ('id', 'name', 'code', 'title', 'value'):
                if common not in result and hasattr(obj, common):
                    try:
                        result[common] = make_json_serializable(getattr(obj, common))
                    except Exception:
                        continue
            return result if result else str(obj)
    except Exception:
        pass
    try:
        return str(obj)
    except Exception:
        return "unserializable_object"


def update_job_status(upload_id, status=None, progress=None, error=None, result=None):
    lock = get_job_lock(upload_id)
    with lock:
        if upload_id not in processing_jobs:
            return
        job = processing_jobs[upload_id]
        if status is not None:
            job['status'] = str(status)
        if progress is not None:
            job['progress'] = int(progress)
        if error is not None:
            job['error'] = str(error)
        if result is not None:
            job['result'] = make_json_serializable(result)
        print(f"[{upload_id}] status={job.get('status')} progress={job.get('progress')} error={job.get('error')}")


def debug_result_data(upload_id, result_data):
    print(f"\n[{upload_id}] === RESULT DATA DEBUG ===")
    if not result_data:
        print(f"[{upload_id}] Result is empty")
        return
    try:
        print(f"[{upload_id}] keys: {list(result_data.keys()) if isinstance(result_data, dict) else type(result_data)}")
        tt = result_data.get('timetables', result_data.get('timetables_raw', []))
        print(f"[{upload_id}] timetables count: {len(tt) if tt else 0}")
        input_data = result_data.get('input_data')
        print(f"[{upload_id}] input_data present: {input_data is not None}")
    except Exception as e:
        print(f"[{upload_id}] Debug error: {e}")


# Consolidated serialization that matches expected frontend keys
def serialize_input_data(input_data):
    """
    Convert input_data object into a dictionary using robust attribute lookups.
    The output keys align with the frontend/validation expectations:
      - courses: list of dicts, each containing student_groupsID and facultyId
      - rooms, studentgroups, faculties, days, hours
    """
    try:
        courses = []
        for idx, course in enumerate(getattr(input_data, 'courses', []) or []):
            # student_groups IDs
            sg_ids = []
            if hasattr(course, 'student_groups'):
                for sg in getattr(course, 'student_groups') or []:
                    sg_id = getattr(sg, 'id', None) or getattr(sg, 'Id', None) or getattr(sg, 'group_id', None)
                    sg_ids.append(sg_id)
            elif hasattr(course, 'studentGroupIds'):
                sg_ids = getattr(course, 'studentGroupIds') or []
            elif hasattr(course, 'student_groupsID'):
                sg_ids = getattr(course, 'student_groupsID') or []
                
            # faculty id
            faculty_id = None
            if hasattr(course, 'faculty'):
                faculty = getattr(course, 'faculty')
                faculty_id = getattr(faculty, 'id', None) or getattr(faculty, 'faculty_id', None)
            elif hasattr(course, 'facultyId'):
                faculty_id = getattr(course, 'facultyId')

            courses.append({
                "id": getattr(course, 'id', idx),
                "name": getattr(course, 'name', getattr(course, 'code', f"Course {idx}")),
                "student_groupsID": sg_ids,
                "facultyId": faculty_id,
                "hours_per_week": getattr(course, 'hours_per_week', getattr(course, 'credits', 3)),
                "requires_lab": getattr(course, 'requires_lab', False),
                "department": getattr(course, 'department', '')
            })

        rooms = []
        for idx, room in enumerate(getattr(input_data, 'rooms', []) or []):
            rooms.append({
                "id": getattr(room, 'id', getattr(room, 'Id', idx)),
                "name": getattr(room, 'name', getattr(room, 'Name', f"Room {idx}")),
                "capacity": getattr(room, 'capacity', getattr(room, 'Capacity', 50)),
                "type": getattr(room, 'room_type', getattr(room, 'type', 'classroom'))
            })

        studentgroups = []
        for idx, sg in enumerate(getattr(input_data, 'student_groups', []) or []):
            studentgroups.append({
                "id": getattr(sg, 'id', idx),
                "name": getattr(sg, 'name', f"Group {idx}"),
                "size": getattr(sg, 'no_students', getattr(sg, 'size', 30)),
                "department": getattr(sg, 'department', getattr(sg, 'dept', '')),
                "level": getattr(sg, 'level', ''),
                "courseIDs": getattr(sg, 'courseIDs', []),
                "hours_required": getattr(sg, 'hours_required', [])
            })

        faculties = []
        for idx, f in enumerate(getattr(input_data, 'faculties', []) or []):
            faculties.append({
                "id": getattr(f, 'id', idx),
                "name": getattr(f, 'name', f"Faculty {idx}"),
                "department": getattr(f, 'department', ''),
                "unavailable_times": getattr(f, 'unavailable_times', [])
            })

        return {
            "courses": courses,
            "rooms": rooms,
            "studentgroups": studentgroups,
            "faculties": faculties,
            "days": getattr(input_data, 'days', 5),
            "hours": getattr(input_data, 'hours', 8)
        }
    except Exception as e:
        print(f"serialize_input_data error: {e}")
        return None


class TimetableProcessor:
    def __init__(self, upload_id, input_data, config):
        self.upload_id = upload_id
        self.input_data = input_data
        self.config = config or {}

    def update_job_progress(self, job_id, pct=None, message=None):
        if pct is None:
            lock = get_job_lock(job_id)
            with lock:
                pct = processing_jobs.get(job_id, {}).get('progress', 0)
        update_job_status(job_id, progress=int(pct), status="processing")

    def update_job_result(self, job_id, result):
        update_job_status(job_id, status="completed", progress=100, result=result)

    def update_job_error(self, job_id, error_msg):
        update_job_status(job_id, status="error", error=error_msg)

    def run_optimization(self):
        job_id = self.upload_id
        try:
            update_job_status(job_id, status='processing', progress=10)
            
            pop_size = int(self.config.get('population_size', 50))
            max_gen = int(self.config.get('max_generations', 5))
            mutation_factor = float(self.config.get('mutation_factor', 0.4))
            crossover_rate = float(self.config.get('crossover_rate', 0.9))

            # 1. Instantiate the engine from differential_evolution.py
            print(f"[{job_id}] Initializing DifferentialEvolution engine...")
            de_engine = DifferentialEvolution(self.input_data, pop_size, mutation_factor, crossover_rate)
            
            # (Optional) Progress update can be integrated here if run() is a loop
            update_job_status(job_id, progress=25)

            # 2. Run the optimization
            print(f"[{job_id}] Running optimization for {max_gen} generations...")
            best_solution, fitness_history, generations_completed, diversity_history = de_engine.run(max_gen)

            update_job_status(job_id, progress=90)
            
            # 3. Get the final fitness and detailed violations
            final_fitness = de_engine.evaluate_fitness(best_solution)
            constraint_details = de_engine.constraints.get_detailed_constraint_violations(best_solution)

            # 4. Format the output for the frontend
            print(f"[{job_id}] Formatting timetable data for frontend...")
            all_timetables_raw = de_engine.print_all_timetables(best_solution, self.input_data.days, self.input_data.hours)
            
            # Convert StudentGroup objects to dictionaries for JSON
            all_timetables_json = []
            for tt_data in all_timetables_raw:
                group = tt_data['student_group']
                all_timetables_json.append({
                    'student_group': {
                        'id': group.id,
                        'name': group.name,
                        'department': getattr(group, 'department', ''),
                        'level': getattr(group, 'level', '')
                    },
                    'timetable': tt_data['timetable']
                })
            
            # 5. Package the final result
            final_result = {
                'success': True,
                'timetables': all_timetables_json,
                'best_fitness': final_fitness,
                'constraint_details': constraint_details,
                'generations_completed': generations_completed,
                'fitness_history': fitness_history,
                'diversity_history': diversity_history
            }
            
            update_job_status(job_id, status='completed', progress=100, result=final_result)
            print(f"[{job_id}] Optimization completed successfully.")

        except Exception as e:
            print(f"[{job_id}] An error occurred during optimization.")
            traceback.print_exc()
            update_job_status(job_id, status='error', error=str(e))

    def create_basic_timetables(self, de, solution, job_id):
        """Basic timetable creation fallback (keeps structure expected by frontend)."""
        timetables = []
        try:
            student_groups = getattr(de, 'student_groups', [])
            days = getattr(self.input_data, 'days', 5)
            hours = getattr(self.input_data, 'hours', 8)
            for sg in student_groups:
                grid = []
                for h in range(hours):
                    row = [f"{9+h}:00"]
                    for d in range(days):
                        row.append("FREE")
                    grid.append(row)
                timetables.append({'student_group': sg, 'timetable': grid})
        except Exception as e:
            print(f"[{job_id}] create_basic_timetables error: {e}")
        return timetables


# --------- API Endpoints ---------
@app.route('/', methods=['GET'])
def index():
    return jsonify({'status': 'ok', 'message': 'Timetable Generator API is running.'}), 200


@app.route('/upload-excel', methods=['POST'])
def upload_excel():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload .xlsx or .xls files only'}), 400

    try:
        upload_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{upload_id}_{filename}")
        file.save(file_path)
        print(f"[{upload_id}] Uploaded file saved to: {file_path}")

        is_valid, validation_message = validate_excel_structure(file_path)
        if not is_valid:
            return jsonify({'error': f'Excel validation failed: {validation_message}'}), 400

        json_data = transform_excel_to_json(file_path)

        input_data = initialize_input_data_from_json(json_data)

        # small test init of DE to catch immediate errors (non-fatal)
        try:
            DifferentialEvolution(input_data, 10, 0.4, 0.9)
        except Exception as e:
            print(f"[{upload_id}] DE initialization warning: {e}")

        generated_timetables[upload_id] = {
            'input_data': input_data,
            'file_path': file_path,
            'filename': filename,
            'upload_time': datetime.now().isoformat(),
            'json_data': json_data
        }

        # attempt a summary (non-fatal)
        try:
            summary = input_data.get_data_summary()
        except Exception:
            summary = {}

        preview_data = {
            'student_groups': summary.get('student_groups', 0),
            'courses': summary.get('courses', 0),
            'rooms': summary.get('rooms', 0),
            'faculties': summary.get('faculties', 0),
            'total_student_capacity': summary.get('total_student_capacity', 0),
        }

        return jsonify({
            'success': True,
            'upload_id': upload_id,
            'filename': filename,
            'file_size': os.path.getsize(file_path),
            'preview': preview_data,
            'debug_info': {
                'courses': len(getattr(input_data, 'courses', [])),
                'rooms': len(getattr(input_data, 'rooms', [])),
                'student_groups': len(getattr(input_data, 'student_groups', [])),
                'faculties': len(getattr(input_data, 'faculties', []))
            }
        }), 200

    except Exception as exc:
        print(f"[Upload] Error: {exc}")
        return jsonify({'error': f'Failed to process Excel file: {str(exc)}'}), 500


@app.route('/generate-timetable', methods=['POST'])
def generate_timetable():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400

    upload_id = data.get('upload_id')
    config = data.get('config', {}) or {}

    if not upload_id:
        return jsonify({'error': 'upload_id is required'}), 400
    if upload_id not in generated_timetables:
        return jsonify({'error': 'Invalid upload ID. Please upload an Excel file first.'}), 400

    lock = get_job_lock(upload_id)
    with lock:
        if upload_id in processing_jobs and processing_jobs[upload_id]['status'] == 'processing':
            return jsonify({'error': 'Timetable generation already in progress for this upload'}), 409

        processing_jobs[upload_id] = {
            'status': 'processing',
            'progress': 0,
            'start_time': datetime.now().isoformat(),
            'config': make_json_serializable(config),
            'error': None,
            'result': None,
            'input_data': None
        }

    stored = generated_timetables[upload_id]
    input_data = stored['input_data']

    processor = TimetableProcessor(upload_id, input_data, config)

    pop_size = int(config.get('population_size', 50))
    max_gen = int(config.get('max_generations', 40))
    F = float(config.get('F', config.get('mutation_factor', 0.4)))
    CR = float(config.get('CR', config.get('crossover_rate', 0.9)))

    ### MODIFICATION: The 'args' tuple is now empty
    thread = threading.Thread(
        target=processor.run_optimization,
        args=(),  # Corrected line: No arguments are passed directly
        daemon=True
    )
    thread.start()

    return jsonify({
        'success': True,
        'upload_id': upload_id,
        'message': 'Timetable generation started',
        'config': config,
        'estimated_time_minutes': max_gen * 0.05
    }), 202


@app.route('/get-timetable-status/<upload_id>', methods=['GET'])
def get_timetable_status(upload_id):
    if upload_id not in processing_jobs:
        return jsonify({'error': 'No processing job found for this upload ID'}), 404
    try:
        lock = get_job_lock(upload_id)
        with lock:
            job = dict(processing_jobs[upload_id])
        serialized_job = make_json_serializable(job)
        response = {
            'upload_id': str(upload_id),
            'status': str(serialized_job.get('status', 'unknown')),
            'progress': int(serialized_job.get('progress', 0)),
            'start_time': str(serialized_job.get('start_time', '')),
        }
        if serialized_job.get('status') == 'completed':
            response.update({
                'message': 'Timetable generation completed successfully',
                'result': serialized_job.get('result', {})
            })
        elif serialized_job.get('status') == 'error':
            response.update({
                'message': f"Generation failed: {serialized_job.get('error')}",
                'error': serialized_job.get('error')
            })
        else:
            response.update({'message': f"Processing... {response['progress']}% complete"})
        return jsonify(response), 200
    except Exception as e:
        print(f"get_timetable_status error: {e}")
        return jsonify({'error': f'Status check failed: {str(e)}'}), 500


@app.route('/export-timetable', methods=['POST'])
def export_timetable():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400

    upload_id = data.get('upload_id')
    format_type = (data.get('format') or 'excel').lower()

    if not upload_id:
        return jsonify({'error': 'upload_id is required'}), 400
    if upload_id not in processing_jobs:
        return jsonify({'error': 'Invalid upload ID or no results available'}), 404

    lock = get_job_lock(upload_id)
    with lock:
        job = dict(processing_jobs[upload_id])

    if job.get('status') != 'completed':
        return jsonify({'error': f'Timetable not ready for export. Status: {job.get("status")}' }), 400

    try:
        result = job.get('result')
        if not result:
            return jsonify({'error': 'No timetable data available for export'}), 500

        timetable_data = result.get('timetables_raw', []) or result.get('timetables', [])
        if not timetable_data:
            return jsonify({'error': 'No timetable data found in results'}), 500

        if format_type == 'excel':
            excel_buffer = export_service.export_to_excel(timetable_data)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
            tmp.write(excel_buffer.getvalue())
            tmp.close()
            return send_file(tmp.name, as_attachment=True,
                             download_name=f'timetable_{upload_id}.xlsx',
                             mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        elif format_type == 'pdf':
            pdf_buffer = export_service.export_to_pdf(timetable_data)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            tmp.write(pdf_buffer.getvalue())
            tmp.close()
            return send_file(tmp.name, as_attachment=True,
                             download_name=f'timetable_{upload_id}.pdf',
                             mimetype='application/pdf')
        else:
            return jsonify({'error': f'Unsupported format: {format_type}. Supported: excel, pdf'}), 400

    except Exception as exc:
        print(f"Export error: {exc}")
        return jsonify({'error': f'Export failed: {str(exc)}'}), 500


@app.route('/timeslots', methods=['GET', 'OPTIONS'])
def get_time_slots():
    if request.method == 'OPTIONS':
        return '', 200
    time_slots = [
        {'start': '09:00', 'end': '10:00', 'label': '9:00 AM'},
        {'start': '10:00', 'end': '11:00', 'label': '10:00 AM'},
        {'start': '11:00', 'end': '12:00', 'label': '11:00 AM'},
        {'start': '12:00', 'end': '13:00', 'label': '12:00 PM'},
        {'start': '13:00', 'end': '14:00', 'label': '1:00 PM (Break)'},
        {'start': '14:00', 'end': '15:00', 'label': '2:00 PM'},
        {'start': '15:00', 'end': '16:00', 'label': '3:00 PM'},
        {'start': '16:00', 'end': '17:00', 'label': '4:00 PM'},
    ]
    return jsonify(time_slots), 200


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0-clean'
    }), 200


# Simple Dash launching helper (creates session JSON and launches a small script in background)
def find_dash_port(start_port=8050, max_port=8100):
    for port in range(start_port, max_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    return start_port


@app.route('/create-dash-session', methods=['POST', 'OPTIONS'])
@cross_origin()
def create_dash_session_fixed():
    """ENHANCED version of dash session creation with proper error handling"""
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        data = request.get_json() or {}
        upload_id = data.get('uploadId') or data.get('upload_id')
        
        if not upload_id or upload_id not in processing_jobs:
            return jsonify({'error': 'Invalid or missing upload ID'}), 400
            
        job_data = processing_jobs[upload_id]
        if job_data.get('status') != 'completed':
            return jsonify({'error': 'Timetable generation not completed'}), 400
        
        # Get the result data with enhanced error checking
        result = job_data.get('result', {}) or {}
        timetables = result.get('timetables', []) or result.get('timetables_raw', [])
        
        if not timetables:
            print(f"[{upload_id}] No timetables found in result: {list(result.keys())}")
            return jsonify({'error': 'No timetables available'}), 400
        
        # Enhanced data validation and transformation
        processed_timetables = []
        for i, timetable in enumerate(timetables):
            if not isinstance(timetable, dict):
                print(f"[{upload_id}] Timetable {i} is not a dict: {type(timetable)}")
                continue
                
            # Ensure required fields exist
            if 'timetable' not in timetable:
                print(f"[{upload_id}] Timetable {i} missing 'timetable' field")
                continue
                
            # Validate timetable grid structure
            grid = timetable['timetable']
            if not isinstance(grid, list) or not grid:
                print(f"[{upload_id}] Timetable {i} has invalid grid structure")
                continue
                
            processed_timetables.append(timetable)
        
        if not processed_timetables:
            return jsonify({'error': 'No valid timetables found after processing'}), 400
        
        # Get input data with fallback logic
        input_data_raw = result.get('input_data') or generated_timetables.get(upload_id, {}).get('input_data')
        if not input_data_raw:
            print(f"[{upload_id}] No input data found")
            return jsonify({'error': 'No input data available'}), 400
        
        # Serialize input data properly
        if isinstance(input_data_raw, dict):
            input_data_dict = input_data_raw
        else:
            input_data_dict = serialize_input_data(input_data_raw)
        
        if not input_data_dict:
            return jsonify({'error': 'Failed to serialize input data'}), 500
        
        # Create session data with enhanced validation
        try:
            session_data = TimetableDataConverter.create_session_file(
                processed_timetables, input_data_dict, upload_id
            )
            
            # Validate session data structure
            if 'timetables' not in session_data or not session_data['timetables']:
                raise ValueError("Session data missing timetables")
                
        except Exception as e:
            print(f"[{upload_id}] Session data creation failed: {e}")
            return jsonify({'error': f'Session creation failed: {str(e)}'}), 500
        
        # Find available port
        port = find_dash_port()
        dash_url = f"http://127.0.0.1:{port}"
        
        # Create session file with proper error handling
        try:
            session_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
            json.dump(session_data, session_file, indent=2, default=str)
            session_file.close()
            print(f"[{upload_id}] Created session file: {session_file.name}")
        except Exception as e:
            print(f"[{upload_id}] Failed to create session file: {e}")
            return jsonify({'error': 'Failed to create session file'}), 500
        
        # Enhanced launcher script with better error handling
        launcher_script = f'''
import sys
import os
import traceback
import signal

# Set up signal handlers for graceful shutdown
def signal_handler(signum, frame):
    print(f"Received signal {{signum}}, shutting down...")
    os._exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Add current directory to Python path
sys.path.insert(0, '.')

try:
    print("Starting Dash app...")
    from dash_server_interactive import create_dash_app
    
    app = create_dash_app(r"{session_file.name}")
    if app:
        print(f"Dash app created successfully, starting on port {port}")
        try:
            # UPDATED: Use app.run() instead of app.run_server()
            app.run(
                host='127.0.0.1', 
                port={port}, 
                debug=False,
                use_reloader=False  # Important for subprocess
            )
        except Exception as server_e:
            print(f"Server startup error: {{server_e}}")
            traceback.print_exc()
    else:
        print("Failed to create dash app - check session data")
        sys.exit(1)
        
except Exception as e:
    print(f"Dash launch error: {{e}}")
    traceback.print_exc()
    sys.exit(1)
'''
        
        try:
            script_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
            script_file.write(launcher_script)
            script_file.close()
            
            # Start Dash process with enhanced logging
          
            
            proc = subprocess.Popen(
                [sys.executable, script_file.name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',  # Force UTF-8 encoding
                errors='replace',   # Replace undecodable characters
                bufsize=1,
                universal_newlines=True
            )
            
            # Wait a moment for the server to start
            time.sleep(3)
            
            # Check if process is still running
            if proc.poll() is not None:
                stdout, _ = proc.communicate()
                print(f"[{upload_id}] Dash process died immediately: {stdout}")
                return jsonify({'error': 'Dash server failed to start'}), 500
            
            # Store session info
            dash_sessions[upload_id] = {
                'port': port,
                'process': proc,
                'session_file': session_file.name,
                'script_file': script_file.name,
                'created_at': datetime.now().isoformat(),
                'url': dash_url
            }
            
            print(f"[{upload_id}] Dash session created successfully on port {port}")
            
            return jsonify({
                'success': True,
                'dash_url': dash_url,
                'port': port,
                'session_id': upload_id,
                'message': 'Interactive session created successfully'
            }), 200
            
        except Exception as e:
            print(f"[{upload_id}] Process creation failed: {e}")
            return jsonify({'error': f'Failed to start Dash process: {str(e)}'}), 500
        
    except Exception as exc:
        print(f"[{upload_id}] Create dash session error: {exc}")
        traceback.print_exc()
        return jsonify({'error': f'Session creation failed: {str(exc)}'}), 500
    
    
@app.route('/launch-dash', methods=['GET'])
@app.route('/launch-dash/<upload_id>', methods=['GET'])
def launch_dash_endpoint(upload_id=None):
    # Launch in background thread and return expected URL (attempt)
    def _launch(u_id):
        # call create_dash_session_fixed inside a Flask test_request_context so it can read JSON
        try:
            with app.test_request_context(json={'uploadId': u_id}):
                create_dash_session_fixed()
        except Exception:
            pass

    if upload_id is None:
        # pick latest completed
        completed = [uid for uid, j in processing_jobs.items() if j.get('status') == 'completed']
        upload_id = completed[-1] if completed else None
        if upload_id is None:
            return jsonify({'success': False, 'message': 'No completed uploads found'}), 400

    thread = threading.Thread(target=_launch, args=(upload_id,), daemon=True)
    thread.start()
    port = dash_sessions.get(upload_id, {}).get('port', find_dash_port())
    return jsonify({'success': True, 'message': 'Dash starting', 'url': f'http://localhost:{port}', 'upload_id': upload_id}), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    host = '0.0.0.0'
    print(f"Starting Timetable Generator API on {host}:{port}")
    app.run(host=host, port=port, debug=False)