#!/usr/bin/env python3
"""
Corrected app.py - Timetable Generator API
Fixed issues:
1. DifferentialEvolution constructor signature and method calls
2. Thread-safe job status updates
3. Proper error handling and defensive programming
4. Consistent method signatures across API calls
5. Fixed evolve() method calls without parameters
6. Proper initialization flow
"""

import os
import json
import uuid
import tempfile
import threading
import numpy as np
import random
from datetime import datetime
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
# Dash UI is now disabled - all UI handled by React frontend
# from Dash_UI import create_app  # Use the new lightweight Dash UI factory

# Your project imports (must exist in repo)
from transformer_api import transform_excel_to_json, validate_excel_structure
from input_data_api import initialize_input_data_from_json
from differential_evolution_api import DifferentialEvolution
from export_service import create_export_service, TimetableExportService
from constraints import Constraints

# Keep the API pipeline consistent by default.
# If you explicitly want to try the OG implementation, set USE_OG_DE=1.
DifferentialEvolutionClass = DifferentialEvolution
if os.environ.get('USE_OG_DE', '').strip() in {'1', 'true', 'TRUE', 'yes', 'YES'}:
    try:
        import importlib.util
        base_dir = os.path.dirname(__file__)
        og_candidates = [
            os.path.join(base_dir, 'differential_evolution.py'),
            os.path.join(base_dir, 'differential-evolution.py'),
        ]
        for path in og_candidates:
            if os.path.exists(path):
                spec = importlib.util.spec_from_file_location('de_og_module', path)
                if spec and spec.loader:
                    de_og_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(de_og_module)
                    if hasattr(de_og_module, 'DifferentialEvolution'):
                        DifferentialEvolutionClass = getattr(de_og_module, 'DifferentialEvolution')
                        print(f"Using OG DifferentialEvolution from: {os.path.basename(path)}")
                        break
    except Exception as e:
        print(f"Warning: Could not load OG DifferentialEvolution. Using API version. Error: {e}")

# --- Config & app setup ---
FRONTEND_HTML_PATH = Path(__file__).parent / "timetable_generator.html"

app = Flask(__name__)

# Dash UI is now disabled - all UI handled by React frontend
# # Create the Dash app instance from the new UI module
# # This Dash instance is later mounted under /interactive/
# dash_app = create_app()

# Configure CORS explicitly for local dev and typical headers
CORS(
    app,
    resources={r"/*": {"origins": [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost",
        "http://127.0.0.1",
        "*"
    ]}},
    supports_credentials=False,
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With", "Origin", "Accept"],
    expose_headers=["Content-Type", "Content-Disposition"]
)

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'change-this-in-prod')

# CORS headers for API responses
@app.after_request
def add_cors_headers(resp):
    try:
        # Allow cross-origin requests from React dev server and production
        origin = request.headers.get('Origin')
        if origin:
            resp.headers['Access-Control-Allow-Origin'] = origin
        else:
            resp.headers['Access-Control-Allow-Origin'] = '*'
        resp.headers['Access-Control-Allow-Credentials'] = 'false'
    except Exception:
        pass
    return resp

ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

# Thread-safe job storage with locks
processing_jobs = {}       # upload_id -> { status, progress, result, error, ... }
generated_timetables = {}  # upload_id -> stored input/metadata
job_locks = {}             # upload_id -> threading.Lock()

# Exporter instance
export_service = create_export_service()


# --- Helpers ---
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_job_lock(upload_id):
    """Get or create a lock for a specific job"""
    if upload_id not in job_locks:
        job_locks[upload_id] = threading.Lock()
    return job_locks[upload_id]


def make_json_serializable(obj):
    """
    Convert custom objects to JSON-serializable format
    Recursively handles complex nested structures
    """
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif hasattr(obj, '__dict__'):
        # Convert custom objects to dictionaries using their attributes
        result = {}
        try:
            # Get all non-private, non-method attributes
            for attr_name in dir(obj):
                if (not attr_name.startswith('_') and 
                    not callable(getattr(obj, attr_name, None))):
                    try:
                        attr_value = getattr(obj, attr_name)
                        # Skip methods, properties, and complex objects that might cause recursion
                        if not callable(attr_value):
                            result[attr_name] = make_json_serializable(attr_value)
                    except (AttributeError, TypeError, ValueError):
                        # Skip attributes that can't be accessed or serialized
                        continue
                        
            # Also try common attributes that might not show up in dir()
            for common_attr in ['id', 'name', 'code', 'title', 'value']:
                if (hasattr(obj, common_attr) and 
                    common_attr not in result):
                    try:
                        attr_val = getattr(obj, common_attr)
                        if attr_val is not None and not callable(attr_val):
                            result[common_attr] = str(attr_val)
                    except (AttributeError, TypeError):
                        continue
                        
            return result if result else str(obj)
        except Exception:
            # If all else fails, convert to string
            return str(obj)
    else:
        # Fallback: convert to string
        try:
            return str(obj)
        except Exception:
            return "unserializable_object"


def update_job_status(upload_id, status=None, progress=None, error=None, result=None):
    """Thread-safe update to the processing_jobs dict with JSON serialization."""
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
            # Ensure result is JSON serializable before storing
            job['result'] = make_json_serializable(result)
        
        # simple stdout log for debugging
        print(f"[{upload_id}] status={job.get('status')} progress={job.get('progress')} error={job.get('error')}")


# --- Timetable Processor ---
class TimetableProcessor:
    def __init__(self, upload_id, input_data, config):
        self.upload_id = upload_id
        self.input_data = input_data
        self.config = config or {}
        self.start_time = datetime.now()

    def update_job_progress(self, job_id, pct=None, message=None):
        """Update job progress with thread safety"""
        if pct is None:
            lock = get_job_lock(job_id)
            with lock:
                pct = processing_jobs.get(job_id, {}).get('progress', 0)
        update_job_status(job_id, progress=int(pct), status="processing")

    def update_job_result(self, job_id, result):
        update_job_status(job_id, status="completed", progress=100, result=result)

    def update_job_error(self, job_id, error_msg):
        update_job_status(job_id, status="error", error=error_msg)

    def make_timetables_json_safe(self, all_timetables):
        """Convert timetables into a JSON-friendly structure to avoid deep recursion."""
        safe_list = []
        for item in all_timetables or []:
            try:
                sg = item.get('student_group')
                # Extract basic identifiers only
                sg_name = None
                sg_id = None
                if sg is not None:
                    try:
                        if hasattr(sg, 'name') and getattr(sg, 'name') is not None:
                            sg_name = str(getattr(sg, 'name'))
                    except Exception:
                        pass
                    for attr in ['id', 'student_group_id', 'group_id']:
                        try:
                            if hasattr(sg, attr) and getattr(sg, attr) is not None:
                                sg_id = str(getattr(sg, attr))
                                break
                        except Exception:
                            continue
                if not sg_name:
                    # Fallback to string repr (kept minimal)
                    try:
                        sg_name = str(sg) if sg is not None else 'Unknown Group'
                    except Exception:
                        sg_name = 'Unknown Group'

                # Timetable grid rows
                rows = item.get('timetable', [])
                rows = make_json_serializable(rows)

                safe_list.append({
                    'student_group': {
                        'name': sg_name,
                        'id': sg_id
                    },
                    'timetable': rows
                })
            except Exception:
                # Skip any problematic entry instead of blocking the whole job
                continue
        return safe_list

    def build_empty_timetables(self, de):
        """Build empty grids per student group so UI can render even if optimization failed."""
        try:
            days = int(getattr(de.input_data, 'days', 5) or 5)
            # Keep fallback aligned with the core scheduler (hours=9).
            hours = int(getattr(de.input_data, 'hours', 9) or 9)
            day_start_time = 9
            data = []
            for sg in getattr(de.input_data, 'student_groups', []) or []:
                rows = []
                for h in range(hours):
                    time_label = f"{day_start_time + h}:00"
                    row = [time_label] + ["FREE" for _ in range(days)]
                    rows.append(row)
                data.append({"student_group": sg, "timetable": rows})
            return data
        except Exception as e:
            print(f"Warning: build_empty_timetables failed: {e}")
            return []

    def run_optimization(self, job_id, input_data, pop_size, max_gen, F, CR):
        """
        Runs the differential evolution optimization in the background.
        This version is modified to be closer to the Sept 13 `differential_evolution.py` script's flow.
        """
        self.start_time = datetime.now()
        de = None
        try:
            # Ensure randomized runs
            try:
                import secrets
                seed = secrets.randbits(64)
                random.seed(seed)
                np.random.seed(None) # Use system entropy for numpy
                print(f"[{job_id}] RNG seeded for this run (seed bits set) with pop={pop_size}, gens={max_gen}")
            except Exception as seed_err:
                print(f"[{job_id}] Warning: RNG seeding failed: {seed_err}")

            # Initialize DE with correct parameters
            try:
                de = DifferentialEvolutionClass(input_data, pop_size, F, CR)
            except TypeError:
                try:
                    de = DifferentialEvolutionClass(input_data=input_data, pop_size=pop_size, F=F, CR=CR)
                except TypeError:
                    de = DifferentialEvolutionClass(input_data, pop_size)
        except Exception as e:
            print(f"FATAL: DifferentialEvolution initialization failed: {e}")
            self.update_job_error(job_id, f"DE Initialization failed: {e}")
            return

        best_solution = None
        fitness_history = []
        final_generation = 0
        best_fitness = float("inf")

        try:
            self.update_job_progress(job_id, pct=5)

            # This block mirrors the Sept 13 differential_evolution.py execution flow
            # The run method returns: best_solution, fitness_history, final_generation, diversity_history
            if hasattr(de, 'run'):
                try:
                    print(f"[{job_id}] Starting DE algorithm run for {max_gen} generations...")
                    run_result = de.run(max_gen)
                    
                    # Sept 13 version returns exactly 4 values
                    if isinstance(run_result, tuple) and len(run_result) >= 2:
                        best_solution = run_result[0]
                        fitness_history = run_result[1]
                        
                        # Optional: capture generation and diversity if available
                        if len(run_result) > 2:
                            final_generation = run_result[2]
                        else:
                            final_generation = len(fitness_history) if fitness_history else max_gen
                            
                        if len(run_result) > 3:
                            diversity_history = run_result[3]
                    else:
                        # Fallback if return format is unexpected
                        best_solution = run_result if not isinstance(run_result, tuple) else run_result[0]
                        fitness_history = []
                        final_generation = max_gen
                    
                    # Get the best fitness score
                    if fitness_history and len(fitness_history) > 0:
                        best_fitness = float(fitness_history[-1])
                        print(f"[{job_id}] DE completed - Final fitness: {best_fitness:.4f}")
                    else:
                        best_fitness = float("inf")
                        print(f"[{job_id}] DE completed - No fitness history available")

                    # CRITICAL: The Sept 13 version re-evaluates the best solution with verbose=True
                    # to get final violations printed to console
                    if best_solution is not None and hasattr(de, 'evaluate_fitness'):
                        print(f"[{job_id}] Re-evaluating final best solution for constraint violations...")
                        try:
                            # Try with verbose flag if supported, otherwise call without it
                            de.evaluate_fitness(best_solution, verbose=True)
                        except TypeError:
                            # API version doesn't support verbose parameter
                            de.evaluate_fitness(best_solution)
                    
                    self.update_job_progress(job_id, pct=85)

                except Exception as e:
                    print(f"[{job_id}] ERROR: de.run() failed. Error: {e}")
                    import traceback
                    traceback.print_exc()
                    best_solution = None
                    fitness_history = []
                    final_generation = 0
            else:
                # Fallback if `run` method is not present
                print(f"[{job_id}] ERROR: DE instance lacks a `run` method.")
                self.update_job_error(job_id, "DE instance is missing the 'run' method.")
                return

        except Exception as main_error:
            print(f"[{job_id}] FATAL ERROR in run_optimization: {main_error}")
            import traceback
            traceback.print_exc()
            self.update_job_error(job_id, f"Optimization failed: {str(main_error)}")
            return
        finally:
            # Progress update after optimization finishes
            self.update_job_progress(job_id, pct=90)


        # Post-processing
        self.update_job_progress(job_id, pct=95)

        # Final repairs if method exists
        if best_solution is not None:
            try:
                if hasattr(de, 'verify_and_repair_course_allocations'):
                    best_solution = de.verify_and_repair_course_allocations(best_solution)
            except Exception as e:
                print(f"Warning: Course allocation repair failed: {e}")

        # Generate timetables
        all_timetables = []
        if best_solution is not None:
            try:
                # The original script uses `print_all_timetables` to get the final grid data.
                # CRITICAL: print_all_timetables requires 4 parameters matching Sept 13 signature
                if hasattr(de, 'print_all_timetables'):
                    days = getattr(de.input_data, 'days', 5)
                    # Keep fallback aligned with the core scheduler (hours=9).
                    hours = getattr(de.input_data, 'hours', 9)
                    day_start_time = 9
                    print(f"[{job_id}] Generating timetables with days={days}, hours={hours}, start_time={day_start_time}")
                    all_timetables = de.print_all_timetables(best_solution, days, hours, day_start_time)
                    print(f"[{job_id}] Generated {len(all_timetables) if all_timetables else 0} timetables")
                else:
                    print(f"[{job_id}] Warning: DE instance lacks `print_all_timetables` method.")
            except Exception as e:
                print(f"[{job_id}] ERROR: Timetable generation failed: {e}")
                import traceback
                traceback.print_exc()
                all_timetables = []

        # Fallback: if empty, build blank timetables so UI still works
        if not all_timetables and de is not None:
            print(f"[{job_id}] No timetables produced; building empty grids as fallback")
            all_timetables = self.build_empty_timetables(de)

        # Build UI card summaries
        timetable_cards = self.format_timetable_results_from_raw(all_timetables)

        # Get constraint violations - BOTH summary and detailed for UI
        violations = {}
        detailed_violations = {}
        if best_solution is not None and hasattr(de, 'constraints'):
            try:
                # Get summary violations (numerical counts)
                if hasattr(de.constraints, 'get_constraint_violations'):
                    violations = de.constraints.get_constraint_violations(best_solution, debug=True)
                    print(f"[{job_id}] Got constraint violations summary: {violations.get('total', 0)} total violations")
                
                # Get detailed violations (with locations and descriptions) for UI
                if hasattr(de.constraints, 'get_detailed_constraint_violations'):
                    detailed_violations = de.constraints.get_detailed_constraint_violations(best_solution)
                    print(f"[{job_id}] Got detailed constraint violations with {len(detailed_violations)} constraint types")
                    # Print summary of detailed violations
                    for constraint_type, violation_list in detailed_violations.items():
                        if violation_list:
                            print(f"  - {constraint_type}: {len(violation_list)} violations")
            except Exception as e:
                print(f"Warning: Constraint violation check failed: {e}")
                import traceback
                traceback.print_exc()

        # Build parsed timetables for frontend
        parsed = []
        if all_timetables:
            exporter = TimetableExportService()
            for item in all_timetables:
                try:
                    student_group = item.get("student_group")
                    if hasattr(student_group, "name"):
                        group_name = str(student_group.name)
                    else:
                        group_name = str(student_group)
                    rows = exporter._grid_to_rows(item.get("timetable", []))
                    serializable_rows = make_json_serializable(rows)
                    parsed.append({"group": group_name, "rows": serializable_rows})
                except Exception as e:
                    print(f"Warning: Parsing timetable failed: {e}")

        # Convert raw timetables into a safe, lightweight structure
        self.update_job_progress(job_id, pct=99)
        safe_all_timetables = self.make_timetables_json_safe(all_timetables)

        # Make all result data JSON serializable
        result = {
            "timetables": timetable_cards,
            "timetables_raw": safe_all_timetables,
            "parsed_timetables": parsed,
            "fitness_score": best_fitness if best_fitness != float("inf") else None,
            "generations_completed": int(final_generation) + 1 if isinstance(final_generation, (int, np.integer)) else 1,
            "fitness_history": fitness_history[-20:] if isinstance(fitness_history, list) else [],
            "summary": make_json_serializable(self.generate_summary_safe(de, best_solution, violations)) if best_solution is not None else {},
            # Backwards-compatible summary counts
            "constraint_violations": make_json_serializable(violations),
            # Detailed per-violation payload used by the frontend constraint breakdown menu
            # (locations, groups, courses, etc.).
            "constraint_violation_details": make_json_serializable(detailed_violations) if detailed_violations else {},
            "performance_metrics": {
                "population_size": pop_size,
                "total_events": len(getattr(de, "events_list", [])),
                "scheduled_events": self.count_scheduled_events(best_solution),
                "optimization_time_seconds": (datetime.now() - self.start_time).total_seconds(),
            },
        }

        # Persist violations separately for Dash UI and print a concise summary to console
        try:
            dash_data_dir = os.path.join(os.path.dirname(__file__), 'data')
            os.makedirs(dash_data_dir, exist_ok=True)
            
            # Save DETAILED violations for Dash UI (with locations and descriptions)
            violations_path = os.path.join(dash_data_dir, 'constraint_violations.json')
            with open(violations_path, 'w', encoding='utf-8') as vf:
                import json as _json
                # Save the detailed violations, not the summary
                detailed_to_save = make_json_serializable(detailed_violations) if detailed_violations else {}
                _json.dump(detailed_to_save, vf, indent=2, ensure_ascii=False)
                print(f"[{job_id}] Saved {len(detailed_to_save)} detailed constraint types to {violations_path}")
            
            # Print a compact summary
            try:
                if isinstance(detailed_violations, dict):
                    print("[DETAILED VIOLATIONS] Summary:")
                    for k, val in detailed_violations.items():
                        try:
                            count = len(val) if isinstance(val, list) else (int(val) if isinstance(val, (int, float)) else 1)
                        except Exception:
                            count = 0
                        if count > 0:
                            print(f"  - {k}: {count}")
                else:
                    print("[DETAILED VIOLATIONS] No detailed violations dict available")
            except Exception as _log_err:
                print(f"[DETAILED VIOLATIONS] Warning: could not print summary: {_log_err}")
        except Exception as dash_save_error:
            print(f"[{job_id}] WARNING: Could not save constraint violations for Dash UI. Error: {dash_save_error}")
            import traceback
            traceback.print_exc()

        try:
            dash_data_dir = os.path.join(os.path.dirname(__file__), 'data')
            os.makedirs(dash_data_dir, exist_ok=True)
            dash_save_path = os.path.join(dash_data_dir, 'timetable_data.json')
            fresh_save_path = os.path.join(dash_data_dir, 'fresh_timetable_data.json')

            if safe_all_timetables:
                import json
                data_to_save = {
                    'timetables': safe_all_timetables,
                    'manual_cells': []
                }
                # Write session file used by Dash for persistence + manual edits
                with open(dash_save_path, 'w', encoding='utf-8') as f:
                    json.dump(data_to_save, f, indent=2)
                # Write fresh results file preferred by Dash on startup
                try:
                    with open(fresh_save_path, 'w', encoding='utf-8') as f2:
                        json.dump(safe_all_timetables, f2, indent=2)
                    print(f"[{job_id}] Successfully saved data for Dash UI at: {dash_save_path} and fresh: {fresh_save_path}")
                except Exception as fresh_err:
                    print(f"[{job_id}] WARNING: Could not save fresh timetable JSON. Error: {fresh_err}")

        except Exception as dash_save_error:
            print(f"[{job_id}] WARNING: Could not save data for Dash UI. Error: {dash_save_error}")

        # Save and mark job completed
        self.update_job_result(job_id, result)
        return result

    def count_scheduled_events(self, solution):
        """Count scheduled events in solution"""
        if solution is None:
            return 0
        
        count = 0
        try:
            if isinstance(solution, np.ndarray):
                # Count non-None values
                count = np.count_nonzero(solution != None)
            else:
                # Handle other solution formats
                for room_schedule in solution:
                    if not room_schedule:
                        continue
                    for event in room_schedule:
                        if event is not None:
                            count += 1
        except Exception as e:
            print(f"Warning: Could not count scheduled events: {e}")
        
        return count

    def format_timetable_results_from_raw(self, all_timetables):
        """Create compact UI cards from raw timetables - JSON serializable"""
        timetables = []
        if not all_timetables:
            return timetables
            
        for timetable_data in all_timetables:
            try:
                student_group = timetable_data.get('student_group', {})
                timetable_rows = timetable_data.get('timetable', [])

                courses = set()
                total_hours = 0
                
                for row in timetable_rows:
                    if not row or len(row) < 2:
                        continue
                    # Skip time label (first column)
                    for i, cell in enumerate(row[1:], 1):
                        if cell and 'Course:' in str(cell) and 'BREAK' not in str(cell).upper():
                            try:
                                course_part = str(cell).split('Course:')[1].split(',')[0].strip()
                                if course_part and course_part != "Unknown":
                                    courses.add(course_part)
                                    total_hours += 1
                            except Exception:
                                continue

                # Safely extract group information and convert to JSON-safe types
                title = "Unknown Group"
                student_group_id = None
                student_count = 0
                
                if hasattr(student_group, "name"):
                    title = str(student_group.name)
                elif hasattr(student_group, "id"):
                    title = f"Group {student_group.id}"
                    
                if hasattr(student_group, 'id'):
                    student_group_id = str(student_group.id)  # Convert to string
                    
                if hasattr(student_group, 'no_students'):
                    student_count = int(student_group.no_students) if student_group.no_students else 0

                timetables.append({
                    'title': title,
                    'department': str(self.extract_department(student_group)),
                    'level': str(self.extract_level(student_group)),
                    'student_group_id': student_group_id,
                    'courses': [str(c) for c in list(courses)[:10]],  # Convert to strings
                    'total_courses': len(courses),
                    'total_hours_scheduled': total_hours,
                    'student_count': student_count
                })
            except Exception as e:
                print(f"Warning: Error formatting timetable card: {e}")
                continue
                
        return timetables

    def extract_level(self, student_group):
        """Extract level from student group"""
        try:
            if hasattr(student_group, 'level') and student_group.level:
                return f"{student_group.level} Level"
            
            name = getattr(student_group, "name", "") or ""
            name_lower = name.lower()
            
            if "year 1" in name_lower or name.startswith("1"):
                return "100 Level"
            elif "year 2" in name_lower or name.startswith("2"):
                return "200 Level"
            elif "year 3" in name_lower or name.startswith("3"):
                return "300 Level"
            elif "year 4" in name_lower or name.startswith("4"):
                return "400 Level"
        except Exception:
            pass
        return "Unknown Level"

    def extract_department(self, student_group):
        """Extract department from student group"""
        try:
            if hasattr(student_group, 'dept') and getattr(student_group, 'dept'):
                return student_group.dept
            
            name = getattr(student_group, "name", "") or ""
            parts = name.split()
            if len(parts) > 1:
                return ' '.join(parts[1:])
        except Exception:
            pass
        return "Unknown Department"

    def generate_summary_safe(self, de, best_solution, violations):
        """Safe summary builder that won't crash if properties are missing - returns JSON serializable data"""
        try:
            total_events = len(getattr(de, 'events_list', []))
        except Exception:
            total_events = 0
        
        scheduled_events = self.count_scheduled_events(best_solution)
        
        # Calculate completion rates safely
        group_completion_rates = []
        try:
            student_groups = getattr(de, 'student_groups', [])
            for student_group in student_groups:
                expected = sum(getattr(student_group, 'hours_required', []) or [])
                actual = 0
                
                try:
                    if hasattr(de, 'count_course_occurrences'):
                        counts = de.count_course_occurrences(best_solution, student_group)
                        actual = sum(counts.values()) if isinstance(counts, dict) else 0
                except Exception:
                    actual = 0
                
                if expected > 0:
                    group_completion_rates.append((actual / expected) * 100)
        except Exception:
            group_completion_rates = []

        avg_completion_rate = (sum(group_completion_rates) / len(group_completion_rates) 
                              if group_completion_rates else 0)

        # Safe fitness evaluation
        fitness_score = None
        try:
            if best_solution is not None and hasattr(de, 'evaluate_fitness'):
                fitness_score = float(de.evaluate_fitness(best_solution))  # Ensure it's a float
        except Exception:
            fitness_score = None

        # Ensure all values are JSON serializable
        return {
            'total_student_groups': int(len(getattr(de, 'student_groups', []))),
            'total_courses': int(len(getattr(de, 'courses', []))),
            'total_rooms': int(len(getattr(de, 'rooms', []))),
            'total_events': int(total_events),
            'scheduled_events': int(scheduled_events),
            'completion_rate': float(avg_completion_rate),
            'scheduling_efficiency': float((scheduled_events / total_events * 100) if total_events > 0 else 0),
            'hard_constraints_satisfied': bool(violations.get('total', float('inf')) < 100 if isinstance(violations, dict) else False),
            'fitness_score': fitness_score,
            'constraint_satisfaction_score': float(max(0, 100 - violations.get('total', 100)) if isinstance(violations, dict) else 0),
            'groups_fully_scheduled': int(len([r for r in group_completion_rates if r >= 100]))
        }


# --- API Endpoints ---

# JSON 500 handler for unexpected errors
@app.errorhandler(500)
def handle_internal_error(e):
    return jsonify({
        'error': 'Internal server error',
        'details': str(e)
    }), 500

@app.route('/upload-excel', methods=['POST'])
def upload_excel():
    """Upload Excel and transform into internal input_data used by DE."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload .xlsx or .xls files only'}), 400

    # Extra guard for legacy .xls
    ext = file.filename.rsplit('.', 1)[1].lower()
    if ext == 'xls':
        try:
            import xlrd  # type: ignore
            ver = getattr(xlrd, '__version__', '2')
            # xlrd >= 2.0 removed xls support
            if ver and str(ver).split('.')[0] >= '2':
                return jsonify({'error': 'Legacy .xls files are not supported by current environment. Please save the file as .xlsx and try again.'}), 400
        except Exception:
            return jsonify({'error': 'Legacy .xls files may not be supported. Please upload as .xlsx.'}), 400

    try:
        upload_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{upload_id}_{filename}")
        file.save(file_path)
        print(f"Uploaded file saved to: {file_path}")

        # Validate & transform
        is_valid, validation_message = validate_excel_structure(file_path)
        if not is_valid:
            return jsonify({'error': f'Excel validation failed: {validation_message}'}), 400

        try:
            json_data = transform_excel_to_json(file_path)
        except RuntimeError as exc:
            return jsonify({'error': f'Excel parsing error: {str(exc)}'}), 400

        # Persist the transformed input so external verifiers can load the exact dataset.
        try:
            dash_data_dir = os.path.join(os.path.dirname(__file__), 'data')
            os.makedirs(dash_data_dir, exist_ok=True)
            last_input_path = os.path.join(dash_data_dir, 'last_input_data.json')
            with open(last_input_path, 'w', encoding='utf-8') as f:
                json.dump(make_json_serializable(json_data), f, indent=2)
            print(f"[UPLOAD] Saved last_input_data.json -> {last_input_path}")
        except Exception as save_err:
            print(f"[UPLOAD] Warning: could not save last_input_data.json: {save_err}")

        try:
            input_data = initialize_input_data_from_json(json_data)
        except Exception as exc:
            return jsonify({'error': f'Failed to initialize input data: {str(exc)}'}), 400

        # Store input_data and metadata for later processing
        generated_timetables[upload_id] = {
            'input_data': input_data,
            'file_path': file_path,
            'filename': filename,
            'upload_time': datetime.now().isoformat(),
            'json_data': json_data
        }

        # Generate preview safely
        try:
            summary = input_data.get_data_summary()
        except Exception as e:
            print(f"Warning: Could not get data summary: {e}")
            summary = {}

        preview_data = {
            'student_groups': summary.get('student_groups', []),
            'courses': summary.get('courses', []),
            'rooms': summary.get('rooms', []),
            'faculties': summary.get('faculties', []),
            'total_student_capacity': summary.get('total_student_capacity', 0),
        }

        return jsonify({
            'success': True,
            'upload_id': upload_id,
            'filename': filename,
            'file_size': os.path.getsize(file_path),
            'preview': preview_data
        }), 200

    except Exception as exc:
        print(f"Upload error: {exc}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Failed to process Excel file: {str(exc)}'}), 500


@app.route('/generate-timetable', methods=['POST'])
def generate_timetable():
    """Kick off timetable generation in background thread."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400

    upload_id = data.get('upload_id')
    config = data.get('config', {}) or {}

    if not upload_id:
        return jsonify({'error': 'upload_id is required'}), 400
    if upload_id not in generated_timetables:
        return jsonify({'error': 'Invalid upload ID. Please upload an Excel file first.'}), 400
    
    # Thread-safe check for existing processing job
    lock = get_job_lock(upload_id)
    with lock:
        if upload_id in processing_jobs and processing_jobs[upload_id]['status'] == 'processing':
            return jsonify({'error': 'Timetable generation already in progress for this upload'}), 409

    try:
        stored = generated_timetables[upload_id]
        input_data = stored['input_data']

        # Initialize job record with thread safety and ensure all values are serializable
        job_data = {
            'status': 'processing',
            'progress': 0,
            'start_time': datetime.now().isoformat(),
            'config': make_json_serializable(config),  # Ensure config is serializable
            'error': None,
            'result': None
        }
        
        with lock:
            processing_jobs[upload_id] = job_data

        # Remove stale fresh timetable so UI won't show previous results while new run is in progress
        try:
            dash_data_dir = os.path.join(os.path.dirname(__file__), 'data')
            fresh_path = os.path.join(dash_data_dir, 'fresh_timetable_data.json')
            if os.path.exists(fresh_path):
                os.remove(fresh_path)
                print(f"[GEN] Removed stale fresh file: {fresh_path}")
        except Exception as rm_err:
            print(f"[GEN] Warning: Could not remove stale fresh file: {rm_err}")

        processor = TimetableProcessor(upload_id, input_data, config)

        # Extract config parameters with defaults
        pop_size = int(config.get('population_size', 50))
        max_gen = int(config.get('max_generations', 40))
        F = float(config.get('F', config.get('mutation_factor', 0.4)))
        CR = float(config.get('CR', config.get('crossover_rate', 0.9)))

        # Start optimization in separate thread
        thread = threading.Thread(
            target=processor.run_optimization,
            args=(upload_id, input_data, pop_size, max_gen, F, CR),
            daemon=True
        )
        thread.start()
        print(f"[GEN] Started generation thread for {upload_id} (pop={pop_size}, gens={max_gen}, F={F}, CR={CR})")

        return jsonify({
            'success': True,
            'upload_id': upload_id,
            'message': 'Timetable generation started',
            'config': config,
            'estimated_time_minutes': max_gen * 0.05
        }), 202

    except Exception as exc:
        print(f"Generation start error: {exc}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        
        # Update job status safely
        with lock:
            if upload_id in processing_jobs:
                processing_jobs[upload_id]['status'] = 'error'
                processing_jobs[upload_id]['error'] = str(exc)
        
        return jsonify({'error': f'Failed to start timetable generation: {str(exc)}'}), 500


@app.route('/get-timetable-status/<upload_id>', methods=['GET'])
def get_timetable_status(upload_id):
    if upload_id not in processing_jobs:
        return jsonify({'error': 'No processing job found for this upload ID'}), 404

    try:
        # Thread-safe read of job status
        lock = get_job_lock(upload_id)
        with lock:
            job = processing_jobs[upload_id].copy()  # Create a copy to avoid race conditions
        
        # Make sure all job data is JSON serializable
        serialized_job = make_json_serializable(job)
        
        response = {
            'upload_id': str(upload_id),
            'status': str(serialized_job.get('status', 'unknown')),
            'progress': int(serialized_job.get('progress', 0)),
            'start_time': str(serialized_job.get('start_time', '')),
        }

        if serialized_job.get('status') == 'completed':
            # Ensure result is fully serializable
            result = serialized_job.get('result', {})
            if result:
                result = make_json_serializable(result)
            
            response.update({
                'message': 'Timetable generation completed successfully',
                'result': result
            })
        elif serialized_job.get('status') == 'error':
            error_msg = str(serialized_job.get('error', 'Unknown error'))
            response.update({
                'message': f'Generation failed: {error_msg}',
                'error': error_msg
            })
        else:
            progress = int(serialized_job.get('progress', 0))
            response.update({
                'message': f'Processing... {progress}% complete'
            })

        return jsonify(response), 200
    
    except Exception as e:
        print(f"Error in get_timetable_status: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        
        # Return a safe error response
        return jsonify({
            'upload_id': str(upload_id),
            'status': 'error',
            'progress': 0,
            'error': f'Status check failed: {str(e)}',
            'message': f'Status check failed: {str(e)}'
        }), 500


@app.route('/get-timetable-status', methods=['GET'])
def list_timetable_jobs():
    try:
        summary = {}
        for uid, job in processing_jobs.items():
            summary[uid] = {
                'status': str(job.get('status')),
                'progress': int(job.get('progress', 0)),
                'has_result': bool(job.get('result') is not None)
            }
        return jsonify({
            'count': len(summary),
            'jobs': summary
        }), 200
    except Exception as e:
        return jsonify({'error': f'Failed to list jobs: {str(e)}'}), 500


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

    # Thread-safe read of job
    lock = get_job_lock(upload_id)
    with lock:
        job = processing_jobs[upload_id].copy()
    
    if job.get('status') != 'completed':
        return jsonify({'error': f'Timetable not ready for export. Status: {job.get("status")}' }), 400

    try:
        result = job.get('result')
        if not result:
            return jsonify({'error': 'No timetable data available for export'}), 500

        timetable_data = result.get('timetables_raw', [])
        if not timetable_data:
            return jsonify({'error': 'No timetable data found in results'}), 500

        if format_type == 'excel':
            excel_buffer = export_service.export_to_excel(timetable_data)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
            tmp.write(excel_buffer.getvalue())
            tmp.close()
            return send_file(
                tmp.name,
                as_attachment=True,
                download_name=f'timetable_{upload_id}.xlsx',
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        elif format_type == 'pdf':
            pdf_buffer = export_service.export_to_pdf(timetable_data)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            tmp.write(pdf_buffer.getvalue())
            tmp.close()
            return send_file(
                tmp.name,
                as_attachment=True,
                download_name=f'timetable_{upload_id}.pdf',
                mimetype='application/pdf'
            )
        else:
            return jsonify({'error': f'Unsupported format: {format_type}. Supported: excel, pdf'}), 400

    except Exception as exc:
        print(f"Export error: {exc}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Export failed: {str(exc)}'}), 500


@app.route('/api/download-template', methods=['GET'])
def download_template():
    """Download the Excel template file for timetable input"""
    try:
        template_path = os.path.join(os.path.dirname(__file__), 'data', 'NEWLY updated Timetable_Input_Template.xlsx')
        
        if not os.path.exists(template_path):
            return jsonify({'error': 'Template file not found'}), 404
        
        return send_file(
            template_path,
            as_attachment=True,
            download_name='Timetable_Input_Template.xlsx',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as exc:
        print(f"Template download error: {exc}")
        return jsonify({'error': f'Template download failed: {str(exc)}'}), 500


# ============================================================================
# NEW JSON-ONLY API ENDPOINTS FOR REACT FRONTEND
# ============================================================================

@app.route('/api/get-rooms-data', methods=['GET'])
def get_rooms_data():
    """Get rooms data for room selection modal"""
    try:
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        rooms_path = os.path.join(data_dir, 'rooms-data.json')
        
        if not os.path.exists(rooms_path):
            return jsonify({'rooms': []}), 200
            
        with open(rooms_path, 'r', encoding='utf-8') as f:
            rooms_data = json.load(f)
        
        return jsonify({'rooms': rooms_data}), 200
    except Exception as exc:
        print(f"Error loading rooms data: {exc}")
        return jsonify({'error': f'Failed to load rooms data: {str(exc)}'}), 500


@app.route('/api/get-constraint-violations/<upload_id>', methods=['GET'])
def get_constraint_violations(upload_id):
    """Get detailed constraint violations for a timetable"""
    try:
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        violations_path = os.path.join(data_dir, 'constraint_violations.json')
        
        if not os.path.exists(violations_path):
            return jsonify({'violations': {}}), 200
            
        with open(violations_path, 'r', encoding='utf-8') as f:
            violations_data = json.load(f)
        
        return jsonify({'violations': violations_data}), 200
    except Exception as exc:
        print(f"Error loading constraint violations: {exc}")
        return jsonify({'error': f'Failed to load constraint violations: {str(exc)}'}), 500


@app.route('/api/save-timetable-changes', methods=['POST'])
def save_timetable_changes():
    """Save modified timetable data with manual changes"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        timetables = data.get('timetables', [])
        manual_cells = data.get('manual_cells', [])
        upload_id = data.get('upload_id')
        
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Save current state
        save_data = {
            'timetables': timetables,
            'manual_cells': manual_cells,
            'upload_id': upload_id,
            'timestamp': datetime.now().isoformat()
        }
        
        save_path = os.path.join(data_dir, 'timetable_data.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)

        # Recompute constraint violations so the frontend Errors modal stays accurate after manual edits.
        # IMPORTANT: Use the same constraint engine used during initial generation (constraints.py),
        # so the UI and verification script can share a single source of truth.
        try:
            import re
            from collections import defaultdict

            def _parse_cell(cell_value: str):
                if not cell_value:
                    return None, None
                s = str(cell_value).strip()
                if not s or s.upper() in {'BREAK', 'FREE'}:
                    return None, None
                if "\n" in s:
                    parts = [p.strip() for p in s.split("\n") if p.strip()]
                    if not parts:
                        return None, None
                    course_code = parts[0]
                    room_name = parts[1] if len(parts) > 1 else None
                    return course_code, room_name
                m_course = re.search(r"\bCourse\s*:\s*([^,]+)", s, flags=re.IGNORECASE)
                m_room = re.search(r"\bRoom\s*:\s*(.+)$", s, flags=re.IGNORECASE)
                course_code = m_course.group(1).strip() if m_course else None
                room_name = m_room.group(1).strip() if m_room else None
                return course_code, room_name

            # Load input_data: prefer in-memory per upload_id; fallback to persisted last_input_data.json
            input_data = None
            try:
                if upload_id and upload_id in generated_timetables:
                    input_data = generated_timetables.get(upload_id, {}).get('input_data')
            except Exception:
                input_data = None
            if input_data is None:
                try:
                    last_input_path = os.path.join(data_dir, 'last_input_data.json')
                    if os.path.exists(last_input_path):
                        with open(last_input_path, 'r', encoding='utf-8') as lf:
                            last_input_json = json.load(lf)
                        input_data = initialize_input_data_from_json(last_input_json)
                except Exception as load_exc:
                    print(f"Warning: could not load last_input_data.json for engine constraint recompute: {load_exc}")

            if input_data is None:
                raise RuntimeError("input_data unavailable; cannot run engine constraint recompute")

            rooms = getattr(input_data, 'rooms', []) or []
            hours_per_day = int(getattr(input_data, 'hours', 9) or 9)
            days_per_week = int(getattr(input_data, 'days', 5) or 5)
            timeslots_count = hours_per_day * days_per_week

            # room name/id -> room_idx
            room_index = {}
            for idx, r in enumerate(rooms):
                name = str(getattr(r, 'name', '') or '').strip()
                rid = str(getattr(r, 'id', '') or getattr(r, 'Id', '') or '').strip()
                if name:
                    room_index[name] = idx
                if rid:
                    room_index[rid] = idx

            cons = Constraints(input_data)

            # pool[(group_id, course_code)] = [event_id, event_id, ...]
            pool = defaultdict(list)
            for event_id, event in (cons.events_map or {}).items():
                try:
                    gid = str(getattr(event.student_group, 'id', '') or '').strip()
                    course_code = str(getattr(event, 'course_id', '') or '').strip()
                    if gid and course_code:
                        pool[(gid, course_code)].append(int(event_id))
                except Exception:
                    continue
            for k in list(pool.keys()):
                pool[k] = sorted(pool[k])

            chromosome = np.empty((len(rooms), timeslots_count), dtype=object)
            chromosome[:] = None

            # group name -> group_id (from input_data)
            group_name_to_id = {}
            for g in getattr(input_data, 'student_groups', []) or []:
                gname = str(getattr(g, 'name', '') or '').strip()
                gid = str(getattr(g, 'id', '') or '').strip()
                if gname and gid:
                    group_name_to_id[gname] = gid

            for entry in timetables or []:
                sg_obj = entry.get('student_group') or {}
                if isinstance(sg_obj, dict):
                    group_id = str(sg_obj.get('id') or '').strip()
                    group_name = str(sg_obj.get('name') or '').strip()
                else:
                    group_id = str(getattr(sg_obj, 'id', '') or '').strip()
                    group_name = str(getattr(sg_obj, 'name', '') or sg_obj or '').strip()

                if not group_id and group_name:
                    group_id = group_name_to_id.get(group_name, '')
                if not group_id:
                    continue

                rows = entry.get('timetable') or []
                for h_idx, row in enumerate(rows):
                    if not isinstance(row, list) or len(row) < 2:
                        continue
                    if h_idx >= hours_per_day:
                        continue
                    for d_idx in range(min(days_per_week, max(0, len(row) - 1))):
                        cell = row[d_idx + 1]
                        course_code, room_name = _parse_cell(cell)
                        if not course_code:
                            continue
                        course_code = str(course_code).strip()
                        room_name = str(room_name or '').strip()

                        room_idx = room_index.get(room_name)
                        if room_idx is None:
                            continue

                        timeslot_idx = (d_idx * hours_per_day) + h_idx
                        if timeslot_idx < 0 or timeslot_idx >= timeslots_count:
                            continue

                        key = (group_id, course_code)
                        if not pool.get(key):
                            continue

                        event_id = pool[key].pop(0)
                        if chromosome[room_idx, timeslot_idx] is None:
                            chromosome[room_idx, timeslot_idx] = event_id

            updated_violations = cons.get_detailed_constraint_violations(chromosome) or {}
            violations_path = os.path.join(data_dir, 'constraint_violations.json')
            with open(violations_path, 'w', encoding='utf-8') as vf:
                json.dump(make_json_serializable(updated_violations), vf, ensure_ascii=False, indent=2)
        except Exception as exc2:
            print(f"Warning: could not recompute constraint violations after save (engine mode): {exc2}")
            # Fall back to simplified Dash/UI recompute if available.
            try:
                rooms_path = os.path.join(data_dir, 'rooms-data.json')
                rooms_data = []
                if os.path.exists(rooms_path):
                    with open(rooms_path, 'r', encoding='utf-8') as rf:
                        rooms_data = json.load(rf) or []

                from Dash_UI import recompute_constraint_violations_simplified

                updated_violations = recompute_constraint_violations_simplified(timetables, rooms_data) or {}
                violations_path = os.path.join(data_dir, 'constraint_violations.json')
                with open(violations_path, 'w', encoding='utf-8') as vf:
                    json.dump(updated_violations, vf, ensure_ascii=False, indent=2)
            except Exception as exc3:
                print(f"Warning: could not recompute constraint violations after save (fallback dash mode): {exc3}")
        
        print(f"Saved timetable changes for upload {upload_id}")
        return jsonify({'status': 'success', 'message': 'Timetable saved successfully'}), 200
        
    except Exception as exc:
        print(f"Error saving timetable changes: {exc}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Failed to save timetable: {str(exc)}'}), 500


@app.route('/api/get-saved-timetable/<upload_id>', methods=['GET'])
def get_saved_timetable(upload_id):
    """Get previously saved timetable with manual changes"""
    try:
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        save_path = os.path.join(data_dir, 'timetable_data.json')
        
        if not os.path.exists(save_path):
            return jsonify({'timetables': None, 'manual_cells': []}), 200
            
        with open(save_path, 'r', encoding='utf-8') as f:
            save_data = json.load(f)
        
        # Check if this is the correct upload_id
        if save_data.get('upload_id') != upload_id:
            return jsonify({'timetables': None, 'manual_cells': []}), 200
        
        return jsonify({
            'timetables': save_data.get('timetables', []),
            'manual_cells': save_data.get('manual_cells', []),
            'timestamp': save_data.get('timestamp')
        }), 200
        
    except Exception as exc:
        print(f"Error loading saved timetable: {exc}")
        return jsonify({'error': f'Failed to load saved timetable: {str(exc)}'}), 500


@app.route('/', methods=['GET'])
def index():
    return jsonify({'status': 'ok', 'message': 'Timetable Generator API is running.'}), 200

# ============================================================================
# DASH UI ROUTING - DISABLED (Frontend now handles all UI)
# ============================================================================
# The Dash UI has been replaced by the React frontend
# All timetable rendering, drag-drop, and interactions now happen in React
# Backend only provides JSON data through API endpoints

# # Ensure Dash knows it is mounted under /interactive so it generates correct asset URLs
# # IMPORTANT: Since we mount Dash via DispatcherMiddleware at '/interactive', do NOT set
# # requests_pathname_prefix or routes_pathname_prefix here. Dash will respect SCRIPT_NAME
# # from the WSGI mount and generate correct asset URLs automatically.
# try:
#     if hasattr(dash_app, 'config'):
#         dash_app.config.suppress_callback_exceptions = True
# except Exception as e:
#     print(f"Warning: Could not adjust Dash config: {e}")

# # Mount Dash UI under /interactive using DispatcherMiddleware (most reliable across Dash versions)
# from werkzeug.middleware.dispatcher import DispatcherMiddleware
# from flask import redirect

# # Redirect shims in case any absolute URLs get generated without the prefix
# @app.route('/_dash-component-suites/<path:path>')
# def dash_bundle_redirect(path):
#     return redirect(f'/interactive/_dash-component-suites/{path}', code=302)

# @app.route('/_dash-layout')
# def dash_layout_redirect():
#     return redirect('/interactive/_dash-layout', code=302)

# @app.route('/_dash-dependencies')
# def dash_deps_redirect():
#     return redirect('/interactive/_dash-dependencies', code=302)

# @app.route('/_dash-update-component', methods=['POST'])
# def dash_update_redirect():
#     # 307 preserves method & body
#     return redirect('/interactive/_dash-update-component', code=307)

# @app.route('/_favicon.ico')
# def dash_favicon_redirect():
#     return redirect('/interactive/_favicon.ico', code=302)

# @app.route('/assets/<path:path>')
# def dash_assets_redirect(path):
#     return redirect(f'/interactive/assets/{path}', code=302)

# # Convenience redirect to ensure trailing slash
# @app.route('/interactive')
# def dash_trailing_redirect():
#     return redirect('/interactive/', code=302)

# # Unconditionally mount the Dash server under /interactive
# try:
#     app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {'/interactive': dash_app.server})
#     print("Dash app mounted via DispatcherMiddleware at /interactive/")
# except Exception as e:
#     print(f"Warning: Failed to mount Dash app via DispatcherMiddleware: {e}")

# # If something still fails later in startup, provide a simple fallback page at /interactive
# @app.route('/interactive/', defaults={'path': ''})
# @app.route('/interactive/<path:path>')
# def interactive_fallback(path):
#     try:
#         # If the request is for dash internal endpoints, let the mounted app handle them (no fallback)
#         if path.startswith('_dash') or path.startswith('assets'):
#             return ('', 404)
#         data_dir = os.path.join(os.path.dirname(__file__), 'data')
#         fresh_path = os.path.join(data_dir, 'fresh_timetable_data.json')
#         saved_path = os.path.join(data_dir, 'timetable_data.json')
#         content = None
#         title = None
#         if os.path.exists(fresh_path):
#             try:
#                 with open(fresh_path, 'r', encoding='utf-8') as f:
#                     content = f.read()
#                     title = 'Latest fresh_timetable_data.json'
#             except Exception as read_err:
#                 content = f"Error reading fresh_timetable_data.json: {read_err}"
#                 title = 'fresh_timetable_data.json (error)'
#         if content is None and os.path.exists(saved_path):
#             try:
#                 with open(saved_path, 'r', encoding='utf-8') as f:
#                     content = f.read()
#                     title = 'Latest timetable_data.json'
#             except Exception as read_err:
#                 content = f"Error reading timetable_data.json: {read_err}"
#                 title = 'timetable_data.json (error)'
#         if content is None:
#             content = 'No fresh or saved timetable JSON found. Please generate a timetable using the API.'
#             title = 'No timetable JSON found'
#         html = f"""
#         <!doctype html>
#         <html>
#           <head>
#             <meta charset='utf-8'/>
#             <title>Interactive Timetable (Fallback)</title>
#             <style>body{{font-family:Arial,Helvetica,sans-serif;padding:20px}} pre{{white-space:pre-wrap;background:#f6f8fa;padding:12px;border-radius:6px;border:1px solid #e1e4e8}}</style>
#           </head>
#           <body>
#             <h2>Interactive Timetable (Fallback)</h2>
#             <p>The Dash UI may still be initializing. If this page persists, check backend logs.</p>
#             <h3>{title}</h3>
#             <pre>{content}</pre>
#           </body>
#         </html>
#         """
#         return html, 200
#     except Exception as e:
#         return jsonify({'error': f'Fallback page error: {str(e)}'}), 500

# ----------------------------------------------------
#  New Export Endpoints (SST / TYD / Lecturer)
# ----------------------------------------------------
import output_data
import io


def _load_saved_timetables_if_available(upload_id: str):
    """Prefer the latest saved timetable edits for this upload_id (manual changes)."""
    try:
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        save_path = os.path.join(data_dir, 'timetable_data.json')
        if not os.path.exists(save_path):
            return None

        with open(save_path, 'r', encoding='utf-8') as f:
            save_data = json.load(f)

        if str(save_data.get('upload_id')) != str(upload_id):
            return None

        timetables = save_data.get('timetables')
        if isinstance(timetables, list) and len(timetables) > 0:
            return timetables
    except Exception as exc:
        print(f"Warning: could not load saved timetables for export: {exc}")
    return None


def _get_export_timetables(upload_id: str):
    """Return timetables for export, preferring saved edits over original DE output."""
    saved = _load_saved_timetables_if_available(upload_id)
    if saved is not None:
        return saved

    job = processing_jobs.get(upload_id, {})
    result = job.get('result', {}) if isinstance(job, dict) else {}
    # Prefer timetables_raw if present, else timetables.
    timetables_raw = result.get('timetables_raw')
    if timetables_raw:
        return timetables_raw
    timetables = result.get('timetables')
    if timetables:
        return timetables
    return None

@app.route('/api/export/sst/<upload_id>', methods=['GET'])
def export_sst_timetables(upload_id):
    """Export SST timetables for a specific upload ID."""
    if upload_id not in processing_jobs:
        return jsonify({'error': 'Invalid upload ID'}), 404
        
    job = processing_jobs[upload_id]
    if job.get('status') != 'completed':
        return jsonify({'error': 'Timetable processing not complete'}), 400
        
    try:
        export_data = _get_export_timetables(upload_id)
        if not export_data:
            return jsonify({'error': 'No timetable data found'}), 500

        excel_bytes, filename = output_data.export_sst_timetables_bytes_from_data(export_data)
        # Defensive: some code paths may return a nested tuple
        if isinstance(excel_bytes, tuple) and len(excel_bytes) == 2:
            excel_bytes, nested_name = excel_bytes
            if not filename:
                filename = nested_name
        if not excel_bytes:
            return jsonify({'error': filename or 'Export failed'}), 500
        
        return send_file(
            io.BytesIO(excel_bytes),
            as_attachment=True,
            download_name=filename or f'SST_Timetables_{upload_id}.xlsx',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        print(f"Error exporting SST: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/tyd/<upload_id>', methods=['GET'])
def export_tyd_timetables(upload_id):
    """Export TYD timetables for a specific upload ID."""
    if upload_id not in processing_jobs:
        return jsonify({'error': 'Invalid upload ID'}), 404
        
    job = processing_jobs[upload_id]
    if job.get('status') != 'completed':
        return jsonify({'error': 'Timetable processing not complete'}), 400
        
    try:
        export_data = _get_export_timetables(upload_id)
        if not export_data:
            return jsonify({'error': 'No timetable data found'}), 500

        excel_bytes, filename = output_data.export_tyd_timetables_bytes_from_data(export_data)
        # Defensive: some code paths may return a nested tuple
        if isinstance(excel_bytes, tuple) and len(excel_bytes) == 2:
            excel_bytes, nested_name = excel_bytes
            if not filename:
                filename = nested_name
        if not excel_bytes:
            return jsonify({'error': filename or 'Export failed'}), 500
        
        return send_file(
            io.BytesIO(excel_bytes),
            as_attachment=True,
            download_name=filename or f'TYD_Timetables_{upload_id}.xlsx',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        print(f"Error exporting TYD: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/lecturer/<upload_id>', methods=['GET'])
def export_lecturer_timetables(upload_id):
    """Export Lecturer timetables for a specific upload ID."""
    if upload_id not in processing_jobs:
        return jsonify({'error': 'Invalid upload ID'}), 404
        
    job = processing_jobs[upload_id]
    if job.get('status') != 'completed':
        return jsonify({'error': 'Timetable processing not complete'}), 400
        
    try:
        export_data = _get_export_timetables(upload_id)
        if not export_data:
            return jsonify({'error': 'No timetable data found'}), 500

        excel_bytes, filename = output_data.export_lecturer_timetables_bytes_from_data(export_data)
        # Defensive: some code paths may return a nested tuple
        if isinstance(excel_bytes, tuple) and len(excel_bytes) == 2:
            excel_bytes, nested_name = excel_bytes
            if not filename:
                filename = nested_name
        if not excel_bytes:
            return jsonify({'error': filename or 'Export failed'}), 500
        
        return send_file(
            io.BytesIO(excel_bytes),
            as_attachment=True,
            download_name=filename or f'Lecturer_Timetables_{upload_id}.xlsx',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        print(f"Error exporting Lecturer: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

print("✓ Dash UI routing disabled - Frontend now handles all timetable UI")

if __name__ == '__main__':
    print("Starting Timetable Generator API...")
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.environ.get('PORT', 7860))
    print(f"Running on port {port}, debug={debug_mode}")
    app.run(debug=debug_mode, host='0.0.0.0', port=port)