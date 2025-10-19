#!/usr/bin/env python3
import os
import uuid
import tempfile
import threading
import numpy as np
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
import json

# Your project imports (must exist in repo)
from transformer_api import transform_excel_to_json, validate_excel_structure
from input_data_api import initialize_input_data_from_json
from differential_evolution_api_2 import DifferentialEvolutionEnhanced as DifferentialEvolution
from export_service import create_export_service, TimetableExportService
from data_converter import TimetableDataConverter

# After creating the Flask app
app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

# Configure CORS to allow your Vercel frontend
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://pau-timetable-scheduler.vercel.app",  # Your Vercel frontend
            "https://elameen123-pau-timetable-scheduler.hf.space",  # Your HF Space
            "http://localhost:3000",  # Local development
            "http://127.0.0.1:3000"   # Local development
        ],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"]
    }
})

# Add after_request handler for additional CORS headers
@app.after_request
def after_request(response):
    origin = request.headers.get('Origin')
    if origin in [
        'https://pau-timetable-scheduler.vercel.app',
        'https://elameen123-pau-timetable-scheduler.hf.space',
        'http://localhost:3000',
        'http://127.0.0.1:3000'
    ]:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization,Accept'
        response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
        response.headers['Access-Control-Allow-Credentials'] = 'false'
    return response

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'change-this-in-prod')

ALLOWED_EXTENSIONS = {'xlsx', 'xls'}

# Thread-safe job storage with locks
processing_jobs = {}       # upload_id -> { status, progress, result, error, ... }
generated_timetables = {}  # upload_id -> stored input/metadata
job_locks = {}             # upload_id -> threading.Lock()

dash_sessions = {}

# Exporter instance
export_service = create_export_service()


# --- DEBUG HELPER FUNCTIONS ---
def debug_print(section, data, upload_id=None):
    """Enhanced debug printing with clear sections"""
    prefix = f"[{upload_id}] " if upload_id else ""
    print(f"\n{prefix}{'='*80}")
    print(f"{prefix}DEBUG: {section}")
    print(f"{prefix}{'='*80}")
    
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list):
                print(f"{prefix}{key}: [{len(value)} items]")
                if value:  # Show first item as sample
                    sample = value[0]
                    if hasattr(sample, '__dict__'):
                        print(f"{prefix}  Sample: {type(sample).__name__} object")
                        for attr in ['id', 'name', 'code', 'title']:
                            if hasattr(sample, attr):
                                print(f"{prefix}    {attr}: {getattr(sample, attr)}")
                    else:
                        print(f"{prefix}  Sample: {sample}")
            elif hasattr(value, '__dict__'):
                print(f"{prefix}{key}: {type(value).__name__} object")
                for attr in ['id', 'name', 'code', 'title']:
                    if hasattr(value, attr):
                        print(f"{prefix}  {attr}: {getattr(value, attr)}")
            else:
                print(f"{prefix}{key}: {value}")
    elif isinstance(data, list):
        print(f"{prefix}List with {len(data)} items:")
        for i, item in enumerate(data[:3]):  # Show first 3 items
            if hasattr(item, '__dict__'):
                print(f"{prefix}  [{i}]: {type(item).__name__} object")
                for attr in ['id', 'name', 'code', 'title']:
                    if hasattr(item, attr):
                        print(f"{prefix}    {attr}: {getattr(item, attr)}")
            else:
                print(f"{prefix}  [{i}]: {item}")
        if len(data) > 3:
            print(f"{prefix}  ... and {len(data) - 3} more items")
    else:
        print(f"{prefix}{data}")
    print(f"{prefix}{'='*80}\n")


def debug_input_data(input_data, upload_id):
    """Debug the input_data object in detail"""
    debug_print("INPUT DATA ANALYSIS", {
        "courses_count": len(input_data.courses),
        "rooms_count": len(input_data.rooms),
        "student_groups_count": len(input_data.student_groups),
        "faculties_count": len(input_data.faculties),
        "hours": input_data.hours,
        "days": input_data.days
    }, upload_id)
    
    # Debug courses
    if input_data.courses:
        print(f"[{upload_id}] COURSES DETAIL:")
        for i, course in enumerate(input_data.courses):
            print(f"[{upload_id}]   Course {i}: {course.name} ({course.code})")
            print(f"[{upload_id}]     Credits: {course.credits}")
            print(f"[{upload_id}]     Faculty: {course.facultyId}")
            print(f"[{upload_id}]     Required Room: {course.required_room_type}")
            if i >= 2:  # Limit output
                print(f"[{upload_id}]   ... and {len(input_data.courses) - 3} more courses")
                break
    
    # Debug student groups
    if input_data.student_groups:
        print(f"[{upload_id}] STUDENT GROUPS DETAIL:")
        for i, group in enumerate(input_data.student_groups):
            print(f"[{upload_id}]   Group {i}: {group.name} ({group.id})")
            print(f"[{upload_id}]     Students: {group.no_students}")
            print(f"[{upload_id}]     Courses: {group.courseIDs}")
            print(f"[{upload_id}]     Teachers: {group.teacherIDS}")
            print(f"[{upload_id}]     Hours Required: {group.hours_required}")
            print(f"[{upload_id}]     No Courses: {group.no_courses}")
    
    # Debug rooms
    if input_data.rooms:
        print(f"[{upload_id}] ROOMS DETAIL:")
        for i, room in enumerate(input_data.rooms):
            print(f"[{upload_id}]   Room {i}: {room.name} ({room.Id})")
            print(f"[{upload_id}]     Capacity: {room.capacity}")
            print(f"[{upload_id}]     Type: {room.room_type}")
            print(f"[{upload_id}]     Building: {room.building}")
            if i >= 2:  # Limit output
                print(f"[{upload_id}]   ... and {len(input_data.rooms) - 3} more rooms")
                break
    
    # Debug faculties
    if input_data.faculties:
        print(f"[{upload_id}] FACULTIES DETAIL:")
        for i, faculty in enumerate(input_data.faculties):
            print(f"[{upload_id}]   Faculty {i}: {faculty.name} ({faculty.faculty_id})")
            print(f"[{upload_id}]     Department: {faculty.department}")
            print(f"[{upload_id}]     Course: {faculty.courseID}")
            if i >= 2:  # Limit output
                print(f"[{upload_id}]   ... and {len(input_data.faculties) - 3} more faculties")
                break


def debug_de_algorithm(de_algorithm, upload_id):
    """Debug the DE algorithm setup"""
    debug_print("DE ALGORITHM SETUP", {
        "population_size": de_algorithm.pop_size,
        "F": de_algorithm.F,
        "CR": de_algorithm.CR,
        "events_count": len(de_algorithm.events_list),
        "events_map_size": len(de_algorithm.events_map),
        "rooms_count": len(de_algorithm.rooms),
        "timeslots_count": len(de_algorithm.timeslots),
        "student_groups_count": len(de_algorithm.student_groups)
    }, upload_id)
    
    # Debug events
    if de_algorithm.events_list:
        print(f"[{upload_id}] EVENTS DETAIL (first 10):")
        for i, event in enumerate(de_algorithm.events_list[:10]):
            print(f"[{upload_id}]   Event {i}: Group={event.student_group.name}, Course={event.course_id}, Faculty={event.faculty_id}")
    
    # Debug population
    if hasattr(de_algorithm, 'population') and de_algorithm.population is not None:
        print(f"[{upload_id}] POPULATION: Shape = {de_algorithm.population.shape}")
        print(f"[{upload_id}] POPULATION: Sample chromosome shape = {de_algorithm.population[0].shape}")
        print(f"[{upload_id}] POPULATION: Sample values = {de_algorithm.population[0][:3, :3]}")


def debug_solution(solution, de_algorithm, upload_id):
    """Debug the solution from DE algorithm"""
    if solution is None:
        print(f"[{upload_id}] WARNING: Solution is None!")
        return
        
    debug_print("DE SOLUTION ANALYSIS", {
        "solution_shape": solution.shape if hasattr(solution, 'shape') else f"Type: {type(solution)}",
        "solution_type": str(type(solution))
    }, upload_id)
    
    # Count non-None events
    scheduled_count = 0
    total_slots = 0
    
    if isinstance(solution, np.ndarray):
        print(f"[{upload_id}] SOLUTION MATRIX ANALYSIS:")
        for room_idx in range(min(3, len(solution))):  # Show first 3 rooms
            room = de_algorithm.rooms[room_idx]
            room_name = getattr(room, 'name', f'Room{room_idx}')
            scheduled_in_room = np.count_nonzero(solution[room_idx] != None)
            total_in_room = len(solution[room_idx])
            print(f"[{upload_id}]   {room_name}: {scheduled_in_room}/{total_in_room} slots filled")
            
            # Show first few scheduled events
            for slot_idx, event_id in enumerate(solution[room_idx][:8]):
                if event_id is not None:
                    if event_id in de_algorithm.events_map:
                        event = de_algorithm.events_map[event_id]
                        print(f"[{upload_id}]     Slot {slot_idx}: {event.student_group.name} - {event.course_id}")
                    else:
                        print(f"[{upload_id}]     Slot {slot_idx}: Event ID {event_id} not found in events_map")
        
        scheduled_count = np.count_nonzero(solution != None)
        total_slots = solution.size
    
    print(f"[{upload_id}] TOTAL SCHEDULED: {scheduled_count}/{total_slots} events")


def debug_timetables_generation(all_timetables, upload_id):
    """Debug timetable generation process"""
    print(f"[{upload_id}] TIMETABLES GENERATION DEBUG:")
    print(f"[{upload_id}]   Generated {len(all_timetables)} timetables")
    
    for i, timetable in enumerate(all_timetables):
        if i >= 3:  # Limit output
            print(f"[{upload_id}]   ... and {len(all_timetables) - 3} more timetables")
            break
            
        student_group = timetable.get('student_group', {})
        timetable_grid = timetable.get('timetable', [])
        
        group_name = "Unknown"
        if hasattr(student_group, 'name'):
            group_name = student_group.name
        elif isinstance(student_group, dict):
            group_name = student_group.get('name', 'Unknown')
        
        print(f"[{upload_id}]   Timetable {i}: {group_name}")
        print(f"[{upload_id}]     Grid rows: {len(timetable_grid)}")
        
        # Check grid content
        non_empty_cells = 0
        for row in timetable_grid:
            if isinstance(row, list) and len(row) > 1:
                for cell in row[1:]:  # Skip time column
                    if cell and cell != "FREE" and "Course:" in str(cell):
                        non_empty_cells += 1
        
        print(f"[{upload_id}]     Non-empty cells: {non_empty_cells}")
        
        # Show sample row
        if timetable_grid and len(timetable_grid) > 0:
            sample_row = timetable_grid[0]
            print(f"[{upload_id}]     Sample row: {sample_row}")
    

# --- EXISTING HELPER FUNCTIONS ---
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

def debug_all_timetables(all_timetables, job_id):
    """Debug the generated timetables with detailed analysis"""
    print(f"[{job_id}] " + "="*80)
    print(f"[{job_id}] TIMETABLES GENERATION DEBUG")
    print(f"[{job_id}] " + "="*80)
    
    if not all_timetables:
        print(f"[{job_id}] ❌ No timetables generated!")
        return
    
    print(f"[{job_id}] ✅ Generated {len(all_timetables)} timetables")
    
    for i, timetable in enumerate(all_timetables):
        if i >= 5:  # Limit output to first 5 timetables
            print(f"[{job_id}]   ... and {len(all_timetables) - 5} more timetables")
            break
            
        try:
            student_group = timetable.get('student_group', {})
            timetable_grid = timetable.get('timetable', [])
            
            # Get group name safely
            group_name = "Unknown Group"
            if hasattr(student_group, 'name'):
                group_name = student_group.name
            elif isinstance(student_group, dict):
                group_name = student_group.get('name', 'Unknown Group')
            elif hasattr(student_group, 'id'):
                group_name = f"Group {student_group.id}"
            
            print(f"[{job_id}]   Timetable {i+1}: {group_name}")
            print(f"[{job_id}]     Grid rows: {len(timetable_grid)}")
            
            # Count scheduled vs free slots
            scheduled_count = 0
            free_count = 0
            total_slots = 0
            
            for row in timetable_grid:
                if isinstance(row, list) and len(row) > 1:
                    # Skip time column (first column)
                    for cell in row[1:]:
                        total_slots += 1
                        if cell and cell != "FREE" and "FREE" not in str(cell).upper():
                            scheduled_count += 1
                        else:
                            free_count += 1
            
            utilization = (scheduled_count / total_slots * 100) if total_slots > 0 else 0
            
            print(f"[{job_id}]     Scheduled slots: {scheduled_count}")
            print(f"[{job_id}]     Free slots: {free_count}")
            print(f"[{job_id}]     Total slots: {total_slots}")
            print(f"[{job_id}]     Utilization: {utilization:.1f}%")
            
            # Show sample of scheduled content
            if timetable_grid and len(timetable_grid) > 0:
                sample_row = timetable_grid[0] if len(timetable_grid) > 0 else []
                print(f"[{job_id}]     Sample row: {sample_row}")
                
                # Find and show a few scheduled classes
                scheduled_classes = []
                for row in timetable_grid[:3]:  # Check first 3 rows
                    if isinstance(row, list) and len(row) > 1:
                        for cell in row[1:]:
                            if cell and cell != "FREE" and "Course:" in str(cell):
                                scheduled_classes.append(str(cell)[:50] + "..." if len(str(cell)) > 50 else str(cell))
                                if len(scheduled_classes) >= 2:
                                    break
                    if len(scheduled_classes) >= 2:
                        break
                
                if scheduled_classes:
                    print(f"[{job_id}]     Sample classes:")
                    for j, class_info in enumerate(scheduled_classes):
                        print(f"[{job_id}]       {j+1}. {class_info}")
        
        except Exception as e:
            print(f"[{job_id}]   Timetable {i+1}: Error analyzing - {e}")
    
    # Overall statistics
    total_scheduled = 0
    total_slots = 0
    groups_with_classes = 0
    
    for timetable in all_timetables:
        try:
            timetable_grid = timetable.get('timetable', [])
            group_scheduled = 0
            group_total = 0
            
            for row in timetable_grid:
                if isinstance(row, list) and len(row) > 1:
                    for cell in row[1:]:
                        group_total += 1
                        if cell and cell != "FREE" and "FREE" not in str(cell).upper():
                            group_scheduled += 1
            
            total_scheduled += group_scheduled
            total_slots += group_total
            
            if group_scheduled > 0:
                groups_with_classes += 1
                
        except Exception as e:
            print(f"[{job_id}] Error in statistics calculation: {e}")
    
    overall_utilization = (total_scheduled / total_slots * 100) if total_slots > 0 else 0
    
    print(f"\n[{job_id}] OVERALL STATISTICS:")
    print(f"[{job_id}]   Total groups: {len(all_timetables)}")
    print(f"[{job_id}]   Groups with classes: {groups_with_classes}")
    print(f"[{job_id}]   Total scheduled slots: {total_scheduled}")
    print(f"[{job_id}]   Total available slots: {total_slots}")
    print(f"[{job_id}]   Overall utilization: {overall_utilization:.1f}%")
    
    print(f"[{job_id}] " + "="*80)
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

    # Replace the run_optimization method in your TimetableProcessor class with this version:

    def run_optimization(self, job_id, input_data, pop_size, max_gen, F, CR):
        """
        Runs the differential evolution optimization in the background WITH DEBUG LOGGING.
        Now includes proper input_data storage.
        """
        
        print(f"\n[{job_id}] {'='*100}")
        print(f"[{job_id}] STARTING OPTIMIZATION WITH DEBUG LOGGING")
        print(f"[{job_id}] {'='*100}")
        
        self.start_time = datetime.now()
        de = None
        
        # SERIALIZE INPUT DATA FIRST - This is the key fix!
        print(f"[{job_id}] 🔄 Serializing input_data for storage...")
        serialized_input_data = serialize_input_data(input_data)
        if serialized_input_data:
            print(f"[{job_id}] ✅ Input data serialized successfully")
        else:
            print(f"[{job_id}] ⚠️ Failed to serialize input_data, will use basic structure")
            serialized_input_data = {
                "courses": [],
                "rooms": [],
                "studentgroups": [],
                "faculties": [],
                "days": 5,
                "hours": 8
            }
        
        try:
            # Step 1: Debug input data
            debug_input_data(input_data, job_id)
            
            # Step 2: Initialize DE with debug
            print(f"[{job_id}] Initializing DE algorithm...")
            de = DifferentialEvolution(input_data, pop_size, F, CR)
            debug_de_algorithm(de, job_id)
            
            # The population should already be initialized in the constructor
            # but ensure it's properly set up
            if not hasattr(de, 'population') or de.population is None:
                print(f"[{job_id}] Population not initialized, creating...")
                de.population = de.initialize_population()
                debug_print("POPULATION AFTER MANUAL INIT", {
                    "shape": de.population.shape,
                    "sample": de.population[0][:3, :3]
                }, job_id)

            best_solution = None
            best_fitness = float("inf")  # Changed from -inf since we're minimizing
            fitness_history = []
            final_generation = 0

            print(f"[{job_id}] Starting evolution loop for {max_gen} generations...")

            # Main evolution loop with debug
            for gen in range(max_gen):
                try:
                    print(f"[{job_id}] Generation {gen+1}/{max_gen}")
                    
                    # Check if DE has run method
                    if hasattr(de, 'run'):
                        print(f"[{job_id}] Using DE.run() method...")
                        result = de.run(1)  # Run 1 generation at a time
                        
                        if isinstance(result, tuple) and len(result) >= 4:
                            current_best, fitness_hist, gen_completed, diversity = result
                            current_fitness = fitness_hist[-1] if fitness_hist else float("inf")
                        else:
                            print(f"[{job_id}] Unexpected result format from DE.run(): {type(result)}")
                            current_best = result if result is not None else de.population[0]
                            current_fitness = de.evaluate_fitness(current_best)
                            fitness_hist = [current_fitness]
                            gen_completed = gen + 1
                    else:
                        print(f"[{job_id}] Using step-by-step evolution...")
                        # Manual evolution step
                        for i in range(de.pop_size):
                            mutant = de.mutate(i)
                            trial = de.crossover(de.population[i], mutant)
                            de.select(i, trial)
                        
                        # Find current best
                        current_best = min(de.population, key=de.evaluate_fitness)
                        current_fitness = de.evaluate_fitness(current_best)
                        fitness_hist = [current_fitness]
                        gen_completed = gen + 1

                    # Update tracking
                    fitness_history.extend(fitness_hist)
                    final_generation = gen_completed
                    
                    if current_fitness < best_fitness:
                        best_solution = current_best.copy()
                        best_fitness = current_fitness
                        print(f"[{job_id}] 🎯 New best fitness: {best_fitness}")

                    # Update progress
                    progress = min(95, int(30 + (gen / max_gen) * 65))
                    self.update_job_progress(job_id, progress, f"Generation {gen+1}/{max_gen}, Best fitness: {best_fitness:.2f}")

                    # Early termination check
                    if best_fitness <= 0:
                        print(f"[{job_id}] 🎉 Perfect solution found! Early termination.")
                        break

                except Exception as gen_error:
                    print(f"[{job_id}] ❌ Error in generation {gen}: {gen_error}")
                    import traceback
                    print(f"[{job_id}] Generation traceback: {traceback.format_exc()}")
                    continue  # Continue to next generation

            # Final processing
            if best_solution is None:
                print(f"[{job_id}] ⚠️ No best solution found, using last population member")
                best_solution = de.population[0] if de.population is not None else None

            if best_solution is not None:
                print(f"[{job_id}] 🏆 Optimization completed!")
                print(f"[{job_id}] Final best fitness: {best_fitness}")
                print(f"[{job_id}] Generations completed: {final_generation}")

                # Generate all timetables
                print(f"[{job_id}] 📊 Generating timetables...")
                all_timetables = de.print_all_timetables(best_solution, input_data.days, input_data.hours, 9)
                
                # Debug the generated timetables
                debug_all_timetables(all_timetables, job_id)

                # Calculate constraint violations for display
                print(f"[{job_id}] 🔍 Calculating constraint violations...")
                constraint_violations = {}
                try:
                    violations = de.constraints.count_violations(best_solution)
                    constraint_violations = violations if violations else {}
                except Exception as cv_error:
                    print(f"[{job_id}] ⚠️ Error calculating violations: {cv_error}")
                    constraint_violations = {}

                # Prepare final result with REAL input data
                result = {
                    'timetables': all_timetables,
                    'timetables_raw': all_timetables,
                    'parsed_timetables': all_timetables,  # For frontend compatibility
                    'fitness_score': float(best_fitness),
                    'generations_completed': int(final_generation),
                    'fitness_history': [float(f) for f in fitness_history],
                    'constraint_violations': constraint_violations,
                    'input_data': serialized_input_data,  # ← THIS IS THE KEY FIX!
                    'summary': {
                        'total_timetables': len(all_timetables),
                        'best_fitness': float(best_fitness),
                        'generations': int(final_generation),
                        'violations': len(constraint_violations)
                    }
                }

                print(f"[{job_id}] 📦 Result prepared with {len(all_timetables)} timetables")
                print(f"[{job_id}] 💾 Input data included: {serialized_input_data is not None}")
                
                # Debug the result data
                debug_result_data(job_id, result)

                # Update job with success
                self.update_job_result(job_id, result)
                
            else:
                error_msg = "Optimization failed - no solution generated"
                print(f"[{job_id}] ❌ {error_msg}")
                self.update_job_error(job_id, error_msg)

        except Exception as e:
            error_msg = f"Optimization failed: {str(e)}"
            print(f"[{job_id}] ❌ {error_msg}")
            import traceback
            print(f"[{job_id}] Full traceback: {traceback.format_exc()}")
            self.update_job_error(job_id, error_msg)
    
    def create_basic_timetables(self, de, solution, job_id):
        """Create basic timetables manually if print_all_timetables is not available"""
        print(f"[{job_id}] Creating basic timetables manually...")
        timetables = []
        
        try:
            for student_group in de.student_groups:
                print(f"[{job_id}]   Creating timetable for {student_group.name}")
                
                # Create time slots
                days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
                hours = [f"{9+i}:00-{10+i}:00" for i in range(8)]
                
                # Create timetable grid
                grid = []
                classes_found = 0
                
                for hour_idx, hour in enumerate(hours):
                    row = [hour]  # First column is time
                    
                    for day_idx in range(5):  # 5 days
                        cell_content = "FREE"
                        
                        # Calculate timeslot index
                        timeslot_idx = hour_idx * 5 + day_idx
                        
                        # Find any class for this student group at this time
                        for room_idx in range(len(de.rooms)):
                            try:
                                if timeslot_idx < len(solution[room_idx]):
                                    event_id = solution[room_idx][timeslot_idx]
                                    
                                    if event_id is not None and event_id in de.events_map:
                                        event = de.events_map[event_id]
                                        
                                        if event.student_group.id == student_group.id:
                                            # Found a class for this student group
                                            room = de.rooms[room_idx]
                                            course = de.input_data.getCourse(event.course_id)
                                            faculty = de.input_data.getFaculty(event.faculty_id)
                                            
                                            course_name = course.code if course else event.course_id
                                            room_name = getattr(room, 'name', f"Room{room_idx}")
                                            faculty_name = faculty.name if faculty else "TBA"
                                            
                                            cell_content = f"Course: {course_name}, Room: {room_name}, Lecturer: {faculty_name}"
                                            classes_found += 1
                                            break
                            except (IndexError, AttributeError) as e:
                                continue
                        
                        row.append(cell_content)
                    grid.append(row)
                
                timetable_data = {
                    'student_group': student_group,
                    'timetable': grid
                }
                timetables.append(timetable_data)
                print(f"[{job_id}]     Generated {classes_found} classes for {student_group.name}")
                
        except Exception as e:
            print(f"[{job_id}] Error creating basic timetables: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"[{job_id}] Created {len(timetables)} basic timetables")
        return timetables

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

def debug_solution(self, solution, upload_id=None):
        """Debug and repair solution with optional upload_id parameter"""
        
        if upload_id:
            print(f"[{upload_id}] Debugging solution...")
        else:
            print("Debugging solution...")
        
        try:
            # Add your debugging logic here
            # For now, just return the solution as-is
            
            # You could add course allocation repairs here:
            # repaired_solution = self.verify_and_repair_course_allocations(solution)
            # return repaired_solution
            
            return solution
            
        except Exception as e:
            if upload_id:
                print(f"[{upload_id}] Debug solution failed: {e}")
            else:
                print(f"Debug solution failed: {e}")
            return solution  
# --- API Endpoints ---

@app.route('/', methods=['GET'])
def index():
    return jsonify({'status': 'ok', 'message': 'Timetable Generator API is running.'}), 200

@app.route('/upload-excel', methods=['POST'])
def upload_excel():
    """Upload Excel and transform into internal input_data used by DE - WITH DEBUG LOGGING."""
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

        # STEP 1: Validate Excel structure
        print(f"[{upload_id}] Starting Excel validation...")
        is_valid, validation_message = validate_excel_structure(file_path)
        if not is_valid:
            print(f"[{upload_id}] Validation failed: {validation_message}")
            return jsonify({'error': f'Excel validation failed: {validation_message}'}), 400
        print(f"[{upload_id}] Excel validation passed")

        # STEP 2: Transform Excel to JSON
        print(f"[{upload_id}] Starting Excel to JSON transformation...")
        json_data = transform_excel_to_json(file_path)
        debug_print("TRANSFORMED JSON DATA", {
            "courses_count": len(json_data.get('courses', [])),
            "rooms_count": len(json_data.get('rooms', [])),
            "student_groups_count": len(json_data.get('studentgroups', [])),
            "faculties_count": len(json_data.get('faculties', [])),
            "meta": json_data.get('_meta', {})
        }, upload_id)
        
        # Show sample data
        if json_data.get('courses'):
            print(f"[{upload_id}] Sample course: {json_data['courses'][0]}")
        if json_data.get('studentgroups'):
            print(f"[{upload_id}] Sample student group: {json_data['studentgroups'][0]}")

        # STEP 3: Initialize input data
        print(f"[{upload_id}] Initializing input data...")
        input_data = initialize_input_data_from_json(json_data)
        debug_input_data(input_data, upload_id)

        # STEP 4: Test DE initialization
        print(f"[{upload_id}] Testing DE algorithm initialization...")
        try:
            test_de = DifferentialEvolution(input_data, 10, 0.4, 0.9)  # Small population for test
            debug_de_algorithm(test_de, upload_id)
            print(f"[{upload_id}] DE algorithm test initialization successful")
        except Exception as e:
            print(f"[{upload_id}] WARNING: DE algorithm test failed: {e}")
            import traceback
            traceback.print_exc()

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
            print(f"[{upload_id}] Warning: Could not get data summary: {e}")
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
                'courses': len(input_data.courses),
                'rooms': len(input_data.rooms),
                'student_groups': len(input_data.student_groups),
                'faculties': len(input_data.faculties)
            }
        }), 200

    except Exception as exc:
        print(f"[Upload] Upload error: {exc}")
        import traceback
        print(f"[Upload] Full traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Failed to process Excel file: {str(exc)}'}), 500


@app.route('/generate-timetable', methods=['POST'])
def generate_timetable():
    """Kick off timetable generation in background thread - WITH DEBUG LOGGING."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400

    upload_id = data.get('upload_id')
    config = data.get('config', {}) or {}

    if not upload_id:
        return jsonify({'error': 'upload_id is required'}), 400
    if upload_id not in generated_timetables:
        return jsonify({'error': 'Invalid upload ID. Please upload an Excel file first.'}), 400
    
    print(f"[{upload_id}] Starting generation with config: {config}")
    
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

        processor = TimetableProcessor(upload_id, input_data, config)

        # Extract config parameters with defaults
        pop_size = int(config.get('population_size', 50))
        max_gen = int(config.get('max_generations', 40))
        F = float(config.get('F', config.get('mutation_factor', 0.4)))
        CR = float(config.get('CR', config.get('crossover_rate', 0.9)))

        print(f"[{upload_id}] Starting optimization thread with parameters:")
        print(f"[{upload_id}]   Population size: {pop_size}")
        print(f"[{upload_id}]   Max generations: {max_gen}")
        print(f"[{upload_id}]   F (mutation): {F}")
        print(f"[{upload_id}]   CR (crossover): {CR}")

        # Start optimization in separate thread
        thread = threading.Thread(
            target=processor.run_optimization,
            args=(upload_id, input_data, pop_size, max_gen, F, CR),
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

    except Exception as exc:
        print(f"[{upload_id}] Generation start error: {exc}")
        import traceback
        print(f"[{upload_id}] Full traceback: {traceback.format_exc()}")
        
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

@app.route('/timeslots', methods=['GET', 'OPTIONS'])
def get_time_slots():
    """Get available time slots - MATCHES FRONTEND"""
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        # Default time slots (9 AM to 5 PM)
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

    except Exception as e:
        return jsonify({'error': f'Failed to get time slots: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0-debug'
    }), 200
    
# app.py - Enhanced with Dash integration
import threading
import subprocess
import time
from flask import Flask, request, jsonify

class DashServerManager:
    def __init__(self):
        self.active_sessions = {}  # upload_id -> port
        self.dash_processes = {}   # upload_id -> process
        
    def find_available_port(start_port=8050, max_port=8100):
        """Find an available port for Dash server"""
        for port in range(start_port, max_port):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('127.0.0.1', port))
                    return port
            except OSError:
                continue
        raise RuntimeError("No available ports found")

    def create_dash_session_enhanced(self, upload_id, session_data):
        """
        FIXED: Create Dash session with proper error handling
        """
        try:
            print(f"[{upload_id}] Creating Dash session...")
            
            # Find available port
            port = self.find_available_port()
            print(f"[{upload_id}] Assigned port: {port}")
            
            # Create session file
            session_file = f"temp_session_{upload_id}.json"
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
            
            print(f"[{upload_id}] Session file created: {session_file}")
            
            # Create Dash launcher script
            dash_script = f"""
import sys
import os
import json
sys.path.append('.')

# Add proper error handling
try:
    print("Starting Dash server for upload {upload_id}...")
    
    # Import your dash module
    from dash_server_interactive import create_dash_app
    
    # Create and run Dash app
    app = create_dash_app('{session_file}')
    if app:
        print("✅ Dash app created successfully")
        print(f"🌐 Starting server on http://127.0.0.1:{port}")
        app.run_server(host='127.0.0.1', port={port}, debug=False)
    else:
        print("❌ Failed to create Dash app")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Dash server error: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
            
            # Write launcher script
            launcher_file = f"dash_launcher_{upload_id}.py"
            with open(launcher_file, 'w') as f:
                f.write(dash_script)
            
            # Start Dash process
            print(f"[{upload_id}] Starting Dash process...")
            process = subprocess.Popen(
                ['python', launcher_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Store session info
            self.active_sessions[upload_id] = port
            self.dash_processes[upload_id] = process
            
            # Wait a moment for server to start
            import time
            time.sleep(2)
            
            # Check if process is still running
            if process.poll() is None:
                print(f"[{upload_id}] ✅ Dash server started successfully on port {port}")
                return port
            else:
                # Process died, check error
                stdout, stderr = process.communicate()
                print(f"[{upload_id}] ❌ Dash process failed:")
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")
                raise Exception(f"Dash process failed: {stderr}")
            
        except Exception as e:
            print(f"[{upload_id}] ❌ Error creating Dash session: {e}")
            raise
    
    def cleanup_session(self, upload_id):
        """Clean up session resources"""
        try:
            if upload_id in self.dash_processes:
                process = self.dash_processes[upload_id]
                process.terminate()
                del self.dash_processes[upload_id]
            
            if upload_id in self.active_sessions:
                del self.active_sessions[upload_id]
            
            # Clean up files
            for file_pattern in [f"temp_session_{upload_id}.json", f"dash_launcher_{upload_id}.py"]:
                if os.path.exists(file_pattern):
                    os.remove(file_pattern)
                    
        except Exception as e:
            print(f"Cleanup error for {upload_id}: {e}")

# Initialize manager
dash_manager = DashServerManager()

# FIXED: Enhanced serialization function
def serialize_input_data_fixed(input_data):
    """
    FIXED serialization with proper field mapping
    """
    try:
        print("🔄 Starting FIXED input_data serialization...")
        
        # Serialize courses with CORRECT field mapping
        courses = []
        for course in input_data.courses:
            # Create student_groupsID (note the 's' and 'ID')
            student_groups_ids = []
            if hasattr(course, 'student_groups'):
                student_groups_ids = [sg.id if hasattr(sg, 'id') else str(i) 
                                    for i, sg in enumerate(course.student_groups)]
            elif hasattr(course, 'studentGroupIds'):
                student_groups_ids = course.studentGroupIds
            elif hasattr(course, 'student_groupsID'):
                student_groups_ids = course.student_groupsID
            
            # Create facultyId (correct case)
            faculty_id = None
            if hasattr(course, 'faculty'):
                faculty_id = course.faculty.id if hasattr(course.faculty, 'id') else str(course.faculty)
            elif hasattr(course, 'facultyId'):
                faculty_id = course.facultyId
            
            course_dict = {
                "id": getattr(course, 'id', len(courses)),
                "name": getattr(course, 'name', f'Course {len(courses)}'),
                "student_groupsID": student_groups_ids,  # CRITICAL: Correct field name
                "facultyId": faculty_id,  # CRITICAL: Correct field name
                "hours_per_week": getattr(course, 'hours_per_week', 3),
                "requires_lab": getattr(course, 'requires_lab', False),
                "department": getattr(course, 'department', '')
            }
            courses.append(course_dict)
        
        # Serialize other entities
        rooms = []
        for room in input_data.rooms:
            rooms.append({
                "id": getattr(room, 'id', getattr(room, 'Id', len(rooms))),
                "name": getattr(room, 'name', getattr(room, 'Name', f'Room {len(rooms)}')),
                "capacity": getattr(room, 'capacity', getattr(room, 'Capacity', 50)),
                "type": getattr(room, 'type', getattr(room, 'Type', 'classroom'))
            })
        
        studentgroups = []
        for sg in input_data.student_groups:
            studentgroups.append({
                "id": getattr(sg, 'id', len(studentgroups)),
                "name": getattr(sg, 'name', f'Group {len(studentgroups)}'),
                "size": getattr(sg, 'size', 30),
                "department": getattr(sg, 'department', ''),
                "level": getattr(sg, 'level', ''),
                "courseIDs": getattr(sg, 'courseIDs', []),
                "hours_required": getattr(sg, 'hours_required', [])
            })
        
        faculties = []
        for faculty in input_data.faculties:
            faculties.append({
                "id": getattr(faculty, 'id', len(faculties)),
                "name": getattr(faculty, 'name', f'Faculty {len(faculties)}'),
                "department": getattr(faculty, 'department', ''),
                "unavailable_times": getattr(faculty, 'unavailable_times', [])
            })
        
        serialized_data = {
            "courses": courses,
            "rooms": rooms,
            "studentgroups": studentgroups,  # Note: consistent with validation
            "faculties": faculties,
            "days": getattr(input_data, 'days', 5),
            "hours": getattr(input_data, 'hours', 8)
        }
        
        print(f"✅ FIXED serialization completed:")
        print(f"   📚 Courses: {len(courses)} (with student_groupsID and facultyId)")
        print(f"   🏢 Rooms: {len(rooms)}")
        print(f"   👥 Student groups: {len(studentgroups)}")
        print(f"   👨‍🏫 Faculties: {len(faculties)}")
        
        return serialized_data
        
    except Exception as e:
        print(f"❌ Error in FIXED serialization: {e}")
        import traceback
        print(f"🔋 Traceback:\n{traceback.format_exc()}")
        return None
# FIXED: Create dash session endpoint
@app.route('/create-dash-session', methods=['POST'])
def create_dash_session_fixed():
    """FIXED version of dash session creation"""
    try:
        data = request.get_json()
        upload_id = data.get('uploadId')
        
        if not upload_id or upload_id not in processing_jobs:
            return jsonify({'error': 'Invalid or missing upload ID'}), 400
            
        job_data = processing_jobs[upload_id]
        if job_data['status'] != 'completed':
            return jsonify({'error': 'Timetable generation not completed'}), 400
        
        # Get the result data
        result = job_data.get('result', {})
        timetables = result.get('timetables', [])
        
        if not timetables:
            return jsonify({'error': 'No timetables available'}), 400
        
        # Get and serialize input data using FIXED function
        input_data = job_data.get('input_data')
        if not input_data:
            return jsonify({'error': 'No input data available'}), 400
        
        input_data_dict = serialize_input_data_fixed(input_data)
        if not input_data_dict:
            return jsonify({'error': 'Failed to serialize input data'}), 500
        
        # Create session data using TimetableDataConverter
        try:
            session_data = TimetableDataConverter.create_session_file(
                timetables, input_data_dict, upload_id
            )
        except Exception as e:
            print(f"❌ Session data creation failed: {e}")
            return jsonify({'error': f'Session creation failed: {str(e)}'}), 500
        
        # Find available port
        port = find_available_port()
        
        # Create temporary session file
        session_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(session_data, session_file, indent=2, default=str)
        session_file.close()
        
        print(f"📄 Created session file: {session_file.name}")
        
        # Launch Dash server in separate process
        try:
            # FIXED: Correct import path
            dash_script = f"""
            import sys
            import os
            sys.path.append('.')

            # FIXED: Correct import
            from dash_server_interactive import create_dash_app

            try:
                app = create_dash_app('{session_file.name}')
                if app:
                    print(f"🚀 Starting Dash server on port {port}")
                    app.run_server(host='127.0.0.1', port={port}, debug=False)
                else:
                    print("❌ Failed to create Dash app")
            except Exception as e:
                print(f"❌ Dash server error: {{e}}")
                import traceback
                print(f"🔋 Traceback: {{traceback.format_exc()}}")
            """
            
            # Write script to temporary file
            script_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
            script_file.write(dash_script)
            script_file.close()
            
            # Launch in background
            def run_dash():
                try:
                    subprocess.run([sys.executable, script_file.name], 
                                 capture_output=True, text=True, timeout=300)
                except subprocess.TimeoutExpired:
                    print(f"⏰ Dash server for {upload_id} timed out")
                except Exception as e:
                    print(f"❌ Error running Dash server: {e}")
            
            thread = threading.Thread(target=run_dash, daemon=True)
            thread.start()
            
            # Wait a moment for server to start
            import time
            time.sleep(3)
            
            # Store session info
            dash_sessions[upload_id] = {
                'port': port,
                'session_file': session_file.name,
                'script_file': script_file.name,
                'created_at': datetime.now().isoformat()
            }
            
            dash_url = f'http://127.0.0.1:{port}'
            
            print(f"✅ Dash session created for {upload_id} on port {port}")
            
            return jsonify({
                'success': True,
                'dashUrl': dash_url,
                'sessionId': upload_id,
                'port': port
            })
            
        except Exception as e:
            print(f"❌ Error launching Dash server: {e}")
            return jsonify({'error': f'Failed to launch Dash server: {str(e)}'}), 500
        
    except Exception as e:
        print(f"❌ Error in create_dash_session: {e}")
        import traceback
        print(f"🔋 Traceback:\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500
    
def debug_result_data(upload_id, result_data):
    """Debug function to print detailed information about result data"""
    print(f"\n[{upload_id}] " + "="*80)
    print(f"[{upload_id}] DETAILED RESULT DATA ANALYSIS")
    print(f"[{upload_id}] " + "="*80)
    
    if not result_data:
        print(f"[{upload_id}] ❌ Result data is None or empty")
        return
    
    print(f"[{upload_id}] Result data type: {type(result_data)}")
    print(f"[{upload_id}] Result data keys: {list(result_data.keys()) if isinstance(result_data, dict) else 'Not a dict'}")
    
    # Check timetables
    timetables = result_data.get('timetables', result_data.get('timetables_raw', []))
    print(f"[{upload_id}] Timetables found: {len(timetables) if timetables else 0}")
    
    if timetables and len(timetables) > 0:
        sample_timetable = timetables[0]
        print(f"[{upload_id}] Sample timetable structure:")
        print(f"[{upload_id}]   Type: {type(sample_timetable)}")
        print(f"[{upload_id}]   Keys: {list(sample_timetable.keys()) if isinstance(sample_timetable, dict) else 'Not a dict'}")
        
        if isinstance(sample_timetable, dict):
            if 'student_group' in sample_timetable:
                sg = sample_timetable['student_group']
                print(f"[{upload_id}]   Student group type: {type(sg)}")
                if hasattr(sg, '__dict__'):
                    print(f"[{upload_id}]   Student group attributes: {list(sg.__dict__.keys())}")
                    print(f"[{upload_id}]   Student group name: {getattr(sg, 'name', 'No name attr')}")
                elif isinstance(sg, dict):
                    print(f"[{upload_id}]   Student group dict keys: {list(sg.keys())}")
                    print(f"[{upload_id}]   Student group name: {sg.get('name', 'No name key')}")
            
            if 'timetable' in sample_timetable:
                tt_grid = sample_timetable['timetable']
                print(f"[{upload_id}]   Timetable grid type: {type(tt_grid)}")
                print(f"[{upload_id}]   Timetable grid length: {len(tt_grid) if tt_grid else 0}")
                if tt_grid and len(tt_grid) > 0:
                    print(f"[{upload_id}]   Sample row: {tt_grid[0] if len(tt_grid) > 0 else 'Empty'}")
    
    # Check input_data
    input_data = result_data.get('input_data')
    print(f"\n[{upload_id}] Input data analysis:")
    print(f"[{upload_id}]   input_data exists: {input_data is not None}")
    print(f"[{upload_id}]   input_data type: {type(input_data)}")
    
    if input_data:
        if isinstance(input_data, dict):
            print(f"[{upload_id}]   input_data keys: {list(input_data.keys())}")
            
            # Check courses structure
            courses = input_data.get('courses', [])
            print(f"[{upload_id}]   Courses: {len(courses) if courses else 0}")
            if courses and len(courses) > 0:
                sample_course = courses[0]
                print(f"[{upload_id}]   Sample course type: {type(sample_course)}")
                if hasattr(sample_course, '__dict__'):
                    print(f"[{upload_id}]   Sample course attributes: {list(sample_course.__dict__.keys())}")
                elif isinstance(sample_course, dict):
                    print(f"[{upload_id}]   Sample course keys: {list(sample_course.keys())}")
                    
            # Check student groups
            student_groups = input_data.get('student_groups', input_data.get('studentgroups', []))
            print(f"[{upload_id}]   Student groups: {len(student_groups) if student_groups else 0}")
            if student_groups and len(student_groups) > 0:
                sample_sg = student_groups[0]
                print(f"[{upload_id}]   Sample student group type: {type(sample_sg)}")
                if hasattr(sample_sg, '__dict__'):
                    print(f"[{upload_id}]   Sample student group attributes: {list(sample_sg.__dict__.keys())}")
                elif isinstance(sample_sg, dict):
                    print(f"[{upload_id}]   Sample student group keys: {list(sample_sg.keys())}")
        else:
            print(f"[{upload_id}]   input_data has __dict__: {hasattr(input_data, '__dict__')}")
            if hasattr(input_data, '__dict__'):
                print(f"[{upload_id}]   input_data attributes: {list(input_data.__dict__.keys())}")
    
    print(f"[{upload_id}] " + "="*80)

def debug_launch_dash_data(upload_id):
    """Debug function specifically for launch_dash_data issues"""
    print(f"\n[{upload_id}] " + "="*80)
    print(f"[{upload_id}] LAUNCH DASH DATA DEBUG")
    print(f"[{upload_id}] " + "="*80)
    
    if upload_id not in processing_jobs:
        print(f"[{upload_id}] ❌ Upload ID not found in processing_jobs")
        return None
    
    job_data = processing_jobs[upload_id]
    print(f"[{upload_id}] Job status: {job_data.get('status')}")
    print(f"[{upload_id}] Job keys: {list(job_data.keys())}")
    
    result = job_data.get('result')
    if result:
        debug_result_data(upload_id, result)
    else:
        print(f"[{upload_id}] ❌ No result in job data")
    
    print(f"[{upload_id}] " + "="*80)
    return result
    
import tempfile
import threading
import socket
from datetime import datetime

def find_dash_port(start_port=8050):
    """Find an available port for Dash"""
    for port in range(start_port, start_port + 50):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    return 8050

def get_latest_completed_upload():
    """Get the most recent completed upload ID"""
    completed_jobs = []
    
    for upload_id, job_data in processing_jobs.items():
        if job_data.get('status') == 'completed' and 'result' in job_data:
            # Get start time for sorting
            start_time = job_data.get('start_time', '')
            completed_jobs.append((upload_id, start_time))
    
    if not completed_jobs:
        return None
    
    # Sort by start time (most recent first)
    completed_jobs.sort(key=lambda x: x[1], reverse=True)
    latest_upload_id = completed_jobs[0][0]
    
    print(f"🎯 Found {len(completed_jobs)} completed jobs")
    print(f"📅 Latest completed upload: {latest_upload_id}")
    
    return latest_upload_id

def create_mock_input_data():
    """Create mock input data if none exists"""
    return {
        "student_groups": [{"name": f"Group {i}", "id": i} for i in range(50)],
        "courses": [{"code": "SAMPLE", "name": "Sample Course", "credits": 3}],
        "lecturers": [{"name": "Sample Lecturer", "id": 1}],
        "rooms": [{"name": "Sample Room", "capacity": 50}],
        "time_slots": ["9:00-10:00", "10:00-11:00", "11:00-12:00", "12:00-13:00",
                      "13:00-14:00", "14:00-15:00", "15:00-16:00", "16:00-17:00"],
        "days": 5,
        "hours": 8
    }

def launch_dash_for_upload(upload_id=None):
    """
    Launch Dash interface using REAL generated data with comprehensive debugging
    """
    # Auto-detect latest upload if none specified
    if upload_id is None:
        upload_id = get_latest_completed_upload()
        if upload_id is None:
            print("❌ No completed uploads found")
            print(f"Available uploads: {list(processing_jobs.keys())}")
            return False
    
    print(f"🚀 Launching Dash interface for upload: {upload_id}")
    
    # Check if upload exists and is completed
    if upload_id not in processing_jobs:
        print(f"❌ Upload ID {upload_id} not found")
        print(f"Available uploads: {list(processing_jobs.keys())}")
        return False
    
    job_data = processing_jobs[upload_id]
    if job_data.get('status') != 'completed':
        print(f"❌ Upload {upload_id} not completed. Status: {job_data.get('status')}")
        return False
    
    result = job_data.get('result')
    if not result:
        print(f"❌ No result data found for upload {upload_id}")
        return False
    
    # DEBUG: Analyze the actual result data structure
    print("\n" + "="*80)
    print("🔍 DEBUGGING ACTUAL RESULT DATA STRUCTURE")
    print("="*80)
    debug_result_data(upload_id, result)
    
    # Get timetables from result
    timetables = result.get('timetables', result.get('timetables_raw', []))
    if not timetables:
        print(f"❌ No timetable data found for upload {upload_id}")
        return False
    
    print(f"✅ Found {len(timetables)} timetables in result")
    
    # Get the REAL input_data
    input_data_dict = result.get('input_data')
    
    print("\n" + "="*80)
    print("🔍 ANALYZING INPUT DATA")
    print("="*80)
    
    if input_data_dict:
        print("✅ Found input_data in result!")
        print(f"📊 Input data type: {type(input_data_dict)}")
        
        if isinstance(input_data_dict, dict):
            print(f"📋 Input data keys: {list(input_data_dict.keys())}")
            
            # Validate structure
            required_keys = ['courses', 'rooms', 'faculties']
            missing_keys = [key for key in required_keys if key not in input_data_dict]
            
            if missing_keys:
                print(f"⚠️ Missing required keys: {missing_keys}")
            else:
                print("✅ All required keys present")
                
                # Check courses structure in detail
                courses = input_data_dict.get('courses', [])
                print(f"📚 Courses: {len(courses)}")
                if courses and len(courses) > 0:
                    sample_course = courses[0]
                    print(f"📝 Sample course structure: {type(sample_course)}")
                    if isinstance(sample_course, dict):
                        print(f"📝 Sample course keys: {list(sample_course.keys())}")
                        
                        # Check for the specific key that was missing
                        if 'student_groupsID' in sample_course:
                            print("✅ 'student_groupsID' key found in courses")
                        else:
                            print("❌ 'student_groupsID' key MISSING in courses")
                            
                        if 'facultyId' in sample_course:
                            print("✅ 'facultyId' key found in courses")
                        else:
                            print("❌ 'facultyId' key MISSING in courses")
                
                # Check student groups
                student_groups = input_data_dict.get('studentgroups', input_data_dict.get('student_groups', []))
                print(f"👥 Student groups: {len(student_groups)} (key: {'studentgroups' if 'studentgroups' in input_data_dict else 'student_groups'})")
                
                # Check faculties
                faculties = input_data_dict.get('faculties', [])
                print(f"👨‍🏫 Faculties: {len(faculties)}")
                
                # Check rooms
                rooms = input_data_dict.get('rooms', [])
                print(f"🏢 Rooms: {len(rooms)}")
    else:
        print("❌ NO input_data found in result!")
        print("This means the generation process didn't store the input_data properly.")
        print("The serialization during generation might have failed.")
        return False
    
    print("="*80)
    
    # Prepare session data using REAL data
    session_data = {
        'timetables': timetables,
        'input_data': input_data_dict,  # Use the REAL input data
        'upload_id': upload_id
    }
    
    try:
        # Create session file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(session_data, temp_file, indent=2, default=str)
        temp_file.close()
        
        print(f"📄 Created session file: {temp_file.name}")
        
        # Import and create Dash app
        try:
            from dash_server_interactive import create_dash_app
            print("✅ Successfully imported dash_server_interactive")
        except ImportError as e:
            print(f"❌ Could not import dash_server_interactive: {e}")
            print("💡 Make sure dash_server_interactive.py is in the current directory")
            return False
        
        try:
            print("🔧 Creating Dash app with REAL data...")
            dash_app = create_dash_app(temp_file.name)
            print("✅ Successfully created Dash app with REAL data!")
        except Exception as e:
            print(f"❌ Error creating Dash app with real data: {e}")
            import traceback
            print(f"📋 Full traceback:\n{traceback.format_exc()}")
            
            # Print the exact error location
            if 'student_groupsID' in str(e):
                print("\n🔍 SPECIFIC ERROR ANALYSIS:")
                print("The error is still related to 'student_groupsID' missing.")
                print("This suggests the input_data serialization didn't work correctly.")
                print("Let's check what actually got stored...")
                
                # Try to load and inspect the session file
                try:
                    with open(temp_file.name, 'r') as f:
                        session_content = json.load(f)
                        session_input_data = session_content.get('input_data', {})
                        session_courses = session_input_data.get('courses', [])
                        if session_courses:
                            print(f"Session file course sample: {session_courses[0]}")
                        else:
                            print("No courses found in session file!")
                except Exception as session_error:
                    print(f"Could not inspect session file: {session_error}")
            
            return False
        
        # Find available port
        port = find_dash_port()
        
        print("\n" + "="*70)
        print("🎯 INTERACTIVE TIMETABLE VIEWER READY WITH REAL DATA!")
        print("="*70)
        print(f"🌐 URL: http://localhost:{port}")
        print(f"📊 Timetables: {len(session_data['timetables'])}")
        print(f"🎫 Upload ID: {upload_id}")
        print(f"⏰ Generated: {job_data.get('start_time', 'Unknown time')}")
        print(f"💾 Using REAL input data from generation process")
        print("="*70)
        print("💡 Features available:")
        print("   • Drag & drop timetable editing")
        print("   • Real-time constraint checking") 
        print("   • Student group navigation")
        print("   • Interactive conflict resolution")
        print("="*70)
        print("🚀 Starting Dash server...")
        print("   Press Ctrl+C to stop the server")
        print("="*70)
        
        # Run the Dash app (this will block until stopped)
        dash_app.run_server(host='127.0.0.1', port=port, debug=False)
        
        # Cleanup session file when done
        try:
            import os
            os.unlink(temp_file.name)
            print("🧹 Cleaned up session file")
        except:
            pass
            
        return True
        
    except Exception as e:
        print(f"❌ Unexpected error launching Dash: {e}")
        import traceback
        print(f"📋 Traceback:\n{traceback.format_exc()}")
        return False

# Flask endpoint to launch Dash interface
@app.route('/launch-dash')
@app.route('/launch-dash/<upload_id>')
def launch_dash_endpoint(upload_id=None):
    """Endpoint to launch Dash interface"""
    try:
        # This endpoint returns immediately and launches Dash in background
        def launch_in_background():
            launch_dash_for_upload(upload_id)
        
        # Start Dash in a separate thread
        dash_thread = threading.Thread(target=launch_in_background, daemon=True)
        dash_thread.start()
        
        # Find the port that will be used
        port = find_dash_port()
        
        return jsonify({
            'success': True,
            'message': f'Dash interface is starting...',
            'url': f'http://localhost:{port}',
            'upload_id': upload_id or get_latest_completed_upload()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Failed to launch Dash interface: {str(e)}'
        }), 400

# Console function for manual launching
def console_launch_dash(upload_id=None):
    """
    Console function to launch Dash interface
    Usage: console_launch_dash() or console_launch_dash("specific-upload-id")
    """
    return launch_dash_for_upload(upload_id)

# Auto-launch function (optional)
def auto_launch_latest_dash():
    """
    Auto-launch Dash for the latest completed upload
    Call this if you want automatic launching
    """
    latest_upload = get_latest_completed_upload()
    if latest_upload:
        print(f"🔍 Auto-launching Dash for latest upload: {latest_upload}")
        return launch_dash_for_upload(latest_upload)
    else:
        print("⏳ No completed uploads found for auto-launch")
        return False

if __name__ == '__main__':
    import os
    print("🚀 Starting Timetable Generator API (DEBUG VERSION)...")
    print(f"🔧 Debug mode: {os.environ.get('FLASK_DEBUG', 'False')}")
    
    # Get port first
    port = int(os.environ.get('PORT', 7860))
    print(f"🌐 Port: {port}")
    
    print("📋 Available endpoints:")
    print("  - GET  /")
    print("  - GET  /health") 
    print("  - POST /upload-excel")
    print("  - POST /generate-timetable")
    print("  - GET  /get-timetable-status/<id>")
    print("  - POST /export-timetable")
    print("  - GET  /timeslots")
    print("  - GET  /launch-dash")
    print("  - GET  /launch-dash/<upload_id>")
    print("\n🔍 DEBUG MODE: Comprehensive logging enabled")
    print("📊 This version will show detailed data flow analysis")
    
    # Check for auto-launch option
    auto_launch = os.environ.get('AUTO_LAUNCH_DASH', 'false').lower() == 'true'
    
    if auto_launch:
        print("\n🎯 AUTO-LAUNCH MODE ENABLED")
        print("Will attempt to launch Dash after Flask starts...")
        
        # Start Flask app in a separate thread
        import threading
        import time
        
        def start_flask():
            app.run(host='0.0.0.0', port=port, debug=False)
        
        flask_thread = threading.Thread(target=start_flask, daemon=True)
        flask_thread.start()
        
        # Wait for Flask to start
        print("⏳ Waiting for Flask to start...")
        time.sleep(3)
        
        # Try auto-launch
        try:
            result = auto_launch_latest_dash()
            if not result:
                print("💡 To launch Dash later, run: python launch_dash.py")
                print("💡 Or visit: htt               p://localhost:7860/launch-dash")
        except Exception as e:
            print(f"❌ Auto-launch failed: {e}")
            print("💡 You can manually launch with: python launch_dash.py")
        
        # Keep the main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n👋 Application stopped by user")
    
    else:
        print("\n💡 MANUAL LAUNCH MODE")
        print("After generating timetables, launch Dash with:")
        print("  • python launch_dash.py")
        print("  • http://localhost:7860/launch-dash")
        print("=" * 60)
        
        # Standard Flask run
        app.run(host='0.0.0.0', port=port, debug=False)