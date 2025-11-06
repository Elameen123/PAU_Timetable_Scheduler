# differential_evolution_api.py
"""
API-compatible differential evolution algorithm that works with input_data instances
passed from the Flask application instead of static imports.
"""

import random
from typing import List
import copy
from utils import Utility
from entitities.Class import Class
import numpy as np
from constraints import Constraints
import re

class DifferentialEvolution:
    def __init__(self, input_data, pop_size: int, F: float, CR: float):
        self.desired_fitness = 0
        self.input_data = input_data
        self.rooms = input_data.rooms
        self.timeslots = input_data.create_time_slots(
            no_hours_per_day=input_data.hours, 
            no_days_per_week=input_data.days, 
            day_start_time=9
        )
        self.student_groups = input_data.student_groups
        self.courses = input_data.courses
        self.events_list, self.events_map = self.create_events()
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.constraints = Constraints(input_data)
        
        # Optimization: Cache fitness values to avoid recalculation
        self.fitness_cache = {}
        
        # Optimization: Pre-calculate building assignments for rooms to speed up evaluation
        self.room_building_cache = {}
        for idx, room in enumerate(self.rooms):
            self.room_building_cache[idx] = self.get_room_building(room)
        
        # Optimization: Pre-calculate engineering groups to avoid repeated checks
        self.engineering_groups = set()
        for student_group in self.student_groups:
            group_name = student_group.name.lower()
            if any(keyword in group_name for keyword in [
                'engineering', 'eng', 'computer science', 'software engineering',
                'mechatronics', 'electrical', 'mechanical', 'csc', 'sen'
            ]):
                self.engineering_groups.add(student_group.id)
        
        self.population = self.initialize_population()

    def create_events(self):
        """Create events list and mapping for the timetabling problem"""
        events_list = []
        event_map = {}

        idx = 0
        for student_group in self.student_groups:
            for i in range(student_group.no_courses):
                hourcount = 1 
                while hourcount <= student_group.hours_required[i]:
                    event = Class(student_group, student_group.teacherIDS[i], student_group.courseIDs[i])
                    events_list.append(event)
                    event_map[idx] = event
                    idx += 1
                    hourcount += 1
                    
        return events_list, event_map

    def initialize_population(self):
        """Initialize the population with valid chromosomes"""
        population = [] 
        for i in range(self.pop_size):
            chromosome = self.create_chromosome()
            population.append(chromosome)
        return np.array(population)

    def create_chromosome(self):
        """Create a single chromosome (timetable solution)"""
        chromosome = np.empty((len(self.rooms), len(self.timeslots)), dtype=object)
        
        # Group events by student group and course to handle them as blocks
        events_by_group_course = {}
        for idx, event in enumerate(self.events_list):
            key = (event.student_group.id, event.course_id)
            if key not in events_by_group_course:
                events_by_group_course[key] = []
            events_by_group_course[key].append(idx)

        # Trackers for optimized placement
        hours_per_day_for_group = {sg.id: [0] * self.input_data.days for sg in self.student_groups}
        non_sst_course_count_for_group = {sg.id: 0 for sg in self.student_groups}
        course_days_used = {}

        # Randomize course processing order for population diversity
        course_items = list(events_by_group_course.items())
        random.shuffle(course_items)

        for (student_group_id, course_id), event_indices in course_items:
            course = self.input_data.getCourse(course_id)
            student_group = self.input_data.getStudentGroup(student_group_id)
            hours_required = len(event_indices)

            if hours_required == 0:
                continue

            # Decide on a split strategy based on course credits
            if hours_required == 3:
                split_strategy = random.choice([(3,), (2, 1), (3,)])
            elif hours_required == 2:
                split_strategy = (2,)
            else:
                split_strategy = (hours_required,)

            event_idx_counter = 0
            course_key = (student_group_id, course_id)
            course_days_used[course_key] = set()
            is_course_placed_in_non_sst = False

            for block_hours in split_strategy:
                placed = False
                block_event_indices = event_indices[event_idx_counter : event_idx_counter + block_hours]
                event_idx_counter += block_hours

                # Prioritize days with fewer hours and those not yet used by this course
                available_days = [d for d in range(self.input_data.days) if d not in course_days_used[course_key]]
                sorted_days = sorted(available_days, key=lambda d: hours_per_day_for_group[student_group_id][d])

                for day_idx in sorted_days:
                    day_start = day_idx * self.input_data.hours
                    day_end = (day_idx + 1) * self.input_data.hours
                    
                    possible_slots = []
                    for room_idx, room in enumerate(self.rooms):
                        if self.is_room_suitable(room, course):
                            room_building = self.room_building_cache[room_idx]
                            is_engineering = student_group.id in self.engineering_groups
                            needs_computer_lab = (
                                course.required_room_type.lower() in ['comp lab', 'computer_lab'] or
                                room.room_type.lower() in ['comp lab', 'computer_lab'] or
                                ('lab' in course.name.lower() and any(k in course.name.lower() for k in ['computer', 'programming', 'software']))
                            )

                            building_allowed = True
                            if needs_computer_lab:
                                pass  # Computer labs can be in any building
                            elif is_engineering:
                                if room_building != 'SST' and non_sst_course_count_for_group[student_group_id] >= 2:
                                    building_allowed = False
                            elif room_building == 'SST':
                                building_allowed = False
                            
                            if building_allowed:
                                # Find consecutive slots
                                for timeslot_start in range(day_start, day_end):
                                    if timeslot_start + block_hours > day_end:
                                        continue
                                    
                                    if all(self.is_slot_available_for_event(chromosome, room_idx, timeslot_start + i, self.events_list[block_event_indices[i]]) for i in range(block_hours)):
                                        possible_slots.append((room_idx, timeslot_start))
                    
                    if possible_slots:
                        room_idx, timeslot_start = random.choice(possible_slots)
                        for i in range(block_hours):
                            chromosome[room_idx, timeslot_start + i] = block_event_indices[i]
                        
                        # Update trackers
                        hours_per_day_for_group[student_group_id][day_idx] += block_hours
                        course_days_used[course_key].add(day_idx)
                        
                        if not is_course_placed_in_non_sst and self.room_building_cache[room_idx] != 'SST' and not needs_computer_lab and is_engineering:
                            non_sst_course_count_for_group[student_group_id] += 1
                            is_course_placed_in_non_sst = True
                        
                        placed = True
                        break
                
                if placed:
                    break

        # Final verification to place any unassigned events
        chromosome = self.verify_and_repair_course_allocations(chromosome)
        return chromosome

    def is_slot_available(self, chromosome, room_idx, timeslot_idx):
        """Check if a timeslot is available"""
        if chromosome[room_idx][timeslot_idx] is not None:
            return False
        
        # Check if this is break time (13:00 - 14:00)
        break_hour = 4  # 13:00 is the 5th hour (index 4) starting from 9:00
        day = timeslot_idx // self.input_data.hours
        hour_in_day = timeslot_idx % self.input_data.hours
        
        # No break time on Tuesday (1) and Thursday (3)
        if hour_in_day == break_hour and day not in [1, 3]:
            return False
        
        return True

    def is_slot_available_for_event(self, chromosome, room_idx, timeslot_idx, event):
        """Check if a slot is available for a specific event, considering lecturer availability"""
        # Check if the slot is physically empty
        if chromosome[room_idx][timeslot_idx] is not None:
            return False

        # Check for break time
        break_hour = 4
        day = timeslot_idx // self.input_data.hours
        hour_in_day = timeslot_idx % self.input_data.hours
        if hour_in_day == break_hour and day not in [1, 3]:
            return False

        # Check lecturer schedule constraints
        if event and event.faculty_id is not None:
            faculty = self.input_data.getFaculty(event.faculty_id)
            if faculty:
                days_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
                day_abbr = days_map.get(day)

                # Check day availability
                is_day_ok = False
                if isinstance(faculty.avail_days, str):
                    if faculty.avail_days.upper() == "ALL":
                        is_day_ok = True
                    else:
                        avail_days = [d.strip().capitalize() for d in faculty.avail_days.split(',')]
                        if day_abbr in avail_days:
                            is_day_ok = True
                elif isinstance(faculty.avail_days, list):
                    avail_days = [d.strip().capitalize() for d in faculty.avail_days]
                    if "All" in avail_days or day_abbr in avail_days:
                        is_day_ok = True
                
                if not is_day_ok:
                    return False

                # Check time availability
                if isinstance(faculty.avail_times, str) and faculty.avail_times.upper() != "ALL":
                    try:
                        start_avail_str, end_avail_str = faculty.avail_times.split('-')
                        start_avail_h = int(start_avail_str.split(':')[0])
                        end_avail_h = int(end_avail_str.split(':')[0])
                        
                        timeslot_obj = self.timeslots[timeslot_idx]
                        # TimeSlot.start_time may be 0-based hour index (int) or 'HH:MM' string; support both
                        if isinstance(getattr(timeslot_obj, 'start_time', None), (int, np.integer)):
                            slot_start_h = 9 + int(timeslot_obj.start_time)
                        else:
                            slot_start_h = int(str(timeslot_obj.start_time).split(':')[0])

                        if not (start_avail_h <= slot_start_h < end_avail_h):
                            return False
                    except (ValueError, IndexError):
                        return False

        return True

    def is_room_suitable(self, room, course):
        """Check if room is suitable for course"""
        if course is None:
            return False
        return room.room_type == course.required_room_type
    
    def get_room_building(self, room):
        """Helper method to determine room building"""
        if hasattr(room, 'building'):
            return room.building.upper()
        elif hasattr(room, 'name') and room.name:
            room_name = room.name.upper()
            if 'SST' in room_name:
                return 'SST'
            elif 'TYD' in room_name:
                return 'TYD'
        elif hasattr(room, 'room_id'):
            room_id = str(room.room_id).upper()
            if 'SST' in room_id:
                return 'SST'
            elif 'TYD' in room_id:
                return 'TYD'
        return 'UNKNOWN'

    def _is_student_group_available(self, chromosome, student_group_id, timeslot_idx):
        """Check if a student group is already scheduled at a given timeslot"""
        for r_idx in range(len(self.rooms)):
            event_id = chromosome[r_idx][timeslot_idx]
            if event_id is not None:
                event = self.events_map.get(event_id)
                if event and event.student_group.id == student_group_id:
                    return False
        return True

    def _is_lecturer_available(self, chromosome, faculty_id, timeslot_idx):
        """Check if a lecturer is already scheduled at a given timeslot"""
        for r_idx in range(len(self.rooms)):
            event_id = chromosome[r_idx][timeslot_idx]
            if event_id is not None:
                event = self.events_map.get(event_id)
                if event and event.faculty_id == faculty_id:
                    return False
        return True

    def find_clash(self, chromosome):
        """Find a random timeslot with a student or lecturer clash"""
        clash_slots = []
        for t_idx in range(len(self.timeslots)):
            simultaneous_events = chromosome[:, t_idx]
            
            student_group_watch = set()
            lecturer_watch = set()
            has_student_clash = False
            has_lecturer_clash = False
            
            event_ids_in_slot = [e for e in simultaneous_events if e is not None]
            if len(event_ids_in_slot) <= 1:
                continue

            for event_id in event_ids_in_slot:
                event = self.events_map.get(event_id)
                if not event: 
                    continue
                
                # Check for student clash
                if event.student_group.id in student_group_watch:
                    has_student_clash = True
                student_group_watch.add(event.student_group.id)
                
                # Check for lecturer clash
                if event.faculty_id and event.faculty_id in lecturer_watch:
                    has_lecturer_clash = True
                if event.faculty_id:
                    lecturer_watch.add(event.faculty_id)
            
            if has_student_clash or has_lecturer_clash:
                clash_slots.append(t_idx)
                
        if clash_slots:
            return random.choice(clash_slots)
        return None
    
    def hamming_distance(self, chromosome1, chromosome2):
        """Calculate Hamming distance between two chromosomes"""
        return np.sum(chromosome1.flatten() != chromosome2.flatten())

    def calculate_population_diversity(self):
        """Calculate population diversity using sampling for efficiency"""
        if self.pop_size <= 10:
            total_distance = 0
            comparisons = 0
            
            for i in range(self.pop_size):
                for j in range(i + 1, self.pop_size):
                    total_distance += self.hamming_distance(self.population[i], self.population[j])
                    comparisons += 1
            
            return total_distance / comparisons if comparisons > 0 else 0
        else:
            # For larger populations, sample 10 random pairs
            total_distance = 0
            comparisons = 10
            
            for _ in range(comparisons):
                i, j = random.sample(range(self.pop_size), 2)
                total_distance += self.hamming_distance(self.population[i], self.population[j])
            
            return total_distance / comparisons

    def mutate(self, target_idx):
        """Mutation operation with targeted clash resolution"""
        mutant_vector = self.population[target_idx].copy()

        # Strategy 1: Targeted Clash Resolution
        if random.random() < 0.7:
            clash_timeslot = self.find_clash(mutant_vector)
            if clash_timeslot is not None:
                events_in_clash = [mutant_vector[r][clash_timeslot] for r in range(len(self.rooms)) if mutant_vector[r][clash_timeslot] is not None]
                
                if events_in_clash:
                    event_id_to_move = random.choice(events_in_clash)
                    event_to_move = self.events_map.get(event_id_to_move)

                    if event_to_move:
                        # Find original position and remove it
                        for r_idx in range(len(self.rooms)):
                            if mutant_vector[r_idx][clash_timeslot] == event_id_to_move:
                                mutant_vector[r_idx][clash_timeslot] = None
                                break
                        
                        # Find a new, completely valid slot for this event
                        possible_slots = []
                        course = self.input_data.getCourse(event_to_move.course_id)
                        for r_idx, room in enumerate(self.rooms):
                            if self.is_room_suitable(room, course):
                                for t_idx in range(len(self.timeslots)):
                                    if (self.is_slot_available_for_event(mutant_vector, r_idx, t_idx, event_to_move) and
                                        self._is_student_group_available(mutant_vector, event_to_move.student_group.id, t_idx)):
                                        possible_slots.append((r_idx, t_idx))
                        
                        if possible_slots:
                            r, t = random.choice(possible_slots)
                            mutant_vector[r][t] = event_id_to_move

        # Strategy 2: Perform a few swaps to introduce small variations
        if random.random() < 0.2:
            for _ in range(random.randint(1, 2)):
                occupied_slots = np.argwhere(mutant_vector != None)
                if len(occupied_slots) < 2: 
                    continue
                idx1, idx2 = random.sample(range(len(occupied_slots)), 2)
                pos1, pos2 = tuple(occupied_slots[idx1]), tuple(occupied_slots[idx2])
                mutant_vector[pos1], mutant_vector[pos2] = mutant_vector[pos2], mutant_vector[pos1]

        return mutant_vector

    def crossover(self, target_vector, mutant_vector):
        """Enhanced Strategic Crossover with conflict resolution"""
        trial_vector = target_vector.copy()
        conflicts = self.constraints.get_all_conflicts(trial_vector)
        
        # Combine all hard conflicts to be resolved
        all_clashes = conflicts.get('student_group', []) + conflicts.get('lecturer', [])
        
        if not all_clashes:
            # If no clashes, perform a more standard DE crossover
            for r in range(len(self.rooms)):
                for t in range(len(self.timeslots)):
                    if random.random() < self.CR:
                        trial_vector[r, t] = mutant_vector[r, t]
            return trial_vector

        # Create a set of positions that have clashes for quick lookup
        clash_positions = set()
        for clash in all_clashes:
            for pos in clash['positions']:
                clash_positions.add(tuple(pos))

        # Iterate through the mutant and bring in non-conflicting genes
        for r in range(len(self.rooms)):
            for t in range(len(self.timeslots)):
                if (r, t) in clash_positions:
                    mutant_gene = mutant_vector[r, t]
                    target_gene = trial_vector[r, t]

                    if mutant_gene != target_gene and mutant_gene is not None:
                        mutant_event = self.events_map.get(mutant_gene)
                        if not mutant_event: 
                            continue

                        is_safe_to_swap = True
                        # Check against all other events in the same timeslot in the trial vector
                        for r_check in range(len(self.rooms)):
                            if r_check != r:
                                existing_event_id = trial_vector[r_check, t]
                                if existing_event_id is not None:
                                    existing_event = self.events_map.get(existing_event_id)
                                    if existing_event:
                                        if (existing_event.student_group.id == mutant_event.student_group.id or
                                            existing_event.faculty_id == mutant_event.faculty_id):
                                            is_safe_to_swap = False
                                            break
                        
                        if is_safe_to_swap:
                            trial_vector[r, t] = mutant_gene
                            
        return trial_vector

    def evaluate_fitness(self, chromosome):
        """Evaluate fitness using cached results"""
        chromosome_key = str(chromosome.tobytes())
        if chromosome_key in self.fitness_cache:
            return self.fitness_cache[chromosome_key]
        
        # Use the centralized Constraints class for consistent evaluation
        fitness = self.constraints.evaluate_fitness(chromosome)
        
        # Cache management: prevent unlimited growth
        if len(self.fitness_cache) > 1000:
            keys_to_remove = list(self.fitness_cache.keys())[:-500]
            for key in keys_to_remove:
                del self.fitness_cache[key]
        
        # Cache the result
        self.fitness_cache[chromosome_key] = fitness
        return fitness

    def select(self, target_idx, trial_vector):
        """Selection operation with hard constraint prioritization"""
        trial_violations = self.constraints.get_constraint_violations(trial_vector)
        target_violations = self.constraints.get_constraint_violations(self.population[target_idx])

        # Define which constraints are "hard" and must be prioritized
        hard_constraints = [
            'student_group_constraints', 
            'lecturer_availability', 
            'course_allocation_completeness',
            'room_time_conflict',
            'break_time_constraint',
            'room_constraints',
            'same_course_same_room_per_day',
            'lecturer_schedule_constraints'
        ]

        trial_hard_violations = sum(trial_violations.get(c, 0) for c in hard_constraints)
        target_hard_violations = sum(target_violations.get(c, 0) for c in hard_constraints)

        accept = False
        if trial_hard_violations < target_hard_violations:
            accept = True
        elif trial_hard_violations == target_hard_violations:
            if trial_violations.get('total', float('inf')) <= target_violations.get('total', float('inf')):
                accept = True

        if accept:
            self.population[target_idx] = trial_vector

    def run(self, max_generations):
        """Run the differential evolution algorithm"""
        fitness_history = []
        best_solution = self.population[0]
        diversity_history = []
        
        # Calculate initial fitness and find best solution
        initial_fitness = [self.evaluate_fitness(ind) for ind in self.population]
        best_idx = np.argmin(initial_fitness)
        best_solution = self.population[best_idx].copy()
        best_fitness = initial_fitness[best_idx]
        
        # Track fitness for early convergence detection
        stagnation_counter = 0

        for generation in range(max_generations):
            generation_improved = False
            
            for i in range(self.pop_size):
                # Step 1: Mutation
                mutant_vector = self.mutate(i)
                
                # Step 2: Crossover
                target_vector = self.population[i]
                trial_vector = self.crossover(target_vector, mutant_vector)
                
                # Step 3: Evaluation and Selection
                old_fitness = self.evaluate_fitness(self.population[i])
                self.select(i, trial_vector)
                new_fitness = self.evaluate_fitness(self.population[i])
                
                # Ensure population member has all events after selection
                self.population[i] = self.verify_and_repair_course_allocations(self.population[i])
                
                if new_fitness < old_fitness:
                    generation_improved = True
                
            # Find best solution more efficiently
            current_fitness = [self.evaluate_fitness(ind) for ind in self.population]
            current_best_idx = np.argmin(current_fitness)
            current_best_fitness = current_fitness[current_best_idx]
            
            if current_best_fitness < best_fitness:
                best_solution = self.population[current_best_idx].copy()
                best_fitness = current_best_fitness
                stagnation_counter = 0
                
                # Ensure best solution has all courses properly allocated
                best_solution = self.verify_and_repair_course_allocations(best_solution)
            else:
                stagnation_counter += 1
            
            fitness_history.append(best_fitness)

            # Calculate diversity less frequently for speed
            if generation % 20 == 0:
                population_diversity = self.calculate_population_diversity()
                diversity_history.append(population_diversity)

            print(f"Best solution for generation {generation+1}/{max_generations} has a fitness of: {best_fitness}")

            if best_fitness == self.desired_fitness:
                print(f"Solution with desired fitness of {self.desired_fitness} found at Generation {generation}!")
                break
            
            # Early termination if no improvement for many generations
            if stagnation_counter > 50 and best_fitness < 100:
                print(f"Early termination due to convergence at generation {generation+1}")
                break

        # Final verification and repair of the best solution
        best_solution = self.verify_and_repair_course_allocations(best_solution)
        
        return best_solution, fitness_history, generation, diversity_history

    def print_timetable(self, individual, student_group, days, hours_per_day, day_start_time=9):
        """Print timetable for a specific student group"""
        timetable = [["" for _ in range(days)] for _ in range(hours_per_day)]
        
        # First, fill break time slots on Mon, Wed, Fri
        break_hour = 4
        if break_hour < hours_per_day:
            for day in range(days):
                if day in [0, 2, 4]:  # Monday, Wednesday, Friday
                    timetable[break_hour][day] = "BREAK"
        
        for room_idx, room_slots in enumerate(individual):
            for timeslot_idx, event in enumerate(room_slots):
                class_event = self.events_map.get(event)
                if class_event is not None and class_event.student_group.id == student_group.id:
                    day = timeslot_idx // hours_per_day
                    hour = timeslot_idx % hours_per_day
                    
                    # Check if it's a break slot that should be skipped in the display
                    is_display_break = (hour == break_hour and day in [0, 2, 4])

                    if day < days and not is_display_break:
                        course = self.input_data.getCourse(class_event.course_id)
                        faculty = self.input_data.getFaculty(class_event.faculty_id)
                        course_code = course.code if course is not None else "Unknown"
                        faculty_name = faculty.name if faculty is not None else "Unknown"
                        room_obj = self.input_data.rooms[room_idx]
                        room_display = getattr(room_obj, "name", getattr(room_obj, "Id", str(room_idx)))
                        timetable[hour][day] = f"Course: {course_code}, Lecturer: {faculty_name}, Room: {room_display}"
        return timetable

    def print_all_timetables(self, individual, days, hours_per_day, day_start_time=9):
        """Generate all timetables for the solution"""
        data = []
        student_groups = self.input_data.student_groups
        
        for student_group in student_groups:
            timetable = self.print_timetable(individual, student_group, days, hours_per_day, day_start_time)
            rows = []
            for hour in range(hours_per_day):
                time_label = f"{day_start_time + hour}:00"
                row = [time_label] + [timetable[hour][day] for day in range(days)]
                rows.append(row)
            data.append({"student_group": student_group, "timetable": rows})
        return data

    def verify_and_repair_course_allocations(self, chromosome):
        """
        Verify that all courses appear the correct number of times for each student group
        and repair any missing allocations with minimal disruption.
        """
        max_repair_passes = 3
        
        for repair_pass in range(max_repair_passes):
            scheduled_events = set()
            for room_idx in range(len(self.rooms)):
                for timeslot_idx in range(len(self.timeslots)):
                    event_id = chromosome[room_idx][timeslot_idx]
                    if event_id is not None:
                        scheduled_events.add(event_id)
            
            missing_events = [event_id for event_id in range(len(self.events_list)) if event_id not in scheduled_events]
            
            if not missing_events:
                break
            
            flexibility_level = repair_pass
            
            course_day_room_mapping = {}
            for r_idx, t_idx in np.argwhere(chromosome != None):
                event_id = chromosome[r_idx][t_idx]
                event = self.events_map.get(event_id)
                if event:
                    course = self.input_data.getCourse(event.course_id)
                    if course:
                        day_idx = t_idx // self.input_data.hours
                        course_id = getattr(course, 'course_id', None) or getattr(course, 'id', None) or getattr(course, 'code', None)
                        course_day_key = (course_id, day_idx, event.student_group.id)
                        course_day_room_mapping[course_day_key] = r_idx

            for missing_event_id in missing_events:
                event = self.events_list[missing_event_id]
                course = self.input_data.getCourse(event.course_id)
                if not course: 
                    continue
                
                course_id = getattr(course, 'course_id', None) or getattr(course, 'id', None) or getattr(course, 'code', None)
                placed = False

                # Strategy 1: Place in the same room as other instances of the same course on the same day
                if flexibility_level == 0:
                    preferred_slots = []
                    for day_idx in range(self.input_data.days):
                        course_day_key = (course_id, day_idx, event.student_group.id)
                        if course_day_key in course_day_room_mapping:
                            preferred_room = course_day_room_mapping[course_day_key]
                            day_start, day_end = day_idx * self.input_data.hours, (day_idx + 1) * self.input_data.hours
                            for timeslot in range(day_start, day_end):
                                if (self.is_slot_available_for_event(chromosome, preferred_room, timeslot, event) and
                                    self._is_student_group_available(chromosome, event.student_group.id, timeslot)):
                                    preferred_slots.append((preferred_room, timeslot))
                    if preferred_slots:
                        room_idx, timeslot_idx = random.choice(preferred_slots)
                        chromosome[room_idx][timeslot_idx] = missing_event_id
                        placed = True

                # Strategy 2: Find any valid slot that respects all hard constraints
                if not placed:
                    valid_slots = []
                    for room_idx, room in enumerate(self.rooms):
                        if self.is_room_suitable(room, course):
                            for timeslot_idx in range(len(self.timeslots)):
                                if (self.is_slot_available_for_event(chromosome, room_idx, timeslot_idx, event) and
                                    self._is_student_group_available(chromosome, event.student_group.id, timeslot_idx)):
                                    valid_slots.append((room_idx, timeslot_idx))
                    if valid_slots:
                        room_idx, timeslot_idx = random.choice(valid_slots)
                        chromosome[room_idx][timeslot_idx] = missing_event_id
                        placed = True

                # Strategy 3 (Final Pass): Only place in a valid, empty slot
                if not placed and flexibility_level >= 2:
                    valid_empty_slots = []
                    for room_idx, room in enumerate(self.rooms):
                        if self.is_room_suitable(room, course):
                            for timeslot_idx in range(len(self.timeslots)):
                                if (chromosome[room_idx][timeslot_idx] is None and
                                    self.is_slot_available_for_event(chromosome, room_idx, timeslot_idx, event) and
                                    self._is_student_group_available(chromosome, event.student_group.id, timeslot_idx)):
                                    valid_empty_slots.append((room_idx, timeslot_idx))
                    
                    if valid_empty_slots:
                        room_idx, timeslot_idx = random.choice(valid_empty_slots)
                        chromosome[room_idx][timeslot_idx] = missing_event_id
                        placed = True
                        
        return chromosome

    def count_course_occurrences(self, chromosome, student_group):
        """Count how many times each course appears for a specific student group"""
        course_counts = {}
        
        for room_idx in range(len(self.rooms)):
            for timeslot_idx in range(len(self.timeslots)):
                event_id = chromosome[room_idx][timeslot_idx]
                if event_id is not None:
                    event = self.events_map.get(event_id)
                    if event and event.student_group.id == student_group.id:
                        course_id = event.course_id
                        course_counts[course_id] = course_counts.get(course_id, 0) + 1
        
        return course_counts

    def diagnose_course_allocations(self, chromosome):
        """Diagnostic method to check course allocations for debugging"""
        print("\n=== COURSE ALLOCATION DIAGNOSIS ===")
        
        # Count total scheduled events
        scheduled_events = set()
        for room_idx in range(len(self.rooms)):
            for timeslot_idx in range(len(self.timeslots)):
                event_id = chromosome[room_idx][timeslot_idx]
                if event_id is not None:
                    scheduled_events.add(event_id)
        
        print(f"Total events: {len(self.events_list)}")
        print(f"Scheduled events: {len(scheduled_events)}")
        print(f"Missing events: {len(self.events_list) - len(scheduled_events)}")
        
        # Check each student group
        for student_group in self.student_groups:
            print(f"\nStudent Group: {student_group.name}")
            course_counts = self.count_course_occurrences(chromosome, student_group)
            
            total_expected = sum(student_group.hours_required)
            total_actual = sum(course_counts.values())
            
            print(f"  Total hours: Expected {total_expected}, Got {total_actual}")
            
            for i, course_id in enumerate(student_group.courseIDs):
                expected = student_group.hours_required[i]
                actual = course_counts.get(course_id, 0)
                status = "✓" if actual == expected else "✗"
                print(f"  {course_id}: Expected {expected}, Got {actual} {status}")
        
        print("=== END DIAGNOSIS ===\n")