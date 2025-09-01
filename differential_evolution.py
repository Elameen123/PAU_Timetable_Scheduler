import random
from typing import List
import copy
from utils import Utility
from entitities.Class import Class
from input_data import input_data
import numpy as np
from constraints import Constraints
import re
import dash
from dash import dcc, html, Input, Output, State, clientside_callback
from dash.dependencies import ALL
import json

# population initialization using input_data
class DifferentialEvolution:
    def __init__(self, input_data, pop_size: int, F: float, CR: float):
        self.desired_fitness = 0
        self.input_data = input_data
        self.rooms = input_data.rooms
        self.timeslots = input_data.create_time_slots(no_hours_per_day=input_data.hours, no_days_per_week=input_data.days, day_start_time=9)
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
        
        self.population = self.initialize_population()  # List to hold all chromosomes

    def create_events(self):
        events_list = []
        event_map = {}

        idx = 0
        for student_group in self.student_groups:
            for i in range(student_group.no_courses):
                # Reset hourcount for each new course to correctly group events
                hourcount = 1 
                while hourcount <= student_group.hours_required[i]:
                    event = Class(student_group, student_group.teacherIDS[i], student_group.courseIDs[i])
                    events_list.append(event)
                    
                    # Add the event to the index map with the current index
                    event_map[idx] = event
                    idx += 1
                    hourcount += 1
                    
        return events_list, event_map

    def initialize_population(self):
        population = [] 
        for i in range(self.pop_size):
            chromosome = self.create_chromosome()
            population.append(chromosome)
        return np.array(population)

    def create_chromosome(self):
        chromosome = np.empty((len(self.rooms), len(self.timeslots)), dtype=object)
        
        # Group events by student group and course to handle them as blocks
        events_by_group_course = {}
        for idx, event in enumerate(self.events_list):
            key = (event.student_group.id, event.course_id)
            if key not in events_by_group_course:
                events_by_group_course[key] = []
            events_by_group_course[key].append(idx)

        # Trackers for optimized placement
        hours_per_day_for_group = {sg.id: [0] * input_data.days for sg in self.student_groups}
        non_sst_course_count_for_group = {sg.id: 0 for sg in self.student_groups}
        course_days_used = {}

        # Randomize course processing order for population diversity
        course_items = list(events_by_group_course.items())
        random.shuffle(course_items)

        for (student_group_id, course_id), event_indices in course_items:
            course = input_data.getCourse(course_id)
            student_group = input_data.getStudentGroup(student_group_id)
            hours_required = len(event_indices)

            if hours_required == 0:
                continue

            # Decide on a split strategy based on course credits
            if hours_required == 3:
                # Must have at least 2 consecutive hours. Prefer 3.
                split_strategy = random.choice([(3,), (2, 1), (3,)]) # Weight towards (3,)
            elif hours_required == 2:
                # Must be 2 consecutive hours.
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
                available_days = [d for d in range(input_data.days) if d not in course_days_used[course_key]]
                sorted_days = sorted(available_days, key=lambda d: hours_per_day_for_group[student_group_id][d])

                for day_idx in sorted_days:
                    day_start = day_idx * input_data.hours
                    day_end = (day_idx + 1) * input_data.hours
                    
                    possible_slots = []
                    for room_idx, room in enumerate(self.rooms):
                        if self.is_room_suitable(room, course):
                            # Apply building constraints
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
                            elif room_building == 'SST': # Non-engineering
                                building_allowed = False
                            
                            if building_allowed:
                                # Find consecutive slots
                                for timeslot_start in range(day_start, day_end):
                                    if timeslot_start + block_hours > day_end:
                                        continue
                                    
                                    # Use the new, more specific check
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
                        break  # Move to the next block
                
                if placed:
                    break

        # Final verification to place any unassigned events
        chromosome = self.verify_and_repair_course_allocations(chromosome)
        return chromosome

    def find_consecutive_slots(self, chromosome, course):
        # Randomly find consecutive time slots in the same room
        two_slot_rooms = []
        for room_idx, room in enumerate(self.rooms):
            if self.is_room_suitable(room, course):
                # Collect pairs of consecutive available time slots
                for i in range(len(self.timeslots) - 1):
                    if self.is_slot_available(chromosome, room_idx, i) and self.is_slot_available(chromosome, room_idx, i + 1):
                        two_slot_rooms.append((room_idx, i, i+1))

        if len(two_slot_rooms) != 0:
            _room_idx, slot1, slot2 = random.choice(two_slot_rooms)           
            return _room_idx, slot1, slot2
        
        return None, None, None

    def find_single_slot(self, chromosome, course):
        # Randomly find a single available slot
        single_slot_rooms = []
        for room_idx, room in enumerate(self.rooms):
            if self.is_room_suitable(room, course):
                for i in range(len(self.timeslots)):
                    if self.is_slot_available(chromosome, room_idx, i):
                        single_slot_rooms.append((room_idx, i))
        
        # Randomly pick from the available single slots
        if len(single_slot_rooms) > 0:
            return random.choice(single_slot_rooms)
        
        # If no valid single slots are found
        return None, None

    def is_slot_available(self, chromosome, room_idx, timeslot_idx):
        # Check if the slot is available (i.e., not already assigned)
        if chromosome[room_idx][timeslot_idx] is not None:
            return False
        
        # Check if this is break time (13:00 - 14:00)
        # Break time is the 5th hour (index 4) of each day starting from 9:00
        break_hour = 4  # 13:00 is the 5th hour (index 4) starting from 9:00
        day = timeslot_idx // input_data.hours
        hour_in_day = timeslot_idx % input_data.hours
        
        # No break time on Tuesday (1) and Thursday (3)
        if hour_in_day == break_hour and day not in [1, 3]:
            return False  # Break time slot is not available
        
        return True

    def is_slot_available_for_event(self, chromosome, room_idx, timeslot_idx, event):
        """
        Checks if a slot is available for a specific event, considering lecturer availability.
        """
        # Check if the slot is physically empty
        if chromosome[room_idx][timeslot_idx] is not None:
            return False

        # Check for break time
        break_hour = 4
        day = timeslot_idx // input_data.hours
        hour_in_day = timeslot_idx % input_data.hours
        if hour_in_day == break_hour and day not in [1, 3]:
            return False

        # Check lecturer schedule constraints
        if event and event.faculty_id is not None:
            faculty = input_data.getFaculty(event.faculty_id)
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
                        slot_start_h = int(timeslot_obj.start_time.split(':')[0])

                        # Corrected logic: The slot is valid if its start time is strictly less than the lecturer's end time.
                        # e.g., if available until 17:00 (end_avail_h=17), the 16:00-17:00 slot (slot_start_h=16) is valid.
                        if not (start_avail_h <= slot_start_h < end_avail_h):
                            return False
                    except (ValueError, IndexError):
                        # This case should be prevented by the initial validation,
                        # but we return False as a safeguard.
                        return False

        return True

    def is_room_suitable(self, room, course):
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
        """Checks if a student group is already scheduled at a given timeslot."""
        for r_idx in range(len(self.rooms)):
            event_id = chromosome[r_idx][timeslot_idx]
            if event_id is not None:
                event = self.events_map.get(event_id)
                if event and event.student_group.id == student_group_id:
                    return False  # Clash: Student group is not available
        return True

    def _is_lecturer_available(self, chromosome, faculty_id, timeslot_idx):
        """Checks if a lecturer is already scheduled at a given timeslot."""
        for r_idx in range(len(self.rooms)):
            event_id = chromosome[r_idx][timeslot_idx]
            if event_id is not None:
                event = self.events_map.get(event_id)
                if event and event.faculty_id == faculty_id:
                    return False  # Clash: Lecturer is not available
        return True

    def find_clash(self, chromosome):
        """Finds a random timeslot with a student or lecturer clash."""
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
                if not event: continue
                
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
        return np.sum(chromosome1.flatten() != chromosome2.flatten())

    def calculate_population_diversity(self):
        # Optimization: Sample diversity calculation instead of full O(n²)
        if self.pop_size <= 10:
            # For small populations, calculate full diversity
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
        mutant_vector = self.population[target_idx].copy()

        # Strategy 1: Targeted Clash Resolution
        if random.random() < 0.7: # High probability to focus on fixing clashes
            clash_timeslot = self.find_clash(mutant_vector)
            if clash_timeslot is not None:
                # Identify one of the clashing events at this timeslot to move
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
                        course = input_data.getCourse(event_to_move.course_id)
                        for r_idx, room in enumerate(self.rooms):
                            if self.is_room_suitable(room, course):
                                for t_idx in range(len(self.timeslots)):
                                    # Use the comprehensive check to ensure lecturer schedule is respected
                                    if (self.is_slot_available_for_event(mutant_vector, r_idx, t_idx, event_to_move) and
                                        self._is_student_group_available(mutant_vector, event_to_move.student_group.id, t_idx)):
                                        possible_slots.append((r_idx, t_idx))
                        
                        if possible_slots:
                            r, t = random.choice(possible_slots)
                            mutant_vector[r][t] = event_id_to_move
                        # If no slot is found, repair will handle it, but we've at least resolved the clash

        # Strategy 2: Perform a few swaps to introduce small variations
        if random.random() < 0.2: # Lower probability for random swaps
            for _ in range(random.randint(1, 2)):
                occupied_slots = np.argwhere(mutant_vector != None)
                if len(occupied_slots) < 2: continue
                idx1, idx2 = random.sample(range(len(occupied_slots)), 2)
                pos1, pos2 = tuple(occupied_slots[idx1]), tuple(occupied_slots[idx2])
                mutant_vector[pos1], mutant_vector[pos2] = mutant_vector[pos2], mutant_vector[pos1]

        return mutant_vector

    def ensure_valid_solution(self, mutant_vector):
        """Ensure same course on same day appears in same room and handle course splits."""
        course_day_room_mapping = {}
        
        # First pass: collect course-day-room mappings
        for room_idx in range(len(self.rooms)):
            for timeslot_idx in range(len(self.timeslots)):
                event_id = mutant_vector[room_idx][timeslot_idx]
                if event_id is not None:
                    class_event = self.events_map.get(event_id)
                    if class_event:
                        day_idx = timeslot_idx // input_data.hours
                        course = input_data.getCourse(class_event.course_id)
                        course_id = getattr(course, 'course_id', None) or getattr(course, 'id', None) or getattr(course, 'code', None) if course else class_event.course_id
                        course_day_key = (course_id, day_idx, class_event.student_group.id)
                        
                        if course_day_key not in course_day_room_mapping:
                            course_day_room_mapping[course_day_key] = room_idx
        
        # Second pass: fix room violations
        events_to_move = []
        for room_idx in range(len(self.rooms)):
            for timeslot_idx in range(len(self.timeslots)):
                event_id = mutant_vector[room_idx][timeslot_idx]
                if event_id is not None:
                    class_event = self.events_map.get(event_id)
                    if class_event:
                        day_idx = timeslot_idx // input_data.hours
                        course = input_data.getCourse(class_event.course_id)
                        course_id = getattr(course, 'course_id', None) or getattr(course, 'id', None) or getattr(course, 'code', None) if course else class_event.course_id
                        course_day_key = (course_id, day_idx, class_event.student_group.id)
                        expected_room = course_day_room_mapping.get(course_day_key)
                        
                        if expected_room is not None and room_idx != expected_room:
                            mutant_vector[room_idx][timeslot_idx] = None
                            events_to_move.append((event_id, expected_room, timeslot_idx))
        
        # Third pass: place moved events in correct rooms
        for event_id, correct_room, original_timeslot in events_to_move:
            placed = False
            # Try to find an available slot in the correct room
            for timeslot in range(len(self.timeslots)):
                if self.is_slot_available(mutant_vector, correct_room, timeslot):
                    mutant_vector[correct_room][timeslot] = event_id
                    placed = True
                    break
            
            # If not placed, it will be handled by the repair function
        
        # Repair course allocations to ensure all events are scheduled
        mutant_vector = self.verify_and_repair_course_allocations(mutant_vector)
        
        return mutant_vector
    
    def count_non_none(self, arr):
        # Flatten the 2D array and count elements that are not None
        return np.count_nonzero(arr != None)
    
    def crossover(self, target_vector, mutant_vector):
        """
        Performs an enhanced Strategic Crossover.
        It attempts to fix multiple conflicts in the target by using genes from the mutant.
        """
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
                # If the current position in the target has a clash
                if (r, t) in clash_positions:
                    mutant_gene = mutant_vector[r, t]
                    target_gene = trial_vector[r, t]

                    # If the mutant gene is different and not None
                    if mutant_gene != target_gene and mutant_gene is not None:
                        # Check if this new gene would introduce a new clash at this timeslot
                        mutant_event = self.events_map.get(mutant_gene)
                        if not mutant_event: continue

                        is_safe_to_swap = True
                        # Check against all other events in the same timeslot in the trial vector
                        for r_check in range(len(self.rooms)):
                            if r_check != r:
                                existing_event_id = trial_vector[r_check, t]
                                if existing_event_id is not None:
                                    existing_event = self.events_map.get(existing_event_id)
                                    if existing_event:
                                        # Check for student group or lecturer clash with the new gene
                                        if (existing_event.student_group.id == mutant_event.student_group.id or
                                            existing_event.faculty_id == mutant_event.faculty_id):
                                            is_safe_to_swap = False
                                            break
                        
                        if is_safe_to_swap:
                            # The swap is considered safe, so perform it
                            trial_vector[r, t] = mutant_gene
                            
        return trial_vector

    
    def evaluate_fitness(self, chromosome):
        # Optimization: Use cached fitness if available
        chromosome_key = str(chromosome.tobytes())
        if chromosome_key in self.fitness_cache:
            return self.fitness_cache[chromosome_key]
        
        # Use the centralized Constraints class for consistent evaluation
        fitness = self.constraints.evaluate_fitness(chromosome)
        
        # Cache management: prevent unlimited growth (reduced frequency for speed)
        if len(self.fitness_cache) > 1000:  # Reduced from 2000 for faster cache management
            # Keep only the most recent 500 entries
            keys_to_remove = list(self.fitness_cache.keys())[:-500]
            for key in keys_to_remove:
                del self.fitness_cache[key]
        
        # Cache the result
        self.fitness_cache[chromosome_key] = fitness
        return fitness

    # Original DE constraint methods preserved for reference
    # These are now replaced by the centralized Constraints class above   

    def check_room_constraints(self, chromosome):
        """
        rooms must meet the capacity and type of the scheduled event
        """
        point = 0
        for room_idx in range(len(self.rooms)):
            room = self.rooms[room_idx]
            for timeslot_idx in range(len(self.timeslots)):
                class_event = self.events_map.get(chromosome[room_idx][timeslot_idx])
                if class_event is not None:
                    course = input_data.getCourse(class_event.course_id)
                    # H1: Room capacity and type constraints
                    if room.room_type != course.required_room_type or class_event.student_group.no_students > room.capacity:
                        point += 1

        return point
       
    
    def check_student_group_constraints(self, chromosome):
        penalty = 0
        for i in range(len(self.timeslots)):
            simultaneous_class_events = chromosome[:, i]
            student_group_watch = set()
            for class_event_idx in simultaneous_class_events:
                if class_event_idx is not None:
                    class_event = self.events_map.get(class_event_idx)
                    student_group = class_event.student_group
                    if student_group.id in student_group_watch:
                        penalty += 1
                    else:
                        student_group_watch.add(student_group.id)

        return penalty
    
    def check_lecturer_availability(self, chromosome):
        penalty = 0
        for i in range(len(self.timeslots)):
            simultaneous_class_events = chromosome[:, i]
            lecturer_watch = set()
            for class_event_idx in simultaneous_class_events:
                if class_event_idx is not None:
                    class_event = self.events_map.get(class_event_idx)
                    faculty_id = class_event.faculty_id
                    if faculty_id in lecturer_watch:
                        penalty += 1
                    else:
                        lecturer_watch.add(faculty_id)

        return penalty


    # def check_room_time_conflict(self, chromosome):
        penalty = 0

        # Check if multiple events are scheduled in the same room at the same time
        for room_idx, room_schedule in enumerate(chromosome):
            for timeslot_idx, class_event in enumerate(room_schedule):
                if class_event is not None:
                    # H3: Ensure only one event is scheduled per timeslot per room
                    if isinstance(class_event, list) and len(class_event) > 1:
                        penalty += 1000  # Penalty for multiple events in the same room at the same time

        return penalty
# -------------------------------------------
    # def check_valid_timeslot(self, chromosome):
    #     penalty = 0
        
    #     for room_idx, room_schedule in enumerate(chromosome):
    #         for timeslot_idx, class_event in enumerate(room_schedule):
    #             if class_event is not None:  # Event scheduled
    #                 # H5: Ensure the timeslot is valid for this event
    #                 if not self.timeslots[timeslot_idx].is_valid_for_event(class_event):
    #                     penalty += 500  # Moderate penalty for invalid timeslot

    #     return penalty


    def select(self, target_idx, trial_vector):
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
            # If hard constraints are equal, compare total fitness (which includes soft constraints)
            if trial_violations.get('total', float('inf')) <= target_violations.get('total', float('inf')):
                accept = True

        if accept:
            # Decouple repair from selection: Accept the trial vector as is.
            # The repair function will be called on all population members later in the main loop.
            self.population[target_idx] = trial_vector


    def run(self, max_generations):
        # Population already initialized in __init__, don't reinitialize
        fitness_history = []
        best_solution = self.population[0]
        diversity_history = []
        
        # Optimization: Calculate initial fitness and find best solution
        initial_fitness = [self.evaluate_fitness(ind) for ind in self.population]
        best_idx = np.argmin(initial_fitness)
        best_solution = self.population[best_idx].copy()
        best_fitness = initial_fitness[best_idx]
        
        # Track fitness for early convergence detection
        stagnation_counter = 0
        last_improvement = best_fitness

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
                
            # Optimization: Find best solution more efficiently
            current_fitness = [self.evaluate_fitness(ind) for ind in self.population]
            current_best_idx = np.argmin(current_fitness)
            current_best_fitness = current_fitness[current_best_idx]
            
            if current_best_fitness < best_fitness:
                best_solution = self.population[current_best_idx].copy()
                best_fitness = current_best_fitness
                stagnation_counter = 0
                last_improvement = best_fitness
                
                # Ensure best solution has all courses properly allocated
                best_solution = self.verify_and_repair_course_allocations(best_solution)
            else:
                stagnation_counter += 1
            
            fitness_history.append(best_fitness)

            # Optimization: Calculate diversity less frequently for speed
            if generation % 20 == 0:  # Only every 20 generations (reduced from 10)
                population_diversity = self.calculate_population_diversity()
                diversity_history.append(population_diversity)

            print(f"Best solution for generation {generation+1}/{max_generations} has a fitness of: {best_fitness}")

            if best_fitness == self.desired_fitness:
                print(f"Solution with desired fitness of {self.desired_fitness} found at Generation {generation}! 🎉")
                break  # Stop if the best solution has no constraint violations
            
            # Early termination if no improvement for many generations
            if stagnation_counter > 50 and best_fitness < 100:
                print(f"Early termination due to convergence at generation {generation+1}")
                break

        # Final verification and repair of the best solution to ensure all courses are allocated
        best_solution = self.verify_and_repair_course_allocations(best_solution)
        
        return best_solution, fitness_history, generation, diversity_history


    def print_timetable(self, individual, student_group, days, hours_per_day, day_start_time=9):
        timetable = [["" for _ in range(days)] for _ in range(hours_per_day)]
        
        # First, fill break time slots on Mon, Wed, Fri
        break_hour = 4  # 13:00 is the 5th hour (index 4) starting from 9:00
        if break_hour < hours_per_day:
            for day in range(days):
                if day in [0, 2, 4]: # Monday, Wednesday, Friday
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
                        course = input_data.getCourse(class_event.course_id)
                        faculty = input_data.getFaculty(class_event.faculty_id)
                        course_code = course.code if course is not None else "Unknown"
                        faculty_name = faculty.name if faculty is not None else "Unknown"
                        room_obj = input_data.rooms[room_idx]
                        room_display = getattr(room_obj, "name", getattr(room_obj, "Id", str(room_idx)))
                        timetable[hour][day] = f"Course: {course_code}, Lecturer: {faculty_name}, Room: {room_display}"
        return timetable

    def print_all_timetables(self, individual, days, hours_per_day, day_start_time=9):
        # app.layout = html.Div([
        #     html.H1("Timetable"),
        #     html.Div([
        #         html.Label("Select Student Group:"),
        #         dcc.Dropdown(
        #             id='student-group-dropdown',
        #             options=[{'label': student_group.name, 'value': student_group.id} for student_group in input_data.student_groups],
        #             value=input_data.student_groups[0].id
        #         ),
        #     ]),
        #     html.Div(id='timetable-container')
        # ])
        data = []
        # Find all unique student groups in the individual
        student_groups = input_data.student_groups
        
        # Print timetable for each student group
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
                    course = input_data.getCourse(event.course_id)
                    if course:
                        day_idx = t_idx // input_data.hours
                        course_id = getattr(course, 'course_id', None) or getattr(course, 'id', None) or getattr(course, 'code', None)
                        course_day_key = (course_id, day_idx, event.student_group.id)
                        course_day_room_mapping[course_day_key] = r_idx

            for missing_event_id in missing_events:
                event = self.events_list[missing_event_id]
                course = input_data.getCourse(event.course_id)
                if not course: continue
                
                course_id = getattr(course, 'course_id', None) or getattr(course, 'id', None) or getattr(course, 'code', None)
                placed = False

                # Strategy 1: Place in the same room as other instances of the same course on the same day
                if flexibility_level == 0:
                    preferred_slots = []
                    for day_idx in range(input_data.days):
                        course_day_key = (course_id, day_idx, event.student_group.id)
                        if course_day_key in course_day_room_mapping:
                            preferred_room = course_day_room_mapping[course_day_key]
                            day_start, day_end = day_idx * input_data.hours, (day_idx + 1) * input_data.hours
                            for timeslot in range(day_start, day_end):
                                # Use the comprehensive check here as well
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
                                    # Lecturer availability is already checked by is_slot_available_for_event
                                    valid_slots.append((room_idx, timeslot_idx))
                    if valid_slots:
                        room_idx, timeslot_idx = random.choice(valid_slots)
                        chromosome[room_idx][timeslot_idx] = missing_event_id
                        placed = True

                # Strategy 3 (Final Pass): Only place in a valid, empty slot.
                if not placed and flexibility_level >= 2:
                    # This is a last resort. Find any valid and completely empty slot.
                    # We avoid displacing other events here as it can cascade issues.
                    valid_empty_slots = []
                    for room_idx, room in enumerate(self.rooms):
                        if self.is_room_suitable(room, course):
                            for timeslot_idx in range(len(self.timeslots)):
                                if (chromosome[room_idx][timeslot_idx] is None and # Must be empty
                                    self.is_slot_available_for_event(chromosome, room_idx, timeslot_idx, event) and
                                    self._is_student_group_available(chromosome, event.student_group.id, timeslot_idx)):
                                    valid_empty_slots.append((room_idx, timeslot_idx))
                    
                    if valid_empty_slots:
                        room_idx, timeslot_idx = random.choice(valid_empty_slots)
                        chromosome[room_idx][timeslot_idx] = missing_event_id
                        placed = True
        return chromosome

    def count_course_occurrences(self, chromosome, student_group):
        """
        Count how many times each course appears for a specific student group
        """
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
        """
        Diagnostic method to check course allocations for debugging
        """
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

# Create DE instance and run optimization
print("Starting Differential Evolution")
de = DifferentialEvolution(input_data, 50, 0.4, 0.9)
best_solution, fitness_history, generation, diversity_history = de.run(1)
print("Differential Evolution completed")

# Get final fitness and detailed breakdown
final_fitness = de.evaluate_fitness(best_solution)
violations = de.constraints.get_constraint_violations(best_solution)

print("\n--- Final Timetable Fitness Breakdown ---")
print(f"Total Fitness Score: {final_fitness:.2f}\n")

print("Constraint Violation Details:")

descriptive_names = {
    'room_constraints': "Room Capacity/Type Conflicts",
    'student_group_constraints': "Student Group Clashes",
    'lecturer_availability': "Lecturer Clashes (Overlapping)",
    'lecturer_schedule_constraints': "Lecturer Schedule Conflicts (Day/Time)",
    'room_time_conflict': "Room Time Slot Conflicts",
    'building_assignments': "Building Assignment Conflicts",
    'same_course_same_room_per_day': "Same Course in Multiple Rooms on Same Day",
    'break_time_constraint': "Classes During Break Time",
    'course_allocation_completeness': "Missing or Extra Classes",
    'single_event_per_day': "Multiple Events on Same Day (Soft)",
    'consecutive_timeslots': "Consecutive Slot Violations",
    'spread_events': "Poor Event Spreading (Soft)"
}

penalty_info = {
    'room_constraints': "1 point per violation",
    'student_group_constraints': "1 point per violation",
    'lecturer_availability': "1 point per violation",
    'lecturer_schedule_constraints': "10 points per violation",
    'room_time_conflict': "10 points per conflict",
    'building_assignments': "0.5 points per violation",
    'same_course_same_room_per_day': "5 points per extra room used",
    'break_time_constraint': "100 points per scheduled class",
    'course_allocation_completeness': "2-5 points per missing hour",
    'single_event_per_day': "0.05 points per extra event on same day",
    'consecutive_timeslots': "0.05 points per hour of multi-hour course",
    'spread_events': "0.025 points per group with clustered events"
}

# Sort and print for clarity
sorted_violations = sorted(violations.items(), key=lambda item: item[0] if item[0] != 'total' else 'zzz')
for constraint, points in sorted_violations:
    if constraint != 'total':
        display_name = descriptive_names.get(constraint, constraint.replace('_', ' ').title())
        penalty_str = penalty_info.get(constraint, "...")
        print(f"- {display_name}: {points} ({penalty_str})")

print(f"\nCalculated Total: {violations.get('total', 'N/A')}")
print("--- End of Fitness Breakdown ---\n")

# Get the timetable data from the DE optimization
all_timetables = de.print_all_timetables(best_solution, input_data.days, input_data.hours, 9)

# Convert student_group objects to dictionaries for JSON serialization
for timetable_data in all_timetables:
    if hasattr(timetable_data['student_group'], 'name'):
        timetable_data['student_group'] = {
            'name': timetable_data['student_group'].name,
            'id': getattr(timetable_data['student_group'], 'id', None)
        }

# Try to load saved timetable if it exists
def load_saved_timetable():
    import json
    import os
    import traceback
    
    save_path = os.path.join(os.path.dirname(__file__), 'data', 'timetable_data.json')
    
    if os.path.exists(save_path):
        try:
            with open(save_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            
            file_size = os.path.getsize(save_path)
            print(f"📁 Loaded saved timetable: {len(saved_data)} groups, {file_size} bytes")
            
            return saved_data
        except Exception as e:
            print(f"❌ Error loading saved timetable: {e}")
            traceback.print_exc()
    
    return None

def clear_saved_timetable():
    """Clear the saved timetable file to start fresh on next run"""
    import os
    save_path = os.path.join(os.path.dirname(__file__), 'data', 'timetable_data.json')
    backup_path = os.path.join(os.path.dirname(__file__), 'data', 'timetable_data_backup.json')
    
    try:
        if os.path.exists(save_path):
            os.remove(save_path)
            print("🗑️ Cleared saved timetable file")
        if os.path.exists(backup_path):
            os.remove(backup_path)
            print("🗑️ Cleared backup timetable file")
    except Exception as e:
        print(f"❌ Error clearing saved files: {e}")

# Load saved data if available, otherwise use freshly optimized data
print("\n=== LOADING TIMETABLE DATA ===")
# On fresh startup, always use the newly optimized data from DE
# Only load saved data during Dash session for persistence
print("✅ Using freshly optimized timetable data")
print(f"   Generated {len(all_timetables)} student groups from DE optimization")
print("=== TIMETABLE DATA LOADING COMPLETE ===\n")

days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
hours = [f"{9 + i}:00" for i in range(input_data.hours)]

# Session tracker to know if we've made any swaps
session_has_swaps = False

app = dash.Dash(__name__)

# Add CSS styles to the app for drag and drop functionality
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
    # Title and dropdown on the same line
    html.Div([
        html.H1("Interactive Drag & Drop Timetable - DE Optimization Results", 
                style={"color": "#11214D", "fontWeight": "600", "fontSize": "24px", 
                      "fontFamily": "Poppins, sans-serif", "margin": "0", "flex": "1"}),
        
        dcc.Dropdown(
            id='student-group-dropdown',
            options=[{'label': timetable_data['student_group']['name'] if isinstance(timetable_data['student_group'], dict) 
                              else timetable_data['student_group'].name, 'value': idx} 
                    for idx, timetable_data in enumerate(all_timetables)],
            value=0,
            style={"width": "280px", "fontSize": "13px", "fontFamily": "Poppins, sans-serif"}
        )
    ], style={"display": "flex", "alignItems": "center", "justifyContent": "space-between", 
             "marginTop": "30px", "marginBottom": "30px", "maxWidth": "1200px", 
             "margin": "30px auto", "padding": "0 15px"}),
    
    # Store for timetable data
    dcc.Store(id="all-timetables-store", data=all_timetables),
    
    # Store for communicating swaps
    dcc.Store(id="swap-data", data=None),
    
    # Hidden div to trigger the setup
    html.Div(id="trigger", style={"display": "none"}),
    
    # Timetable container
    html.Div(id="timetable-container"),
    
    # Button to save current timetable state
    html.Div([
        html.Button("Save Current Timetable", id="save-button", 
                   style={"backgroundColor": "#11214D", "color": "white", "padding": "10px 20px", 
                         "border": "none", "borderRadius": "5px", "fontSize": "14px", "cursor": "pointer",
                         "fontWeight": "600", "boxShadow": "0 1px 3px rgba(0,0,0,0.1)",
                         "transition": "all 0.2s ease", "fontFamily": "Poppins, sans-serif"}),
        html.Div(id="save-status", style={"marginTop": "12px", "fontWeight": "600", 
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

@app.callback(
    [Output("timetable-container", "children"),
     Output("trigger", "children")],
    [Input("all-timetables-store", "data"),
     Input("student-group-dropdown", "value")]
)
def create_timetable(all_timetables_data, selected_group_idx):
    # Only load saved data if we're in an active session and there have been changes
    # Don't load on fresh startup - use the fresh DE results
    global all_timetables
    
    if selected_group_idx is None or not all_timetables_data:
        return html.Div("No data available"), "trigger"
    
    # Get the selected student group data
    timetable_data = all_timetables_data[selected_group_idx]
    student_group_name = timetable_data['student_group']['name'] if isinstance(timetable_data['student_group'], dict) else timetable_data['student_group'].name
    timetable_rows = timetable_data['timetable']
    
    # Create table rows
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
        cells = [html.Td(timetable_rows[row_idx][0], className="time-cell")]  # Time column with special class
        
        for col_idx in range(1, len(timetable_rows[row_idx])):  # Skip time column
            cell_content = timetable_rows[row_idx][col_idx] if timetable_rows[row_idx][col_idx] else "FREE"
            cell_id = {"type": "cell", "group": selected_group_idx, "row": row_idx, "col": col_idx-1}
            
            # Check if this is a break time
            is_break = cell_content == "BREAK"
            cell_class = "cell break-time" if is_break else "cell"
            
            # Make break times non-draggable
            draggable = "false" if is_break else "true"
            
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
        html.H2(f"Timetable for {student_group_name}", 
               className="timetable-title",
               style={"textAlign": "center", "color": "#11214D", "marginBottom": "20px", 
                     "fontWeight": "600", "fontSize": "20px", "fontFamily": "Poppins, sans-serif"}),
        table
    ], className="student-group-container"), "trigger"

# Client-side callback for drag and drop functionality
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
                    
                    // Prevent dragging break times
                    if (this.classList.contains('break-time') || this.textContent.trim() === 'BREAK') {
                        console.log('Cannot drag break time');
                        e.preventDefault();
                        return false;
                    }
                    
                    window.draggedElement = this;
                    
                    // Get row and col from the element's ID
                    const idStr = this.id;
                    try {
                        const idObj = JSON.parse(idStr);
                        window.dragStartData = {
                            group: idObj.group,
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
                    // Don't allow dropping on break times
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
                    
                    console.log('Drop detected');
                    
                    // Prevent dropping on break times
                    if (this.classList.contains('break-time') || this.textContent.trim() === 'BREAK') {
                        console.log('Cannot drop on break time');
                        this.classList.remove('drag-over');
                        return false;
                    }
                    
                    if (window.draggedElement && this !== window.draggedElement) {
                        // Get target data
                        const targetIdStr = this.id;
                        try {
                            const targetIdObj = JSON.parse(targetIdStr);
                            const targetData = {
                                group: targetIdObj.group,
                                row: targetIdObj.row,
                                col: targetIdObj.col,
                                content: this.textContent.trim()
                            };
                            
                            console.log('Swapping:', window.dragStartData, 'with:', targetData);
                            
                            // Perform the swap
                            const tempContent = window.draggedElement.textContent;
                            window.draggedElement.textContent = this.textContent;
                            this.textContent = tempContent;
                            
                            // Trigger callback to update backend data
                            window.dash_clientside.set_props("swap-data", {
                                data: {
                                    source: window.dragStartData,
                                    target: targetData,
                                    timestamp: Date.now()
                                }
                            });
                            
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
    Output("feedback", "style"),
    Input("trigger", "children"),
    prevent_initial_call=False
)

# Callback to handle swaps and update the backend data
@app.callback(
    Output("all-timetables-store", "data"),
    Input("swap-data", "data"),
    State("all-timetables-store", "data"),
    prevent_initial_call=True
)
def handle_swap(swap_data, current_timetables):
    global all_timetables, session_has_swaps
    
    if not swap_data or not current_timetables:
        return current_timetables
    
    try:
        source = swap_data['source']
        target = swap_data['target']
        
        # Make sure both swaps are in the same group
        if source['group'] != target['group']:
            print("Cannot swap between different student groups")
            return current_timetables
        
        # Prevent swapping with BREAK times
        if source['content'] == 'BREAK' or target['content'] == 'BREAK':
            print("Cannot swap with BREAK time")
            return current_timetables
        
        # Mark that we've made swaps in this session
        session_has_swaps = True
        
        group_idx = source['group']
        
        # Update the timetable data
        timetable_rows = current_timetables[group_idx]['timetable']
        
        # Get the current content (skip time column, so add 1 to col index)
        source_content = timetable_rows[source['row']][source['col'] + 1]
        target_content = timetable_rows[target['row']][target['col'] + 1]
        
        # Perform the swap
        timetable_rows[source['row']][source['col'] + 1] = target_content
        timetable_rows[target['row']][target['col'] + 1] = source_content
        
        print(f"Backend swap completed: {source_content} <-> {target_content}")
        
        # Update the global all_timetables variable to ensure persistence
        all_timetables[group_idx]['timetable'] = timetable_rows
        
        # Also update the current_timetables to return
        current_timetables[group_idx]['timetable'] = timetable_rows
        
        # Debug: Print some info about what we're about to save
        print(f"DEBUG: About to save data for group {group_idx}")
        print(f"DEBUG: Source row {source['row']}, col {source['col']}: '{current_timetables[group_idx]['timetable'][source['row']][source['col'] + 1]}'")
        print(f"DEBUG: Target row {target['row']}, col {target['col']}: '{current_timetables[group_idx]['timetable'][target['row']][target['col'] + 1]}'")
        
        # Auto-save the changes to maintain persistence
        try:
            import os
            import json
            import time
            import shutil
            import traceback
            
            save_dir = os.path.join(os.path.dirname(__file__), 'data')
            os.makedirs(save_dir, exist_ok=True)  # Use exist_ok=True to avoid errors
            save_path = os.path.join(save_dir, 'timetable_data.json')
            
            # Add timestamp to track saves
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Attempting to auto-save swap: {source['content']} <-> {target['content']}")
            print(f"Auto-save path: {save_path}")
            print(f"Number of student groups: {len(current_timetables)}")
            
            # Create a backup before saving
            backup_path = os.path.join(save_dir, 'timetable_data_backup.json')
            if os.path.exists(save_path):
                shutil.copy2(save_path, backup_path)
                print(f"Created backup at: {backup_path}")
            
            # Force a file flush and sync to ensure data is written to disk
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(current_timetables, f, indent=2, default=str, ensure_ascii=False)
                f.flush()  # Force write to disk
                os.fsync(f.fileno())  # Force OS to write to physical storage
            
            # Verify the file was written successfully
            if os.path.exists(save_path):
                file_size = os.path.getsize(save_path)
                print(f"✅ Auto-saved successfully! File size: {file_size} bytes")
                
                # Verify we can read it back and check the specific swapped data
                with open(save_path, 'r', encoding='utf-8') as f:
                    test_load = json.load(f)
                print(f"✅ Verification: Can read back {len(test_load)} student groups")
                
                # Verify the actual swap was saved
                saved_source = test_load[group_idx]['timetable'][source['row']][source['col'] + 1]
                saved_target = test_load[group_idx]['timetable'][target['row']][target['col'] + 1]
                print(f"✅ Verification: Saved source = '{saved_source}', target = '{saved_target}'")
                
            else:
                print("❌ ERROR: Auto-save file was not created!")
                
        except Exception as auto_save_error:
            print(f"❌ Auto-save failed with error: {auto_save_error}")
            import traceback
            traceback.print_exc()
        
        return current_timetables
        
    except Exception as e:
        print(f"Error handling swap: {e}")
        return current_timetables

# Callback to handle saving the current timetable state
@app.callback(
    Output("save-status", "children"),
    Input("save-button", "n_clicks"),
    State("all-timetables-store", "data"),
    prevent_initial_call=True
)
def save_timetable(n_clicks, current_timetables):
    if n_clicks and current_timetables:
        try:
            # Update the global variable
            global all_timetables
            all_timetables = current_timetables
            
            # Save to JSON file for persistence
            import json
            import os
            
            # Create a data directory if it doesn't exist
            save_dir = os.path.join(os.path.dirname(__file__), 'data')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # Save the timetable data
            save_path = os.path.join(save_dir, 'timetable_data.json')
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(current_timetables, f, indent=2, default=str, ensure_ascii=False)
            
            print("Timetable state saved to file")
            print(f"Current timetables contain {len(all_timetables)} student groups")
            print(f"Saved to: {save_path}")
            
            return html.Div([
                html.Span("✅ Timetable saved successfully!", style={"color": "green", "fontWeight": "bold"}),
                html.Br(),
                html.Small(f"Saved to: {save_path}", style={"color": "gray"})
            ])
            
        except Exception as e:
            print(f"Error saving timetable: {e}")
            return html.Div([
                html.Span("❌ Error saving timetable!", style={"color": "red", "fontWeight": "bold"}),
                html.Br(),
                html.Small(f"Error: {str(e)}", style={"color": "red"})
            ])
    
    return ""

# Callback to refresh timetable data from file on page load
@app.callback(
    Output("all-timetables-store", "data", allow_duplicate=True),
    Input("student-group-dropdown", "value"),
    prevent_initial_call='initial_duplicate'
)
def refresh_timetable_data(selected_group_idx):
    # Only load saved data if we've made swaps in this session
    global session_has_swaps, all_timetables
    
    if session_has_swaps:
        fresh_saved_data = load_saved_timetable()
        if fresh_saved_data:
            print(f"🔄 Session has swaps - loading saved data: {len(fresh_saved_data)} student groups")
            return fresh_saved_data
    
    # Otherwise, use the current all_timetables (fresh DE results)
    print(f"🔄 Using fresh DE results: {len(all_timetables)} student groups")
    return all_timetables

# Run the Dash app
if __name__ == '__main__':
    app.run(debug=False)


    # import time
# pop_size = 50
# F = 0.5
# max_generations = 500
# CR = 0.8

# de = DifferentialEvolution(input_data, pop_size, F, CR)

# start_time = time.time()
# de.run(max_generations)
# de_time = time.time() - start_time

# print(f'Time: {de_time:.2f}s')