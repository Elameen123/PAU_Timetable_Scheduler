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
import os
import shutil
import traceback

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
                'engineering', 'eng', 'computer science', 'software engineering', 'data science',
                'mechatronics', 'electrical', 'mechanical', 'csc', 'sen', 'data', 'ds'
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

        # --- STRATEGY: Prioritize placing larger courses first ("big rocks first") ---
        course_items = sorted(
            events_by_group_course.items(),
            key=lambda item: len(item[1]),
            reverse=True
        )

        # Trackers for optimized placement
        hours_per_day_for_group = {sg.id: [0] * input_data.days for sg in self.student_groups}

        for (student_group_id, course_id), event_indices in course_items:
            course = input_data.getCourse(course_id)
            student_group = input_data.getStudentGroup(student_group_id)
            hours_required = len(event_indices)

            if hours_required == 0:
                continue

            # --- STRATEGY: Stricter split strategies to enforce consecutive constraints ---
            split_strategies = []
            if hours_required >= 4: # 4-hour courses and above
                split_strategies = [(4,), (2, 2), (3, 1)]
            elif hours_required == 3: # 3-hour courses
                split_strategies = [(3,), (2, 1)]
            elif hours_required == 2: # 2-hour courses
                split_strategies = [(2,)] # MUST be consecutive
            else: # 1-hour courses
                split_strategies = [(1,)]

            course_placed = False
            for split_strategy in split_strategies:
                if course_placed:
                    break

                placements_for_strategy = []
                all_blocks_found = True
                temp_chromosome = chromosome.copy()
                
                event_idx_counter = 0
                
                temp_hours_per_day = hours_per_day_for_group[student_group_id][:]
                temp_course_days_used = set()

                for block_hours in split_strategy:
                    placed = False
                    block_event_indices = event_indices[event_idx_counter : event_idx_counter + block_hours]
                    
                    available_days = [d for d in range(input_data.days) if d not in temp_course_days_used]
                    sorted_days = sorted(available_days, key=lambda d: temp_hours_per_day[d])

                    for day_idx in sorted_days:
                        day_start = day_idx * input_data.hours
                        day_end = (day_idx + 1) * input_data.hours
                        
                        possible_slots = []
                        for room_idx, room in enumerate(self.rooms):
                            # --- STRATEGY: Enforce building constraints during placement ---
                            is_engineering_group = student_group.id in self.engineering_groups
                            room_building = self.room_building_cache.get(room_idx, 'UNKNOWN')
                            
                            # Non-engineering groups cannot use SST rooms
                            if not is_engineering_group and room_building == 'SST':
                                continue # Skip this room entirely for this group

                            if self.is_room_suitable(room, course):
                                for timeslot_start in range(day_start, day_end - block_hours + 1):
                                    is_block_placeable = True
                                    for i in range(block_hours):
                                        ts = timeslot_start + i
                                        event_for_slot = self.events_list[block_event_indices[i]]
                                        if not (self.is_slot_available_for_event(temp_chromosome, room_idx, ts, event_for_slot) and
                                                self._is_student_group_available(temp_chromosome, student_group_id, ts) and
                                                self._is_lecturer_available(temp_chromosome, event_for_slot.faculty_id, ts)):
                                            is_block_placeable = False
                                            break
                                    
                                    if is_block_placeable:
                                        possible_slots.append((room_idx, timeslot_start))
                        
                        if possible_slots:
                            # Prefer slots that don't cause building conflicts if possible
                            preferred_slots = []
                            for r_idx, t_start in possible_slots:
                                room_bldg = self.room_building_cache.get(r_idx, 'UNKNOWN')
                                is_eng_grp = student_group.id in self.engineering_groups
                                if is_eng_grp and room_bldg != 'SST':
                                    pass # This is a potential soft conflict
                                else:
                                    preferred_slots.append((r_idx, t_start))
                            
                            if preferred_slots:
                                room_idx, timeslot_start = random.choice(preferred_slots)
                            else: # If all options cause a soft conflict, just pick one
                                room_idx, timeslot_start = random.choice(possible_slots)

                            for i in range(block_hours):
                                ts = timeslot_start + i
                                event_id = block_event_indices[i]
                                placements_for_strategy.append((room_idx, ts, event_id))
                                temp_chromosome[room_idx, ts] = event_id
                            
                            temp_hours_per_day[day_idx] += block_hours
                            temp_course_days_used.add(day_idx)
                            placed = True
                            event_idx_counter += block_hours
                            break 
                    
                    if not placed:
                        all_blocks_found = False
                        break
                
                if all_blocks_found:
                    for r, t, e_id in placements_for_strategy:
                        chromosome[r, t] = e_id
                    
                    hours_per_day_for_group[student_group_id] = temp_hours_per_day
                    
                    course_placed = True
                    break

        # Final verification to place any unassigned events, now more aggressive
        chromosome = self.verify_and_repair_course_allocations(chromosome)
        
        # Final pass to ensure multi-hour courses are consecutive
        chromosome = self.ensure_consecutive_slots(chromosome)
        
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
                slot_hour = self.timeslots[timeslot_idx].start_time + 9

                # Check day availability
                is_day_ok = False
                avail_days = faculty.avail_days
                if not avail_days or (isinstance(avail_days, str) and avail_days.upper() == "ALL"):
                    is_day_ok = True
                else:
                    if isinstance(avail_days, str):
                        avail_days_list = [d.strip().capitalize() for d in avail_days.split(',')]
                    else: # is a list
                        avail_days_list = [d.strip().capitalize() for d in avail_days]
                    if "All" in avail_days_list or day_abbr in avail_days_list:
                        is_day_ok = True
                
                if not is_day_ok:
                    return False

                # Check time availability (Corrected Logic)
                is_time_ok = False
                avail_times = faculty.avail_times
                if not avail_times or (isinstance(avail_times, str) and avail_times.upper() == "ALL") or \
                   (isinstance(avail_times, list) and any(str(t).strip().upper() == 'ALL' for t in avail_times)):
                    is_time_ok = True
                else:
                    time_list = avail_times if isinstance(avail_times, list) else [avail_times]
                    for time_spec in time_list:
                        time_spec_str = str(time_spec).strip()
                        if '-' in time_spec_str:
                            try:
                                start_str, end_str = time_spec_str.split('-')
                                start_h = int(start_str.split(':')[0])
                                end_h = int(end_str.split(':')[0])
                                if start_h <= slot_hour < end_h:
                                    is_time_ok = True
                                    break
                            except (ValueError, IndexError):
                                continue
                        else:
                            try:
                                h = int(time_spec_str.split(':')[0])
                                if h == slot_hour:
                                    is_time_ok = True
                                    break
                            except (ValueError, IndexError):
                                continue
                
                if not is_time_ok:
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
        
        # Increase mutation attempts to encourage exploration
        mutation_attempts = random.randint(3, 8)

        for _ in range(mutation_attempts):
            strategy = random.choice(['resolve_clash', 'safe_swap', 'safe_move'])

            # Strategy 1: Find a clash and try to resolve it by moving one event
            if strategy == 'resolve_clash':
                clash_timeslot = self.find_clash(mutant_vector)
                if clash_timeslot is not None:
                    # Find events involved in the clash
                    events_in_slot = [(r, mutant_vector[r, clash_timeslot]) for r in range(len(self.rooms)) if mutant_vector[r, clash_timeslot] is not None]
                    if not events_in_slot: continue
                    
                    # Pick one event to move
                    room_to_move_from, event_id_to_move = random.choice(events_in_slot)
                    event_to_move = self.events_map.get(event_id_to_move)
                    if not event_to_move: continue

                    # Find a new, valid, empty slot for this event
                    new_pos = self.find_safe_empty_slot_for_event(mutant_vector, event_to_move, ignore_pos=(room_to_move_from, clash_timeslot))
                    if new_pos:
                        new_r, new_t = new_pos
                        mutant_vector[new_r, new_t] = event_id_to_move
                        mutant_vector[room_to_move_from, clash_timeslot] = None
                        continue # Move successful, try another mutation

            # Strategy 2: Swap two existing events if it's safe
            elif strategy == 'safe_swap':
                occupied_slots = np.argwhere(mutant_vector != None)
                if len(occupied_slots) < 2: continue
                
                idx1, idx2 = random.sample(range(len(occupied_slots)), 2)
                pos1, pos2 = tuple(occupied_slots[idx1]), tuple(occupied_slots[idx2])
                
                event1_id, event2_id = mutant_vector[pos1], mutant_vector[pos2]
                event1, event2 = self.events_map.get(event1_id), self.events_map.get(event2_id)
                
                if not event1 or not event2: continue

                # Check if swapping is feasible
                course1, course2 = self.input_data.getCourse(event1.course_id), self.input_data.getCourse(event2.course_id)
                
                # Check room suitability
                room1_ok_for_event2 = self.is_room_suitable(self.rooms[pos1[0]], course2)
                room2_ok_for_event1 = self.is_room_suitable(self.rooms[pos2[0]], course1)

                if room1_ok_for_event2 and room2_ok_for_event1:
                    # Check clash constraints for the swap
                    # Is event2 OK at pos1's timeslot?
                    clash_free_at_pos1 = not self.constraints.check_student_group_clash_at_slot(mutant_vector, event2.student_group.id, pos1[1], ignore_room_idx=pos2[0]) and \
                                         not self.constraints.check_lecturer_clash_at_slot(mutant_vector, event2.faculty_id, pos1[1], ignore_room_idx=pos2[0])
                    
                    # Is event1 OK at pos2's timeslot?
                    clash_free_at_pos2 = not self.constraints.check_student_group_clash_at_slot(mutant_vector, event1.student_group.id, pos2[1], ignore_room_idx=pos1[0]) and \
                                         not self.constraints.check_lecturer_clash_at_slot(mutant_vector, event1.faculty_id, pos2[1], ignore_room_idx=pos1[0])

                    if clash_free_at_pos1 and clash_free_at_pos2:
                        mutant_vector[pos1], mutant_vector[pos2] = event2_id, event1_id
                        continue

            # Strategy 3: Move a single event to a new, safe, empty location
            elif strategy == 'safe_move':
                occupied_slots = np.argwhere(mutant_vector != None)
                if not len(occupied_slots): continue
                
                pos_to_move = tuple(random.choice(occupied_slots))
                event_id_to_move = mutant_vector[pos_to_move]
                event_to_move = self.events_map.get(event_id_to_move)
                if not event_to_move: continue

                # Find a new, valid, empty slot
                new_pos = self.find_safe_empty_slot_for_event(mutant_vector, event_to_move, ignore_pos=pos_to_move)
                if new_pos:
                    new_r, new_t = new_pos
                    mutant_vector[new_r, new_t] = event_id_to_move
                    mutant_vector[pos_to_move] = None
                    continue

        return mutant_vector

    def find_safe_empty_slot_for_event(self, chromosome, event, ignore_pos=None):
        """Finds a random empty slot that is safe for the given event."""
        course = self.input_data.getCourse(event.course_id)
        if not course: return None

        possible_slots = []
        for r_idx, room in enumerate(self.rooms):
            if self.is_room_suitable(room, course):
                for t_idx in range(len(self.timeslots)):
                    if (r_idx, t_idx) == ignore_pos: continue
                    
                    # Check if slot is physically empty and available (break time, etc.)
                    if chromosome[r_idx, t_idx] is None and self.is_slot_available_for_event(chromosome, r_idx, t_idx, event):
                        # Check for potential clashes if we place the event here
                        student_clash = self.constraints.check_student_group_clash_at_slot(chromosome, event.student_group.id, t_idx, ignore_room_idx=ignore_pos[0] if ignore_pos else -1)
                        lecturer_clash = self.constraints.check_lecturer_clash_at_slot(chromosome, event.faculty_id, t_idx, ignore_room_idx=ignore_pos[0] if ignore_pos else -1)
                        
                        if not student_clash and not lecturer_clash:
                            possible_slots.append((r_idx, t_idx))
        
        return random.choice(possible_slots) if possible_slots else None

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
        
        # Final pass to ensure multi-hour courses are consecutive
        mutant_vector = self.ensure_consecutive_slots(mutant_vector)
        
        return mutant_vector
    
    def count_non_none(self, arr):
        # Flatten the 2D array and count elements that are not None
        return np.count_nonzero(arr != None)
    
    def crossover(self, target_vector, mutant_vector):
        """
        Performs a standard binomial (or uniform) crossover for Differential Evolution.
        This creates a trial vector by mixing genes from the target and mutant vectors
        based on the crossover rate (CR). This is more exploratory than the previous
        strategic crossover and helps to escape local optima.
        """
        trial_vector = target_vector.copy()
        num_rooms, num_timeslots = target_vector.shape
        
        # Ensure at least one gene from the mutant vector is picked (j_rand).
        # This is a key part of the DE algorithm to ensure the trial vector is different from the target.
        j_rand_r = random.randrange(num_rooms)
        j_rand_t = random.randrange(num_timeslots)

        for r in range(num_rooms):
            for t in range(num_timeslots):
                # The gene from the mutant is chosen if a random number is less than CR,
                # or if it's the randomly chosen j_rand position.
                if random.random() < self.CR or (r == j_rand_r and t == j_rand_t):
                    trial_vector[r, t] = mutant_vector[r, t]
                    
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
                
                # Lamarckian Step: Repair the new trial vector *before* evaluation and selection.
                # This ensures we are always comparing valid, repaired solutions.
                trial_vector = self.verify_and_repair_course_allocations(trial_vector)
                trial_vector = self.ensure_consecutive_slots(trial_vector)

                # Step 3: Evaluation and Selection
                self.select(i, trial_vector)
                # Post-selection repair is no longer needed as both trial and target are already repaired.
                
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
        best_solution = self.ensure_consecutive_slots(best_solution)
        
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

    def ensure_consecutive_slots(self, chromosome):
        """
        Scans the timetable and attempts to repair any multi-hour courses
        that have been split into non-consecutive slots.
        """
        # Group all scheduled events by course and student group
        events_by_course = {}
        for r_idx in range(len(self.rooms)):
            for t_idx in range(len(self.timeslots)):
                event_id = chromosome[r_idx, t_idx]
                if event_id is not None:
                    event = self.events_map.get(event_id)
                    if not event: continue
                    
                    course_key = (event.student_group.id, event.course_id)
                    if course_key not in events_by_course:
                        events_by_course[course_key] = []
                    events_by_course[course_key].append({'event_id': event_id, 'pos': (r_idx, t_idx)})

        for course_key, events in events_by_course.items():
            hours_required = len(events)
            if hours_required < 2:
                continue # Not a multi-hour course that needs checking

            # Check if the events are consecutive
            positions = sorted([e['pos'] for e in events], key=lambda p: p[1])
            is_consecutive = True
            for i in range(len(positions) - 1):
                # Check if they are in the same room and adjacent timeslots
                if not (positions[i][0] == positions[i+1][0] and positions[i][1] + 1 == positions[i+1][1]):
                    is_consecutive = False
                    break
            
            if is_consecutive:
                continue # This course is fine, move to the next one

            # --- If not consecutive, attempt to repair ---
            # 1. Find a new, valid, consecutive block of slots for the entire course
            student_group_id, course_id = course_key
            course = self.input_data.getCourse(course_id)
            
            possible_blocks = []
            for r_idx, room in enumerate(self.rooms):
                if self.is_room_suitable(room, course):
                    for t_start in range(len(self.timeslots) - hours_required + 1):
                        is_block_valid = True
                        # Temporarily clear old positions to check availability
                        temp_chromosome = chromosome.copy()
                        for event_info in events:
                            r, t = event_info['pos']
                            temp_chromosome[r, t] = None
                        
                        for i in range(hours_required):
                            t_check = t_start + i
                            event_to_place = self.events_list[events[i]['event_id']]
                            
                            # Check if the new slot is valid for this event
                            if not (self.is_slot_available_for_event(temp_chromosome, r_idx, t_check, event_to_place) and
                                    self._is_student_group_available(temp_chromosome, student_group_id, t_check) and
                                    self._is_lecturer_available(temp_chromosome, event_to_place.faculty_id, t_check)):
                                is_block_valid = False
                                break
                        
                        if is_block_valid:
                            possible_blocks.append((r_idx, t_start))
            
            # 2. If a valid block is found, perform the move
            if possible_blocks:
                new_r, new_t_start = random.choice(possible_blocks)
                
                # Clear the old, non-consecutive event positions
                for event_info in events:
                    r, t = event_info['pos']
                    chromosome[r, t] = None
                
                # Place the events in the new consecutive block
                for i in range(hours_required):
                    chromosome[new_r, new_t_start + i] = events[i]['event_id']

        return chromosome

    def verify_and_repair_course_allocations(self, chromosome):
        """
        A robust method to ensure every required event is scheduled exactly once.
        This function first removes any extra events and then places any missing events.
        """
        max_repair_passes = 3
        for _ in range(max_repair_passes):
            # --- Phase 1: Audit current schedule and identify discrepancies ---
            
            # Get a count of all currently scheduled events
            scheduled_event_counts = {}
            for r_idx in range(len(self.rooms)):
                for t_idx in range(len(self.timeslots)):
                    event_id = chromosome[r_idx, t_idx]
                    if event_id is not None:
                        scheduled_event_counts[event_id] = scheduled_event_counts.get(event_id, 0) + 1

            # --- Phase 2: Remove extra events ---
            
            # Find events that are scheduled more times than they should be (should always be 1)
            extra_event_ids = {event_id for event_id, count in scheduled_event_counts.items() if count > 1}
            
            if extra_event_ids:
                # Create a list of all positions of this duplicated event
                positions_to_clear = []
                for event_id in extra_event_ids:
                    # Find all locations of this duplicated event
                    locations = np.argwhere(chromosome == event_id)
                    # Keep one, mark the rest for removal
                    for i in range(1, len(locations)):
                        positions_to_clear.append(tuple(locations[i]))
                
                # Remove the extra events from the chromosome
                for r_idx, t_idx in positions_to_clear:
                    chromosome[r_idx, t_idx] = None
            
            # --- Phase 3: Add missing events ---

            # Get a fresh set of scheduled events after removals
            scheduled_events = set(np.unique([e for e in chromosome.flatten() if e is not None]))
            
            # Identify all events that are required but not currently in the schedule
            all_required_events = set(range(len(self.events_list)))
            missing_events = list(all_required_events - scheduled_events)
            random.shuffle(missing_events)

            if not missing_events:
                break # If nothing is missing, the repair is done for this pass

            for event_id in missing_events:
                event = self.events_list[event_id]
                course = input_data.getCourse(event.course_id)
                if not course: continue

                placed = False
                
                # Strategy: Find the best possible empty slot that doesn't cause new violations.
                valid_slots = []
                for r_idx, room in enumerate(self.rooms):
                    if self.is_room_suitable(room, course):
                        for t_idx in range(len(self.timeslots)):
                            # Check if slot is empty and if placing the event respects all hard constraints
                            if (chromosome[r_idx, t_idx] is None and
                                self.is_slot_available_for_event(chromosome, r_idx, t_idx, event) and
                                self._is_student_group_available(chromosome, event.student_group.id, t_idx) and
                                self._is_lecturer_available(chromosome, event.faculty_id, t_idx)):
                                valid_slots.append((r_idx, t_idx))
                
                if valid_slots:
                    # Tier 1: If a "perfect" slot is found, place the event there.
                    r, t = random.choice(valid_slots)
                    chromosome[r, t] = event_id
                    placed = True
                else:
                    # Tier 2: If no "perfect" slot is found, find any empty slot in a suitable room.
                    # This prioritizes getting the event on the board, even if it causes other (fixable) violations.
                    imperfect_slots = []
                    for r_idx, room in enumerate(self.rooms):
                        if self.is_room_suitable(room, course):
                            for t_idx in range(len(self.timeslots)):
                                # Just check if the slot is physically empty and respects lecturer schedule/break time
                                if chromosome[r_idx, t_idx] is None and self.is_slot_available_for_event(chromosome, r_idx, t_idx, event):
                                    imperfect_slots.append((r_idx, t_idx))
                    
                    if imperfect_slots:
                        r, t = random.choice(imperfect_slots)
                        chromosome[r, t] = event_id
                        placed = True
                # If still no slot is found (highly unlikely), it will remain missing.

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
best_solution, fitness_history, generation, diversity_history = de.run(50)
print("Differential Evolution completed")

# Get final fitness and detailed breakdown
final_fitness = de.evaluate_fitness(best_solution)
print("\n--- Running Debugging on Final Solution ---")
violations = de.constraints.get_constraint_violations(best_solution, debug=True)
print("--- Debugging Complete ---\n")

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

# Define the desired order for printing constraints
hard_constraint_order = [
    'student_group_constraints',
    'lecturer_availability',
    'lecturer_schedule_constraints',
    'consecutive_timeslots',
    'course_allocation_completeness',
    'same_course_same_room_per_day',
    'room_constraints',
    'room_time_conflict',
    'break_time_constraint'
]

soft_constraint_order = [
    'building_assignments',
    'single_event_per_day',
    'spread_events'
]

# Prepare lines for printing
lines_to_print = {}
max_len = 0

all_constraints_in_violations = [c for c in hard_constraint_order if c in violations] + \
                                [c for c in soft_constraint_order if c in violations]

other_violated_constraints = {k: v for k, v in violations.items() if k not in set(hard_constraint_order + soft_constraint_order) and k != 'total'}
all_constraints_in_violations += sorted(other_violated_constraints.keys())

for constraint in all_constraints_in_violations:
    if constraint in violations:
        points = violations[constraint]
        display_name = descriptive_names.get(constraint, constraint.replace('_', ' ').title())
        
        # Format points to have a consistent look
        if isinstance(points, float):
            # Show 2 decimal places for floats, unless it's a round number
            if points == int(points):
                points_str = str(int(points))
            else:
                points_str = f"{points:.2f}"
        else:
            points_str = str(points)
            
        line_start = f"- {display_name}: {points_str}"
        max_len = max(max_len, len(line_start))
        
        penalty_str = penalty_info.get(constraint, "...")
        lines_to_print[constraint] = (line_start, penalty_str)

print("HARD CONSTRAINTS")
for constraint in hard_constraint_order:
    if constraint in lines_to_print:
        line_start, penalty_str = lines_to_print[constraint]
        # ljust pads the string to the right
        print(f"{line_start.ljust(max_len + 4)}({penalty_str})")

print("\nSOFT CONSTRAINTS")
for constraint in soft_constraint_order:
    if constraint in lines_to_print:
        line_start, penalty_str = lines_to_print[constraint]
        print(f"{line_start.ljust(max_len + 4)}({penalty_str})")

# Print any other constraints that might not be in the main lists
all_printed_constraints = set(hard_constraint_order + soft_constraint_order)
other_constraints_to_print = {k: v for k, v in lines_to_print.items() if k not in all_printed_constraints}

if other_constraints_to_print:
    print("\nOTHER CONSTRAINTS")
    for constraint, (line_start, penalty_str) in sorted(other_constraints_to_print.items()):
        print(f"{line_start.ljust(max_len + 4)}({penalty_str})")

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

# Load rooms data for classroom selection
import json
rooms_data = []
try:
    with open(os.path.join(os.path.dirname(__file__), 'data', 'rooms-data.json'), 'r', encoding='utf-8') as f:
        rooms_data = json.load(f)
    print(f"📚 Loaded {len(rooms_data)} rooms for classroom selection")
except Exception as e:
    print(f"❌ Error loading rooms data: {e}")
    rooms_data = []

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

def has_any_room_conflicts(all_timetables_data):
    """Check if there are any room conflicts across all student groups"""
    if not all_timetables_data:
        return False
    
    # Check each time slot across all groups
    for group_idx in range(len(all_timetables_data)):
        conflicts = detect_room_conflicts(all_timetables_data, group_idx)
        if conflicts:
            return True
    
    return False

def extract_room_from_cell(cell_content):
    """Extract room name from cell content like 'Course: CPE 305, Lecturer: Dr. Smith, Room: Classroom 1'"""
    if not cell_content or cell_content in ["FREE", "BREAK"]:
        return None
    
    # Look for Room: pattern
    import re
    room_match = re.search(r'Room:\s*(.+?)(?:,|$)', cell_content)
    if room_match:
        return room_match.group(1).strip()
    return None

def detect_room_conflicts(all_timetables_data, current_group_idx):
    """Detect room conflicts across all student groups at each time slot"""
    conflicts = {}
    
    if not all_timetables_data:
        return conflicts
    
    # Get all time slots from the current group's timetable
    current_timetable = all_timetables_data[current_group_idx]['timetable']
    
    for row_idx in range(len(current_timetable)):
        for col_idx in range(1, len(current_timetable[row_idx])):  # Skip time column
            timeslot_key = f"{row_idx}_{col_idx-1}"
            room_usage = {}  # room_name -> [(group_idx, group_name, course_info), ...]
            
            # Check all groups for this time slot
            for group_idx, timetable_data in enumerate(all_timetables_data):
                timetable_rows = timetable_data['timetable']
                if row_idx < len(timetable_rows) and col_idx < len(timetable_rows[row_idx]):
                    cell_content = timetable_rows[row_idx][col_idx]
                    room_name = extract_room_from_cell(cell_content)
                    
                    if room_name:
                        if room_name not in room_usage:
                            room_usage[room_name] = []
                        
                        group_name = timetable_data['student_group']['name'] if isinstance(timetable_data['student_group'], dict) else timetable_data['student_group'].name
                        room_usage[room_name].append((group_idx, group_name, cell_content))
            
            # Check for conflicts (same room used by multiple groups)
            for room_name, usage_list in room_usage.items():
                if len(usage_list) > 1:
                    # There's a conflict - mark all groups using this room at this time
                    for group_idx, group_name, cell_content in usage_list:
                        if group_idx == current_group_idx:
                            conflicts[timeslot_key] = {
                                'room': room_name,
                                'conflicting_groups': [u for u in usage_list if u[0] != current_group_idx]
                            }
    
    return conflicts

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
                max-width: 450px;
                width: 90%;
                max-height: 70vh;
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
    
    # Store for rooms data
    dcc.Store(id="rooms-data-store", data=rooms_data),
    
    # Store for communicating swaps
    dcc.Store(id="swap-data", data=None),
    
    # Store for room changes
    dcc.Store(id="room-change-data", data=None),
    
    # Hidden div to trigger the setup
    html.Div(id="trigger", style={"display": "none"}),
    
    # Timetable container
    html.Div(id="timetable-container"),
    
    # Room selection modal (initially hidden)
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
            
            html.Div([                html.Button("Cancel", id="room-cancel-btn", 
                           style={"backgroundColor": "#f5f5f5", "color": "#666", "padding": "8px 16px", 
                                 "border": "1px solid #ddd", "borderRadius": "5px", "marginRight": "10px",
                                 "cursor": "pointer", "fontFamily": "Poppins, sans-serif"}),
                html.Button("Confirm", id="room-confirm-btn", 
                           style={"backgroundColor": "#11214D", "color": "white", "padding": "8px 16px", 
                                 "border": "none", "borderRadius": "5px", "cursor": "pointer",
                                 "fontFamily": "Poppins, sans-serif"})
            ], style={"textAlign": "right", "marginTop": "20px", "paddingTop": "15px", 
                     "borderTop": "1px solid #f0f0f0"})
        ], className="room-selection-modal", id="room-selection-modal", style={"display": "none"})
    ]),
    
    # Conflict warning popup (initially hidden)
    html.Div([
        html.Div([
            html.Span("⚠️ Classroom Conflict", className="conflict-warning-title"),
            html.Button("×", className="conflict-warning-close", id="conflict-close-btn")
        ], className="conflict-warning-header"),
        html.Div(id="conflict-warning-text", className="conflict-warning-content")
    ], className="conflict-warning", id="conflict-warning", style={"display": "none"}),
    
    # Save error popup (initially hidden)
    html.Div([
        html.Div("❌ Cannot Save Timetable", className="save-error-title"),
        html.Div("Please resolve all classroom conflicts before saving the timetable.", className="save-error-content")
    ], className="save-error", id="save-error", style={"display": "none"}),
    
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
    
    # Detect room conflicts across all time slots
    room_conflicts = detect_room_conflicts(all_timetables_data, selected_group_idx)
    
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
            
            # Check for room conflicts
            timeslot_key = f"{row_idx}_{col_idx-1}"
            has_conflict = timeslot_key in room_conflicts
            
            # Determine cell class
            if is_break:
                cell_class = "cell break-time"
                draggable = "false"
            elif has_conflict:
                cell_class = "cell room-conflict"
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
            html.H2(f"Timetable for {student_group_name}", 
                   className="timetable-title",
                   style={"color": "#11214D", "fontWeight": "600", "fontSize": "20px", 
                         "fontFamily": "Poppins, sans-serif", "margin": "0"}),
            html.Div([
                html.Button("‹", className="nav-arrow", id="prev-group-btn",
                           disabled=selected_group_idx == 0),
                html.Button("›", className="nav-arrow", id="next-group-btn", 
                           disabled=selected_group_idx == len(all_timetables_data) - 1)
            ], className="nav-arrows")
        ], className="timetable-header"),
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
        return current_value
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == "prev-group-btn" and current_value > 0:
        return current_value - 1
    elif button_id == "next-group-btn" and current_value < len(all_timetables_data) - 1:
        return current_value + 1
    
    return current_value

# Client-side callback for drag and drop functionality and double-click room selection
clientside_callback(
    """
    function(trigger) {
        console.log('Setting up drag and drop and double-click functionality...');
        
        // Global variables for drag state
        window.draggedElement = null;
        window.dragStartData = null;
        window.selectedCell = null;
        
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
                cell.ondblclick = null;
                
                // Double-click handler for room selection
                cell.ondblclick = function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    
                    console.log('Double-click detected on cell');
                    
                    // Don't allow room selection for break times or free slots
                    if (this.classList.contains('break-time') || 
                        this.textContent.trim() === 'BREAK' || 
                        this.textContent.trim() === 'FREE') {
                        console.log('Cannot select room for break time or free slot');
                        return;
                    }
                    
                    // Store the selected cell
                    window.selectedCell = this;
                    
                    // Show room selection modal
                    const modal = document.getElementById('room-selection-modal');
                    const overlay = document.getElementById('modal-overlay');
                    
                    if (modal && overlay) {
                        modal.style.display = 'block';
                        overlay.style.display = 'block';
                        
                        // Trigger room options loading
                        window.dash_clientside.set_props("room-change-data", {
                            data: {
                                action: 'show_modal',
                                cell_id: this.id,
                                cell_content: this.textContent.trim(),
                                timestamp: Date.now()
                            }
                        });
                        
                        // Focus on search input
                        setTimeout(function() {
                            const searchInput = document.getElementById('room-search-input');
                            if (searchInput) {
                                searchInput.focus();
                            }
                        }, 100);
                    }
                };
                
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
        
        // Setup modal close handlers
        function setupModalHandlers() {
            const modalCloseBtn = document.getElementById('modal-close-btn');
            const roomCancelBtn = document.getElementById('room-cancel-btn');
            const overlay = document.getElementById('modal-overlay');
            const conflictCloseBtn = document.getElementById('conflict-close-btn');
            
            function closeModal() {
                const modal = document.getElementById('room-selection-modal');
                const overlay = document.getElementById('modal-overlay');
                if (modal && overlay) {
                    modal.style.display = 'none';
                    overlay.style.display = 'none';
                }
                window.selectedCell = null;
            }
            
            function closeConflictWarning() {
                const warning = document.getElementById('conflict-warning');
                if (warning) {
                    warning.style.display = 'none';
                }
            }
            
            if (modalCloseBtn) {
                modalCloseBtn.onclick = closeModal;
            }
            
            if (roomCancelBtn) {
                roomCancelBtn.onclick = closeModal;
            }
            
            if (overlay) {
                overlay.onclick = closeModal;
            }
            
            if (conflictCloseBtn) {
                conflictCloseBtn.onclick = closeConflictWarning;
            }
        }
        
        // Setup immediately
        setTimeout(function() {
            setupDragAndDrop();
            setupModalHandlers();
        }, 100);
        
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
                setTimeout(function() {
                    setupDragAndDrop();
                    setupModalHandlers();
                }, 100);
            }
        });
        
        const container = document.getElementById('timetable-container');
        if (container) {
            observer.observe(container, {
                childList: true,
                subtree: true
            });
        }
        
        console.log('Drag and drop and double-click setup complete');
        return window.dash_clientside.no_update;
    }
    """,
    Output("feedback", "style"),
    Input("trigger", "children"),
    prevent_initial_call=False
)

# Client-side callback for room option selection highlighting
clientside_callback(
    """
    function(children) {
        if (!children) return window.dash_clientside.no_update;
        
        // Add click handlers for room options
        setTimeout(function() {
            const roomOptions = document.querySelectorAll('.room-option');
            
            roomOptions.forEach(function(option) {
                option.onclick = function() {
                    // Remove selected class from all options
                    roomOptions.forEach(function(opt) {
                        opt.classList.remove('selected');
                    });
                    
                    // Add selected class to clicked option
                    this.classList.add('selected');
                };
            });
        }, 100);
        
        return window.dash_clientside.no_update;
    }
    """,
    Output("room-options-container", "style"),
    Input("room-options-container", "children"),
    prevent_initial_call=True
)

# Client-side callback to close modal after room confirmation
clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks) {
            // Close the modal
            const modal = document.getElementById('room-selection-modal');
            const overlay = document.getElementById('modal-overlay');
            
            if (modal && overlay) {
                modal.style.display = 'none';
                overlay.style.display = 'none';
            }
            
            // Clear the selected cell
            window.selectedCell = null;
            
            // Show success feedback
            const feedback = document.getElementById('feedback');
            if (feedback) {
                feedback.innerHTML = '✅ Classroom updated successfully';
                feedback.style.color = 'green';
                feedback.style.backgroundColor = '#e8f5e8';
                feedback.style.padding = '10px';
                feedback.style.borderRadius = '5px';
                feedback.style.border = '2px solid #4caf50';
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("room-confirm-btn", "style"),
    Input("room-confirm-btn", "n_clicks"),
    prevent_initial_call=True
)

# Client-side callback to auto-hide save error popup
clientside_callback(
    """
    function(style) {
        if (style && style.display === 'block') {
            // Auto-hide after 4 seconds
            setTimeout(function() {
                const saveError = document.getElementById('save-error');
                if (saveError) {
                    saveError.style.display = 'none';
                }
            }, 4000);
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("save-error", "children"),
    Input("save-error", "style"),
    prevent_initial_call=True
)

# Callback to handle room selection modal and search
@app.callback(
    [Output("room-options-container", "children"),
     Output("room-selection-modal", "style"),
     Output("modal-overlay", "style")],
    [Input("room-change-data", "data"),
     Input("room-search-input", "value")],
    [State("all-timetables-store", "data"),
     State("rooms-data-store", "data"),
     State("student-group-dropdown", "value")],
    prevent_initial_call=True
)
def handle_room_modal(room_change_data, search_value, all_timetables_data, rooms_data, selected_group_idx):
    if not room_change_data or room_change_data.get('action') != 'show_modal':
        return dash.no_update, dash.no_update, dash.no_update
    
    if not rooms_data or not all_timetables_data:
        return html.Div("No rooms data available"), {"display": "block"}, {"display": "block"}
    
    # Parse cell information
    try:
        cell_id_str = room_change_data['cell_id']
        cell_id = json.loads(cell_id_str)
        row_idx = cell_id['row']
        col_idx = cell_id['col']
    except:
        return html.Div("Error parsing cell data"), {"display": "block"}, {"display": "block"}
    
    # Get current room usage for this time slot across all groups
    current_room_usage = get_room_usage_at_timeslot(all_timetables_data, row_idx, col_idx)
    
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
    
    return room_options, {"display": "block"}, {"display": "block"}

def get_room_usage_at_timeslot(all_timetables_data, row_idx, col_idx):
    """Get which rooms are being used at a specific time slot and by which groups"""
    room_usage = {}  # room_name -> [group_names]
    
    for timetable_data in all_timetables_data:
        timetable_rows = timetable_data['timetable']
        if row_idx < len(timetable_rows) and (col_idx + 1) < len(timetable_rows[row_idx]):
            cell_content = timetable_rows[row_idx][col_idx + 1]  # +1 to skip time column
            room_name = extract_room_from_cell(cell_content)
            
            if room_name:
                group_name = timetable_data['student_group']['name'] if isinstance(timetable_data['student_group'], dict) else timetable_data['student_group'].name
                if room_name not in room_usage:
                    room_usage[room_name] = []
                room_usage[room_name].append(group_name)
    
    return room_usage

# Callback to handle room option selection
@app.callback(
    Output("room-change-data", "data", allow_duplicate=True),
    [Input({"type": "room-option", "room_id": ALL, "room_name": ALL}, "n_clicks")],
    [State({"type": "room-option", "room_id": ALL, "room_name": ALL}, "id"),
     State("room-change-data", "data"),
     State("all-timetables-store", "data")],
    prevent_initial_call=True
)
def handle_room_selection(n_clicks_list, room_ids, current_room_data, all_timetables_data):
    if not any(n_clicks_list) or not current_room_data:
        return dash.no_update
    
    # Find which room was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    try:
        triggered_room = json.loads(triggered_id)
        selected_room_name = triggered_room['room_name'];
    except:
        return dash.no_update
    
    # Update the room change data
    return {
        **current_room_data,
        'action': 'room_selected',
        'selected_room': selected_room_name,
        'timestamp': current_room_data.get('timestamp', 0) + 1
    }

# Callback to handle room confirmation and update timetable
@app.callback(
    [Output("all-timetables-store", "data", allow_duplicate=True),
     Output("conflict-warning", "style"),
     Output("conflict-warning-text", "children")],
    [Input("room-confirm-btn", "n_clicks")],
    [State("room-change-data", "data"),
     State("all-timetables-store", "data"),
     State("student-group-dropdown", "value"),
     State("rooms-data-store", "data")],
    prevent_initial_call=True
)
def confirm_room_change(n_clicks, room_change_data, all_timetables_data, selected_group_idx, rooms_data):
    if not n_clicks or not room_change_data or room_change_data.get('action') != 'room_selected':
        return dash.no_update, dash.no_update, dash.no_update
    
    global session_has_swaps
    session_has_swaps = True
    
    try:
        # Parse cell information
        cell_id_str = room_change_data['cell_id']
        cell_id = json.loads(cell_id_str)
        row_idx = cell_id['row']
        col_idx = cell_id['col']
        selected_room = room_change_data['selected_room']
        
        # Update the current group's timetable
        updated_timetables = json.loads(json.dumps(all_timetables_data))  # Deep copy
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
        
        # Check for conflicts after the update
        room_usage = get_room_usage_at_timeslot(updated_timetables, row_idx, col_idx)
        conflict_warning_style = {"display": "none"}
        conflict_warning_text = ""
        
        if selected_room in room_usage and len(room_usage[selected_room]) > 1:
            # There's a conflict
            other_groups = [group for group in room_usage[selected_room] 
                           if group != updated_timetables[selected_group_idx]['student_group']['name']]
            
            conflict_warning_style = {"display": "block"}
            conflict_warning_text = f"This classroom is already in use by: {', '.join(other_groups)}"
        
        # Auto-save the changes
        save_timetable_to_file(updated_timetables)
        
        return updated_timetables, conflict_warning_style, conflict_warning_text
        
    except Exception as e:
        print(f"Error in room change: {e}")
        return dash.no_update, {"display": "block"}, f"Error updating room: {str(e)}"

def extract_course_code_from_cell(cell_content):
    """Extract course code from cell content"""
    if not cell_content or cell_content in ["FREE", "BREAK"]:
        return None
    
    # Look for Course: pattern
    import re
    course_match = re.search(r'Course:\s*([^,]+)', cell_content)
    if course_match:
        return course_match.group(1).strip()
    return None

def update_room_in_cell_content(cell_content, new_room):
    """Update the room name in cell content"""
    if not cell_content or cell_content in ["FREE", "BREAK"]:
        return cell_content
    
    import re
    # Replace the room part
    updated_content = re.sub(r'Room:\s*[^,]*', f'Room: {new_room}', cell_content)
    return updated_content

def update_consecutive_course_rooms(timetables_data, group_idx, course_code, new_room, current_row, current_col):
    """Update consecutive classes of the same course to use the same room"""
    timetable_rows = timetables_data[group_idx]['timetable']
    
    # Check same day (same column) for consecutive hours
    for row_idx in range(len(timetable_rows)):
        if row_idx != current_row:  # Skip the cell we just updated
            cell_content = timetable_rows[row_idx][current_col + 1]
            if extract_course_code_from_cell(cell_content) == course_code:
                # This is the same course, update the room
                updated_content = update_room_in_cell_content(cell_content, new_room)
                timetable_rows[row_idx][current_col + 1] = updated_content

def save_timetable_to_file(timetables_data):
    """Save timetable data to file"""
    try:
        save_path = os.path.join(os.path.dirname(__file__), 'data', 'timetable_data.json')
        backup_path = os.path.join(os.path.dirname(__file__), 'data', 'timetable_data_backup.json')
        
        # Create backup
        if os.path.exists(save_path):
            shutil.copy2(save_path, backup_path)
        
        # Save new data
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(timetables_data, f, indent=2, ensure_ascii=False)
        
        # Force file system sync
        if hasattr(os, 'fsync'):
            f.flush()
            os.fsync(f.fileno())
        
        print(f"✅ Auto-saved timetable changes")
        return True
        
    except Exception as e:
        print(f"❌ Error auto-saving timetable: {e}")
        return False
@app.callback(
    Output("all-timetables-store", "data", allow_duplicate=True),
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
                print(f"[SUCCESS] Auto-saved successfully! File size: {file_size} bytes")
                
                # Verify we can read it back and check the specific swapped data
                with open(save_path, 'r', encoding='utf-8') as f:
                    test_load = json.load(f)
                print(f"[VERIFICATION] Can read back {len(test_load)} student groups")
                
                # Verify the actual swap was saved
                saved_source = test_load[group_idx]['timetable'][source['row']][source['col'] + 1]
                saved_target = test_load[group_idx]['timetable'][target['row']][target['col'] + 1]
                print(f"[VERIFICATION] Saved source = '{saved_source}', target = '{saved_target}'")
                
            else:
                print("[ERROR] Auto-save file was not created!")
                
        except Exception as auto_save_error:
            print(f"[ERROR] Auto-save failed with error: {auto_save_error}")
            import traceback
            traceback.print_exc()
        
        return current_timetables
        
    except Exception as e:
        print(f"Error handling swap: {e}")
        return current_timetables

# Callback to handle saving the current timetable state
@app.callback(
    [Output("save-status", "children"),
     Output("save-error", "style")],
    Input("save-button", "n_clicks"),
    State("all-timetables-store", "data"),
    prevent_initial_call=True
)
def save_timetable(n_clicks, current_timetables):
    if n_clicks and current_timetables:
        # Check for room conflicts before saving
        if has_any_room_conflicts(current_timetables):
            # Show error popup and prevent saving
            return (
                html.Div([
                    html.Span("❌ Cannot save - resolve conflicts first", 
                             style={"color": "#d32f2f", "fontWeight": "bold"})
                ]),
                {"display": "block"}
            )
        
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
            
            return (
                html.Div([
                    html.Span("✅ Timetable saved successfully!", style={"color": "green", "fontWeight": "bold"}),
                    html.Br(),
                    html.Small(f"Saved to: {save_path}", style={"color": "gray"})
                ]),
                {"display": "none"}
            )
            
        except Exception as e:
            print(f"Error saving timetable: {e}")
            return (
                html.Div([
                    html.Span("❌ Error saving timetable!", style={"color": "red", "fontWeight": "bold"}),
                    html.Br(),
                        html.Small(f"Error: {str(e)}", style={"color": "red"})
                ]),
                {"display": "none"}
            )
    
    return ("", {"display": "none"})

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