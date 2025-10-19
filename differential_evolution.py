# differential_evolution.py - Refactored as a callable backend engine

import random
from typing import List
import copy
from utils import Utility
from entitities.Class import Class
# from input_data import input_data ### MODIFICATION: This static import is no longer needed
import numpy as np
from constraints import Constraints
import re
import dash
from dash import dcc, html, Input, Output, State
from dash.dependencies import ALL
import json
import os
import shutil
import traceback

# --- The DifferentialEvolution class and all its methods remain exactly the same as you provided ---
# (The full class code is included for completeness)

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
        
        self.fitness_cache = {}
        
        self.room_building_cache = {}
        for idx, room in enumerate(self.rooms):
            self.room_building_cache[idx] = self.get_room_building(room)
        
        self.engineering_groups = set()
        for student_group in self.student_groups:
            group_name = student_group.name.lower()
            if any(keyword in group_name for keyword in [
                'engineering', 'eng', 'computer science', 'software engineering', 'data science',
                'mechatronics', 'electrical', 'mechanical', 'csc', 'sen', 'data', 'ds'
            ]):
                self.engineering_groups.add(student_group.id)
        
        self.population = self.initialize_population()

    def create_events(self):
        events_list = []
        event_map = {}

        idx = 0
        for student_group in self.student_groups:
            for i in range(student_group.no_courses):
                course = self.input_data.getCourse(student_group.courseIDs[i])
                
                if course and course.credits == 1:
                    required_hours = 3
                else:
                    required_hours = student_group.hours_required[i]
                
                hourcount = 1 
                while hourcount <= required_hours:
                    event = Class(student_group, student_group.teacherIDS[i], student_group.courseIDs[i])
                    events_list.append(event)
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
        
        events_by_group_course = {}
        for idx, event in enumerate(self.events_list):
            key = (event.student_group.id, event.course_id)
            if key not in events_by_group_course:
                events_by_group_course[key] = []
            events_by_group_course[key].append(idx)

        course_items = sorted(
            events_by_group_course.items(),
            key=lambda item: len(item[1]),
            reverse=True
        )

        hours_per_day_for_group = {sg.id: [0] * self.input_data.days for sg in self.student_groups}

        for (student_group_id, course_id), event_indices in course_items:
            course = self.input_data.getCourse(course_id)
            student_group = self.input_data.getStudentGroup(student_group_id)
            hours_required = len(event_indices)

            if hours_required == 0:
                continue

            split_strategies = []
            if hours_required >= 4:
                split_strategies = [(4,), (2, 2), (3, 1)]
            elif hours_required == 3:
                split_strategies = [(3,), (2, 1)]
            elif hours_required == 2:
                split_strategies = [(2,)]
            else:
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
                    
                    available_days = [d for d in range(self.input_data.days) if d not in temp_course_days_used]
                    sorted_days = sorted(available_days, key=lambda d: temp_hours_per_day[d])

                    for day_idx in sorted_days:
                        day_start = day_idx * self.input_data.hours
                        day_end = (day_idx + 1) * self.input_data.hours
                        
                        possible_slots = []
                        for room_idx, room in enumerate(self.rooms):
                            is_engineering_group = student_group.id in self.engineering_groups
                            room_building = self.room_building_cache.get(room_idx, 'UNKNOWN')
                            
                            if not is_engineering_group and room_building == 'SST':
                                continue

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
                            preferred_slots = []
                            for r_idx, t_start in possible_slots:
                                room_bldg = self.room_building_cache.get(r_idx, 'UNKNOWN')
                                is_eng_grp = student_group.id in self.engineering_groups
                                if is_eng_grp and room_bldg != 'SST':
                                    pass
                                else:
                                    preferred_slots.append((r_idx, t_start))
                            
                            if preferred_slots:
                                room_idx, timeslot_start = random.choice(preferred_slots)
                            else:
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

        chromosome = self.verify_and_repair_course_allocations(chromosome)
        chromosome = self.prevent_student_group_clashes(chromosome)
        
        return chromosome
    
    # ... (all other methods from find_consecutive_slots to diagnose_course_allocations remain unchanged) ...
    def find_consecutive_slots(self, chromosome, course):
        two_slot_rooms = []
        for room_idx, room in enumerate(self.rooms):
            if self.is_room_suitable(room, course):
                for i in range(len(self.timeslots) - 1):
                    if self.is_slot_available(chromosome, room_idx, i) and self.is_slot_available(chromosome, room_idx, i + 1):
                        two_slot_rooms.append((room_idx, i, i+1))
        if len(two_slot_rooms) != 0:
            _room_idx, slot1, slot2 = random.choice(two_slot_rooms)           
            return _room_idx, slot1, slot2
        return None, None, None

    def find_single_slot(self, chromosome, course):
        single_slot_rooms = []
        for room_idx, room in enumerate(self.rooms):
            if self.is_room_suitable(room, course):
                for i in range(len(self.timeslots)):
                    if self.is_slot_available(chromosome, room_idx, i):
                        single_slot_rooms.append((room_idx, i))
        if len(single_slot_rooms) > 0:
            return random.choice(single_slot_rooms)
        return None, None

    def is_slot_available(self, chromosome, room_idx, timeslot_idx):
        if chromosome[room_idx][timeslot_idx] is not None:
            return False
        break_hour = 4
        day = timeslot_idx // self.input_data.hours
        hour_in_day = timeslot_idx % self.input_data.hours
        if hour_in_day == break_hour and day not in [1, 3]:
            return False
        return True

    def is_slot_available_for_event(self, chromosome, room_idx, timeslot_idx, event):
        if chromosome[room_idx][timeslot_idx] is not None:
            return False
        break_hour = 4
        day = timeslot_idx // self.input_data.hours
        hour_in_day = timeslot_idx % self.input_data.hours
        if hour_in_day == break_hour and day not in [1, 3]:
            return False
        if event and event.faculty_id is not None:
            faculty = self.input_data.getFaculty(event.faculty_id)
            if faculty:
                days_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
                day_abbr = days_map.get(day)
                slot_hour = 9 + (timeslot_idx % self.input_data.hours)
                is_day_ok = False
                avail_days = faculty.avail_days
                if not avail_days or (isinstance(avail_days, str) and avail_days.upper() == "ALL"):
                    is_day_ok = True
                else:
                    avail_days_list = [d.strip().capitalize() for d in (avail_days.split(',') if isinstance(avail_days, str) else avail_days)]
                    if "All" in avail_days_list or day_abbr in avail_days_list:
                        is_day_ok = True
                if not is_day_ok:
                    return False
                is_time_ok = False
                avail_times = faculty.avail_times
                if not avail_times or (isinstance(avail_times, str) and avail_times.upper() == "ALL") or (isinstance(avail_times, list) and any(str(t).strip().upper() == 'ALL' for t in avail_times)):
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
                            except (ValueError, IndexError): continue
                        else:
                            try:
                                h = int(time_spec_str.split(':')[0])
                                if h == slot_hour:
                                    is_time_ok = True
                                    break
                            except (ValueError, IndexError): continue
                if not is_time_ok:
                    return False
        return True

    def is_room_suitable(self, room, course):
        if course is None: return False
        return room.room_type == course.required_room_type
    
    def get_room_building(self, room):
        if hasattr(room, 'building'): return room.building.upper()
        elif hasattr(room, 'name') and room.name:
            room_name = room.name.upper()
            if 'SST' in room_name: return 'SST'
            elif 'TYD' in room_name: return 'TYD'
        elif hasattr(room, 'room_id'):
            room_id = str(room.room_id).upper()
            if 'SST' in room_id: return 'SST'
            elif 'TYD' in room_id: return 'TYD'
        return 'UNKNOWN'

    def _is_student_group_available(self, chromosome, student_group_id, timeslot_idx):
        for r_idx in range(len(self.rooms)):
            event_id = chromosome[r_idx][timeslot_idx]
            if event_id is not None:
                event = self.events_map.get(event_id)
                if event and event.student_group.id == student_group_id:
                    return False
        return True

    def _is_lecturer_available(self, chromosome, faculty_id, timeslot_idx):
        for r_idx in range(len(self.rooms)):
            event_id = chromosome[r_idx][timeslot_idx]
            if event_id is not None:
                event = self.events_map.get(event_id)
                if event and event.faculty_id == faculty_id:
                    return False
        return True

    def find_clash(self, chromosome):
        clash_slots = []
        for t_idx in range(len(self.timeslots)):
            student_group_watch, lecturer_watch = set(), set()
            has_student_clash, has_lecturer_clash = False, False
            event_ids_in_slot = [e for e in chromosome[:, t_idx] if e is not None]
            if len(event_ids_in_slot) <= 1: continue
            for event_id in event_ids_in_slot:
                event = self.events_map.get(event_id)
                if not event: continue
                if event.student_group.id in student_group_watch: has_student_clash = True
                student_group_watch.add(event.student_group.id)
                if event.faculty_id and event.faculty_id in lecturer_watch: has_lecturer_clash = True
                if event.faculty_id: lecturer_watch.add(event.faculty_id)
            if has_student_clash or has_lecturer_clash: clash_slots.append(t_idx)
        return random.choice(clash_slots) if clash_slots else None
    
    def hamming_distance(self, chromosome1, chromosome2):
        return np.sum(chromosome1.flatten() != chromosome2.flatten())

    def calculate_population_diversity(self):
        if self.pop_size <= 10:
            total_distance = sum(self.hamming_distance(self.population[i], self.population[j]) for i in range(self.pop_size) for j in range(i + 1, self.pop_size))
            comparisons = self.pop_size * (self.pop_size - 1) / 2
            return total_distance / comparisons if comparisons > 0 else 0
        else:
            total_distance = sum(self.hamming_distance(self.population[i], self.population[j]) for _ in range(10) for i, j in [random.sample(range(self.pop_size), 2)])
            return total_distance / 10

    def mutate(self, target_idx):
        mutant_vector = self.population[target_idx].copy()
        for _ in range(random.randint(3, 8)):
            strategy = random.choice(['resolve_clash', 'safe_swap', 'safe_move'])
            if strategy == 'resolve_clash':
                clash_timeslot = self.find_clash(mutant_vector)
                if clash_timeslot is not None:
                    events_in_slot = [(r, mutant_vector[r, clash_timeslot]) for r in range(len(self.rooms)) if mutant_vector[r, clash_timeslot] is not None]
                    if not events_in_slot: continue
                    room_to_move_from, event_id_to_move = random.choice(events_in_slot)
                    event_to_move = self.events_map.get(event_id_to_move)
                    if not event_to_move: continue
                    new_pos = self.find_safe_empty_slot_for_event(mutant_vector, event_to_move, ignore_pos=(room_to_move_from, clash_timeslot))
                    if new_pos:
                        new_r, new_t = new_pos
                        mutant_vector[new_r, new_t] = event_id_to_move
                        mutant_vector[room_to_move_from, clash_timeslot] = None
            elif strategy == 'safe_swap':
                occupied_slots = np.argwhere(mutant_vector != None)
                if len(occupied_slots) < 2: continue
                idx1, idx2 = random.sample(range(len(occupied_slots)), 2)
                pos1, pos2 = tuple(occupied_slots[idx1]), tuple(occupied_slots[idx2])
                event1_id, event2_id = mutant_vector[pos1], mutant_vector[pos2]
                event1, event2 = self.events_map.get(event1_id), self.events_map.get(event2_id)
                if not event1 or not event2: continue
                course1, course2 = self.input_data.getCourse(event1.course_id), self.input_data.getCourse(event2.course_id)
                room1_ok_for_event2 = self.is_room_suitable(self.rooms[pos1[0]], course2)
                room2_ok_for_event1 = self.is_room_suitable(self.rooms[pos2[0]], course1)
                if room1_ok_for_event2 and room2_ok_for_event1:
                    clash_free_at_pos1 = not self.constraints.check_student_group_clash_at_slot(mutant_vector, event2.student_group.id, pos1[1], ignore_room_idx=pos2[0]) and not self.constraints.check_lecturer_clash_at_slot(mutant_vector, event2.faculty_id, pos1[1], ignore_room_idx=pos2[0])
                    clash_free_at_pos2 = not self.constraints.check_student_group_clash_at_slot(mutant_vector, event1.student_group.id, pos2[1], ignore_room_idx=pos1[0]) and not self.constraints.check_lecturer_clash_at_slot(mutant_vector, event1.faculty_id, pos2[1], ignore_room_idx=pos1[0])
                    if clash_free_at_pos1 and clash_free_at_pos2:
                        mutant_vector[pos1], mutant_vector[pos2] = event2_id, event1_id
            elif strategy == 'safe_move':
                occupied_slots = np.argwhere(mutant_vector != None)
                if not len(occupied_slots): continue
                random_index = random.randrange(len(occupied_slots))
                pos_to_move = tuple(occupied_slots[random_index])
                event_id_to_move = mutant_vector[pos_to_move]
                event_to_move = self.events_map.get(event_id_to_move)
                if not event_to_move: continue
                new_pos = self.find_safe_empty_slot_for_event(mutant_vector, event_to_move, ignore_pos=pos_to_move)
                if new_pos:
                    new_r, new_t = new_pos
                    mutant_vector[new_r, new_t] = event_id_to_move
                    mutant_vector[pos_to_move] = None
        return mutant_vector

    def find_safe_empty_slot_for_event(self, chromosome, event, ignore_pos=None):
        course = self.input_data.getCourse(event.course_id)
        if not course: return None
        possible_slots = []
        for r_idx, room in enumerate(self.rooms):
            if self.is_room_suitable(room, course):
                for t_idx in range(len(self.timeslots)):
                    if (r_idx, t_idx) == ignore_pos: continue
                    if chromosome[r_idx, t_idx] is None and self.is_slot_available_for_event(chromosome, r_idx, t_idx, event):
                        student_clash = self.constraints.check_student_group_clash_at_slot(chromosome, event.student_group.id, t_idx, ignore_room_idx=ignore_pos[0] if ignore_pos else -1)
                        lecturer_clash = self.constraints.check_lecturer_clash_at_slot(chromosome, event.faculty_id, t_idx, ignore_room_idx=ignore_pos[0] if ignore_pos else -1)
                        if not student_clash and not lecturer_clash:
                            possible_slots.append((r_idx, t_idx))
        return random.choice(possible_slots) if possible_slots else None

    def ensure_valid_solution(self, mutant_vector):
        course_day_room_mapping = {}
        for room_idx in range(len(self.rooms)):
            for timeslot_idx in range(len(self.timeslots)):
                event_id = mutant_vector[room_idx][timeslot_idx]
                if event_id is not None:
                    class_event = self.events_map.get(event_id)
                    if class_event:
                        day_idx = timeslot_idx // self.input_data.hours
                        course = self.input_data.getCourse(class_event.course_id)
                        course_id = getattr(course, 'course_id', None) or getattr(course, 'id', None) or getattr(course, 'code', None) if course else class_event.course_id
                        course_day_key = (course_id, day_idx, class_event.student_group.id)
                        if course_day_key not in course_day_room_mapping:
                            course_day_room_mapping[course_day_key] = room_idx
        events_to_move = []
        for room_idx in range(len(self.rooms)):
            for timeslot_idx in range(len(self.timeslots)):
                event_id = mutant_vector[room_idx][timeslot_idx]
                if event_id is not None:
                    class_event = self.events_map.get(event_id)
                    if class_event:
                        day_idx = timeslot_idx // self.input_data.hours
                        course = self.input_data.getCourse(class_event.course_id)
                        course_id = getattr(course, 'course_id', None) or getattr(course, 'id', None) or getattr(course, 'code', None) if course else class_event.course_id
                        course_day_key = (course_id, day_idx, class_event.student_group.id)
                        expected_room = course_day_room_mapping.get(course_day_key)
                        if expected_room is not None and room_idx != expected_room:
                            mutant_vector[room_idx][timeslot_idx] = None
                            events_to_move.append((event_id, expected_room, timeslot_idx))
        for event_id, correct_room, original_timeslot in events_to_move:
            placed = False
            for timeslot in range(len(self.timeslots)):
                if self.is_slot_available(mutant_vector, correct_room, timeslot):
                    mutant_vector[correct_room][timeslot] = event_id
                    placed = True
                    break
        mutant_vector = self.verify_and_repair_course_allocations(mutant_vector)
        mutant_vector = self.ensure_consecutive_slots(mutant_vector)
        mutant_vector = self.prevent_student_group_clashes(mutant_vector)
        return mutant_vector

    def prevent_student_group_clashes(self, chromosome):
        for _ in range(5):
            clashes_found = False
            for t_idx in range(len(self.timeslots)):
                student_groups_seen, conflicting_events = {}, []
                for r_idx in range(len(self.rooms)):
                    event_id = chromosome[r_idx, t_idx]
                    if event_id is not None:
                        event = self.events_map.get(event_id)
                        if event:
                            sg_id = event.student_group.id
                            if sg_id in student_groups_seen:
                                conflicting_events.append((r_idx, event_id))
                                clashes_found = True
                            else:
                                student_groups_seen[sg_id] = (r_idx, event_id)
                for r_idx, event_id in conflicting_events:
                    event, course = self.events_map.get(event_id), self.input_data.getCourse(self.events_map.get(event_id).course_id)
                    chromosome[r_idx, t_idx] = None
                    moved, alternative_slots = False, []
                    for alt_r_idx, room in enumerate(self.rooms):
                        if self.is_room_suitable(room, course):
                            for alt_t_idx in range(len(self.timeslots)):
                                if (chromosome[alt_r_idx, alt_t_idx] is None and self.is_slot_available_for_event(chromosome, alt_r_idx, alt_t_idx, event) and self._is_student_group_available(chromosome, event.student_group.id, alt_t_idx) and self._is_lecturer_available(chromosome, event.faculty_id, alt_t_idx)):
                                    alternative_slots.append((alt_r_idx, alt_t_idx))
                    if alternative_slots:
                        alt_r, alt_t = random.choice(alternative_slots)
                        chromosome[alt_r, alt_t] = event_id
                        moved = True
                    else:
                        for alt_r_idx, room in enumerate(self.rooms):
                            if self.is_room_suitable(room, course):
                                for alt_t_idx in range(len(self.timeslots)):
                                    if (chromosome[alt_r_idx, alt_t_idx] is None and self.is_slot_available_for_event(chromosome, alt_r_idx, alt_t_idx, event)):
                                        chromosome[alt_r_idx, alt_t_idx] = event_id
                                        moved = True
                                        break
                                if moved: break
            if not clashes_found: break
        return chromosome

    def verify_no_student_group_clashes(self, chromosome):
        for t_idx in range(len(self.timeslots)):
            student_groups_seen = set()
            for r_idx in range(len(self.rooms)):
                event_id = chromosome[r_idx, t_idx]
                if event_id is not None:
                    event = self.events_map.get(event_id)
                    if event:
                        sg_id = event.student_group.id
                        if sg_id in student_groups_seen:
                            return False
                        student_groups_seen.add(sg_id)
        return True

    def count_non_none(self, arr):
        return np.count_nonzero(arr != None)
    
    def crossover(self, target_vector, mutant_vector):
        trial_vector = target_vector.copy()
        num_rooms, num_timeslots = target_vector.shape
        j_rand_r, j_rand_t = random.randrange(num_rooms), random.randrange(num_timeslots)
        for r in range(num_rooms):
            for t in range(num_timeslots):
                if random.random() < self.CR or (r == j_rand_r and t == j_rand_t):
                    mutant_event_id = mutant_vector[r, t]
                    if mutant_event_id is None:
                        trial_vector[r, t] = None
                    else:
                        mutant_event = self.events_map.get(mutant_event_id)
                        if mutant_event and self._is_student_group_available(trial_vector, mutant_event.student_group.id, t):
                            trial_vector[r, t] = mutant_event_id
        return trial_vector
    
    def evaluate_fitness(self, chromosome):
        chromosome_key = str(chromosome.tobytes())
        if chromosome_key in self.fitness_cache: return self.fitness_cache[chromosome_key]
        fitness = self.constraints.evaluate_fitness(chromosome)
        if len(self.fitness_cache) > 1000:
            for key in list(self.fitness_cache.keys())[:-500]: del self.fitness_cache[key]
        self.fitness_cache[chromosome_key] = fitness
        return fitness

    def check_room_constraints(self, chromosome):
        point = 0
        for room_idx, room in enumerate(self.rooms):
            for timeslot_idx in range(len(self.timeslots)):
                class_event = self.events_map.get(chromosome[room_idx][timeslot_idx])
                if class_event is not None:
                    course = self.input_data.getCourse(class_event.course_id)
                    if room.room_type != course.required_room_type or class_event.student_group.no_students > room.capacity:
                        point += 1
        return point
    
    def check_student_group_constraints(self, chromosome):
        penalty = 0
        for i in range(len(self.timeslots)):
            student_group_watch = set()
            for class_event_idx in chromosome[:, i]:
                if class_event_idx is not None:
                    class_event = self.events_map.get(class_event_idx)
                    student_group = class_event.student_group
                    if student_group.id in student_group_watch: penalty += 1
                    else: student_group_watch.add(student_group.id)
        return penalty
    
    def check_lecturer_availability(self, chromosome):
        penalty = 0
        for i in range(len(self.timeslots)):
            lecturer_watch = set()
            for class_event_idx in chromosome[:, i]:
                if class_event_idx is not None:
                    class_event = self.events_map.get(class_event_idx)
                    faculty_id = class_event.faculty_id
                    if faculty_id in lecturer_watch: penalty += 1
                    else: lecturer_watch.add(faculty_id)
        return penalty

    def select(self, target_idx, trial_vector):
        trial_violations = self.constraints.get_constraint_violations(trial_vector)
        target_violations = self.constraints.get_constraint_violations(self.population[target_idx])
        hard_constraints = ['student_group_constraints', 'lecturer_availability', 'course_allocation_completeness', 'room_time_conflict', 'break_time_constraint', 'room_constraints', 'same_course_same_room_per_day', 'lecturer_schedule_constraints', 'lecturer_workload_constraints']
        trial_hard_violations = sum(trial_violations.get(c, 0) for c in hard_constraints)
        target_hard_violations = sum(target_violations.get(c, 0) for c in hard_constraints)
        accept = False
        if trial_hard_violations < target_hard_violations or (trial_hard_violations == target_hard_violations and trial_violations.get('total', float('inf')) <= target_violations.get('total', float('inf'))):
            accept = True
        if accept:
            self.population[target_idx] = trial_vector

    def run(self, max_generations):
        fitness_history, diversity_history = [], []
        best_solution = self.population[0]
        initial_fitness = [self.evaluate_fitness(ind) for ind in self.population]
        best_idx = np.argmin(initial_fitness)
        best_solution = self.population[best_idx].copy()
        best_fitness = initial_fitness[best_idx]
        stagnation_counter, last_improvement = 0, best_fitness

        for generation in range(max_generations):
            for i in range(self.pop_size):
                mutant_vector = self.mutate(i)
                trial_vector = self.crossover(self.population[i], mutant_vector)
                trial_vector = self.verify_and_repair_course_allocations(trial_vector)
                trial_vector = self.prevent_student_group_clashes(trial_vector)
                trial_vector = self.verify_and_repair_course_allocations(trial_vector)
                self.select(i, trial_vector)
            current_fitness = [self.evaluate_fitness(ind) for ind in self.population]
            current_best_idx = np.argmin(current_fitness)
            current_best_fitness = current_fitness[current_best_idx]
            if current_best_fitness < best_fitness:
                best_solution = self.population[current_best_idx].copy()
                best_fitness = current_best_fitness
                stagnation_counter, last_improvement = 0, best_fitness
            else:
                stagnation_counter += 1
            fitness_history.append(best_fitness)
            if generation % 20 == 0:
                diversity_history.append(self.calculate_population_diversity())
            if best_fitness == self.desired_fitness or stagnation_counter >= 20 or (stagnation_counter > 50 and best_fitness < 100):
                break
        
        best_solution = self.verify_and_repair_course_allocations(best_solution)
        best_solution = self.ensure_consecutive_slots(best_solution)
        best_solution = self.prevent_student_group_clashes(best_solution)
        best_solution = self.verify_and_repair_course_allocations(best_solution)
        
        return best_solution, fitness_history, generation, diversity_history

    def print_timetable(self, individual, student_group, days, hours_per_day, day_start_time=9):
        timetable = [["" for _ in range(days)] for _ in range(hours_per_day)]
        break_hour = 4
        if break_hour < hours_per_day:
            for day in range(days):
                if day in [0, 2, 4]: timetable[break_hour][day] = "BREAK"
        for room_idx, room_slots in enumerate(individual):
            for timeslot_idx, event in enumerate(room_slots):
                class_event = self.events_map.get(event)
                if class_event is not None and class_event.student_group.id == student_group.id:
                    day, hour = timeslot_idx // hours_per_day, timeslot_idx % hours_per_day
                    if not (hour == break_hour and day in [0, 2, 4]):
                        course = self.input_data.getCourse(class_event.course_id)
                        faculty = self.input_data.getFaculty(class_event.faculty_id)
                        course_code = course.code if course is not None else "Unknown"
                        faculty_display = faculty.name if faculty and faculty.name else (faculty.faculty_id if faculty else "Unknown")
                        room_obj = self.input_data.rooms[room_idx]
                        room_display = getattr(room_obj, "name", getattr(room_obj, "Id", str(room_idx)))
                        timetable[hour][day] = f"{course_code}\n{room_display}\n{faculty_display}"
        return timetable

    def print_all_timetables(self, individual, days, hours_per_day, day_start_time=9):
        data = []
        for student_group in self.input_data.student_groups:
            timetable = self.print_timetable(individual, student_group, days, hours_per_day, day_start_time)
            rows = [[f"{day_start_time + hour}:00"] + [timetable[hour][day] for day in range(days)] for hour in range(hours_per_day)]
            data.append({"student_group": student_group, "timetable": rows})
        return data

    def ensure_consecutive_slots(self, chromosome):
        events_by_course = {}
        for r_idx in range(len(self.rooms)):
            for t_idx in range(len(self.timeslots)):
                event_id = chromosome[r_idx, t_idx]
                if event_id is not None:
                    event = self.events_map.get(event_id)
                    if event:
                        course_key = (event.student_group.id, event.course_id)
                        if course_key not in events_by_course: events_by_course[course_key] = []
                        events_by_course[course_key].append({'event_id': event_id, 'pos': (r_idx, t_idx)})
        for course_key, events in events_by_course.items():
            if len(events) < 2: continue
            positions = sorted([e['pos'] for e in events], key=lambda p: p[1])
            is_consecutive = all(positions[i][0] == positions[i+1][0] and positions[i][1] + 1 == positions[i+1][1] for i in range(len(positions) - 1))
            if is_consecutive: continue
            student_group_id, course_id = course_key
            course = self.input_data.getCourse(course_id)
            possible_blocks = []
            for r_idx, room in enumerate(self.rooms):
                if self.is_room_suitable(room, course):
                    for t_start in range(len(self.timeslots) - len(events) + 1):
                        is_block_valid = True
                        temp_chromosome = chromosome.copy()
                        for event_info in events: temp_chromosome[event_info['pos']] = None
                        for i in range(len(events)):
                            t_check = t_start + i
                            event_to_place = self.events_list[events[i]['event_id']]
                            if not (self.is_slot_available_for_event(temp_chromosome, r_idx, t_check, event_to_place) and self._is_student_group_available(temp_chromosome, student_group_id, t_check) and self._is_lecturer_available(temp_chromosome, event_to_place.faculty_id, t_check)):
                                is_block_valid = False
                                break
                        if is_block_valid: possible_blocks.append((r_idx, t_start))
            if possible_blocks:
                new_r, new_t_start = random.choice(possible_blocks)
                for event_info in events: chromosome[event_info['pos']] = None
                for i in range(len(events)): chromosome[new_r, new_t_start + i] = events[i]['event_id']
        return chromosome

    def verify_and_repair_course_allocations(self, chromosome):
        for _ in range(5):
            scheduled_event_counts = {}
            for event_id in chromosome.flatten():
                if event_id is not None: scheduled_event_counts[event_id] = scheduled_event_counts.get(event_id, 0) + 1
            extra_event_ids = {event_id for event_id, count in scheduled_event_counts.items() if count > 1}
            if extra_event_ids:
                for event_id in extra_event_ids:
                    locations = np.argwhere(chromosome == event_id)
                    for i in range(1, len(locations)): chromosome[tuple(locations[i])] = None
            scheduled_events = set(np.unique([e for e in chromosome.flatten() if e is not None]))
            missing_events = list(set(range(len(self.events_list))) - scheduled_events)
            if not missing_events: break
            random.shuffle(missing_events)
            for event_id in missing_events:
                event, course = self.events_list[event_id], self.input_data.getCourse(self.events_list[event_id].course_id)
                if not course: continue
                placed = False
                perfect_slots = []
                for r_idx, room in enumerate(self.rooms):
                    if self.is_room_suitable(room, course):
                        for t_idx in range(len(self.timeslots)):
                            if (chromosome[r_idx, t_idx] is None and self.is_slot_available_for_event(chromosome, r_idx, t_idx, event) and self._is_student_group_available(chromosome, event.student_group.id, t_idx) and self._is_lecturer_available(chromosome, event.faculty_id, t_idx)):
                                perfect_slots.append((r_idx, t_idx))
                if perfect_slots:
                    r, t = random.choice(perfect_slots)
                    chromosome[r, t] = event_id
                    continue
                acceptable_slots = []
                for r_idx, room in enumerate(self.rooms):
                    if self.is_room_suitable(room, course):
                        for t_idx in range(len(self.timeslots)):
                            if (chromosome[r_idx, t_idx] is None and self.is_slot_available_for_event(chromosome, r_idx, t_idx, event)):
                                acceptable_slots.append((r_idx, t_idx))
                if acceptable_slots:
                    r, t = random.choice(acceptable_slots)
                    chromosome[r, t] = event_id
                    continue
                for r_idx, room in enumerate(self.rooms):
                    if self.is_room_suitable(room, course):
                        for t_idx in range(len(self.timeslots)):
                            if self.is_slot_available_for_event(chromosome, r_idx, t_idx, event):
                                displaced_event_id = chromosome[r_idx, t_idx]
                                chromosome[r_idx, t_idx] = event_id
                                if displaced_event_id is not None: self._try_quick_reschedule(chromosome, displaced_event_id)
                                placed = True
                                break
                        if placed: break
        return chromosome
    
    def _try_quick_reschedule(self, chromosome, displaced_event_id):
        displaced_event = self.events_list[displaced_event_id]
        displaced_course = self.input_data.getCourse(displaced_event.course_id)
        for r_idx, room in enumerate(self.rooms):
            if self.is_room_suitable(room, displaced_course):
                for t_idx in range(len(self.timeslots)):
                    if (chromosome[r_idx, t_idx] is None and self.is_slot_available_for_event(chromosome, r_idx, t_idx, displaced_event)):
                        chromosome[r_idx, t_idx] = displaced_event_id
                        return True
        return False

    def count_course_occurrences(self, chromosome, student_group):
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
        print("\n=== COURSE ALLOCATION DIAGNOSIS ===")
        scheduled_events = set()
        for event_id in chromosome.flatten():
            if event_id is not None: scheduled_events.add(event_id)
        print(f"Total events: {len(self.events_list)}, Scheduled: {len(scheduled_events)}, Missing: {len(self.events_list) - len(scheduled_events)}")
        for student_group in self.student_groups:
            print(f"\nStudent Group: {student_group.name}")
            course_counts = self.count_course_occurrences(chromosome, student_group)
            expected_hours = [3 if self.input_data.getCourse(student_group.courseIDs[i]).credits == 1 else student_group.hours_required[i] for i in range(len(student_group.hours_required))]
            print(f"  Total hours: Expected {sum(expected_hours)}, Got {sum(course_counts.values())}")
            for i, course_id in enumerate(student_group.courseIDs):
                expected, actual = expected_hours[i], course_counts.get(course_id, 0)
                print(f"  {course_id}: Expected {expected}, Got {actual} {'✓' if actual == expected else '✗'}")
        print("=== END DIAGNOSIS ===\n")


### MODIFICATION: All of the code below this line is removed.
### This includes the global DE instance creation, the .run() call, all the print statements,
### the Dash app layout definition, callbacks, and the `if __name__ == '__main__':` block.
### The file now ends after the `diagnose_course_allocations` method, making it a pure module.