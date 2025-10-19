# differential_evolution_api.py - CORRECTED AND COMPLETE VERSION
"""
API-compatible differential evolution algorithm with all improvements from differential_evolution.py
This version includes ALL repair functions and constraint enforcement logic.
"""

import random
from typing import List, Dict, Tuple, Optional
import copy
import json
import numpy as np
from constraints import Constraints
from utils import Utility
from entitities.Class import Class
import re


class DifferentialEvolutionEnhanced:
    def __init__(self, input_data, pop_size: int, F: float, CR: float):
        """Initialize the enhanced differential evolution algorithm with optimization features"""
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
        
        # Enhanced optimization features
        self.fitness_cache = {}
        self.stagnation_counter = 0
        self.diversity_history = []
        self.fitness_history = []
        self.last_improvement = float('inf')
        
        # Performance optimizations - pre-calculate building assignments
        self.room_building_cache = {}
        for idx, room in enumerate(self.rooms):
            self.room_building_cache[idx] = self.get_room_building(room)
        
        # Pre-calculate engineering groups
        self.engineering_groups = set()
        for student_group in self.student_groups:
            group_name = student_group.name.lower()
            if any(keyword in group_name for keyword in [
                'engineering', 'eng', 'computer science', 'software engineering',
                'mechatronics', 'electrical', 'mechanical', 'csc', 'sen', 'data', 'ds'
            ]):
                self.engineering_groups.add(student_group.id)
        
        # Advanced constraint handling
        self.consecutive_violations_tracker = []
        self.manual_cells_tracker = set()
        self.population_diversity_threshold = 0.1
        
        self.population = self.initialize_population()

    def create_events(self):
        """Create events list and mapping for the timetabling problem"""
        events_list = []
        event_map = {}

        idx = 0
        for student_group in self.student_groups:
            for i in range(student_group.no_courses):
                # Handle special case: 1-credit courses need 3 hours
                course = self.input_data.getCourse(student_group.courseIDs[i])
                required_hours = 3 if course and course.credits == 1 else student_group.hours_required[i]
                
                for hour in range(required_hours):
                    event = Class(student_group, student_group.teacherIDS[i], student_group.courseIDs[i])
                    events_list.append(event)
                    event_map[idx] = event
                    idx += 1
                    
        return events_list, event_map

    def initialize_population(self):
        """Initialize the population with valid chromosomes"""
        population = []
        
        for _ in range(self.pop_size):
            chromosome = self.create_chromosome()
            # Apply initial repairs to ensure validity
            chromosome = self.verify_and_repair_course_allocations(chromosome)
            try:
                # 1. Ensure same course on same day uses same room (High Priority)
                chromosome = self.ensure_valid_solution(chromosome)
                # 2. Ensure all classes are present (Highest Priority)
                chromosome = self.verify_and_repair_course_allocations(chromosome)
                # 3. Ensure multi-hour classes are consecutive (High Priority)
                chromosome = self.ensure_consecutive_slots(chromosome)
                # 4. Prevent student group and lecturer clashes (Critical Hard Constraint)
                chromosome = self.prevent_student_group_clashes(chromosome)
            except Exception as e:
                print(f" Error during initial repair of chromosome: {e}")
                
            population.append(chromosome)
            
        return np.array(population, dtype=object)

    def create_chromosome(self):
        """
        FIXED: Create a single chromosome using the 'big rocks first' and
        consecutive placement heuristic for better initial fitness.
        """
        chromosome = np.full((len(self.rooms), len(self.timeslots)), None, dtype=object)
        
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
        hours_per_day_for_group = {sg.id: [0] * self.input_data.days for sg in self.student_groups}

        for (student_group_id, course_id), event_indices in course_items:
            course = self.input_data.getCourse(course_id)
            student_group = self.input_data.getStudentGroup(student_group_id)
            hours_required = len(event_indices)

            if hours_required == 0:
                continue

            # --- Split strategies to enforce consecutive constraints ---
            # Default to single blocks if no specific strategy is defined for hours_required
            split_strategies = [(hours_required,)]

            if hours_required >= 4:
                split_strategies = [(4,), (2, 2), (3, 1)]
            elif hours_required == 3:
                split_strategies = [(3,), (2, 1)]
            elif hours_required == 2:
                split_strategies = [(2,)]
            elif hours_required == 1:
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
                            if not self.is_room_suitable(room, course):
                                continue

                            is_engineering_group = student_group.id in self.engineering_groups
                            room_building = self.room_building_cache.get(room_idx, 'UNKNOWN')
                            
                            if not is_engineering_group and room_building == 'SST':
                                continue

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
                            
                            day_of_block = timeslot_start // self.input_data.hours
                            temp_hours_per_day[day_of_block] += block_hours
                            temp_course_days_used.add(day_of_block)
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

        return chromosome
    def get_room_building(self, room):
        """Get building assignment for a room (SST or TYD)"""
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

    def is_room_suitable(self, room, course):
        """Check if a room is suitable for a course"""
        if course is None:
            return False
        return room.room_type == course.required_room_type

    def is_slot_available(self, chromosome, room_idx, timeslot_idx):
        """Check if the slot is available (not already assigned and not break time)"""
        if chromosome[room_idx][timeslot_idx] is not None:
            return False
        
        # Check if this is break time (13:00 - 14:00)
        # Break time is the 5th hour (index 4) of each day starting from 9:00
        break_hour = 4  # 13:00 is the 5th hour (index 4) starting from 9:00
        day = timeslot_idx // self.input_data.hours
        hour_in_day = timeslot_idx % self.input_data.hours
        
        # No break time on Tuesday (1) and Thursday (3)
        if hour_in_day == break_hour and day not in [1, 3]:
            return False  # Break time slot is not available
        
        return True

    def is_slot_available_for_event(self, chromosome, room_idx, timeslot_idx, event):
        """Check if a slot is available for a specific event, considering lecturer availability"""
        # First check if slot is physically available
        if not self.is_slot_available(chromosome, room_idx, timeslot_idx):
            return False
        
        # Check lecturer availability (day and time constraints)
        faculty = self.input_data.getFaculty(event.faculty_id)
        if faculty:
            # Check day availability
            if hasattr(faculty, 'avail_days') and faculty.avail_days:
                days_map = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday"}
                timeslot = self.timeslots[timeslot_idx]
                day_name = days_map.get(timeslot.day, "Unknown")
                
                is_day_ok = False
                for avail_day in faculty.avail_days:
                    if avail_day.strip().lower() in day_name.lower():
                        is_day_ok = True
                        break
                
                if not is_day_ok:
                    return False
            
            # Check time availability - FIX: Parse hour from start_time
            if hasattr(faculty, 'avail_times') and faculty.avail_times:
                timeslot = self.timeslots[timeslot_idx]
                
                # Parse hour from start_time string
                try:
                    if isinstance(timeslot.start_time, str):
                        slot_hour = int(timeslot.start_time.split(':')[0])
                    elif isinstance(timeslot.start_time, (int, float)):
                        slot_hour = int(timeslot.start_time)
                    else:
                        slot_hour = int(str(timeslot.start_time).split(':')[0])
                except (ValueError, AttributeError, IndexError):
                    # Fallback calculation based on timeslot index
                    slot_hour = 9 + (timeslot_idx % self.input_data.hours)
                
                is_time_ok = False
                for avail_time in faculty.avail_times:
                    try:
                        time_spec_str = str(avail_time).strip()
                        if '-' in time_spec_str:
                            start_str, end_str = time_spec_str.split('-')
                            start_h = int(start_str.split(':')[0])
                            end_h = int(end_str.split(':')[0])
                            if start_h <= slot_hour < end_h:
                                is_time_ok = True
                                break
                        else:
                            h = int(time_spec_str.split(':')[0])
                            if h == slot_hour:
                                is_time_ok = True
                                break
                    except (ValueError, IndexError):
                        continue
                
                if not is_time_ok:
                    return False
        
        return True
    def _is_student_group_available(self, chromosome, student_group_id, timeslot_idx):
        """Check if a student group is available at a specific timeslot"""
        for room_idx in range(len(self.rooms)):
            event_id = chromosome[room_idx, timeslot_idx]
            if event_id is not None:
                event = self.events_map[event_id]
                if event.student_group.id == student_group_id:
                    return False
        return True

    def _is_lecturer_available(self, chromosome, faculty_id, timeslot_idx):
        """Check if a lecturer is available at a specific timeslot"""
        for room_idx in range(len(self.rooms)):
            event_id = chromosome[room_idx, timeslot_idx]
            if event_id is not None:
                event = self.events_map[event_id]
                if event.faculty_id == faculty_id:
                    return False
        return True

    def hamming_distance(self, chromosome1, chromosome2):
        """Calculate Hamming distance between two chromosomes"""
        distance = 0
        total_positions = chromosome1.size
        
        flat1 = chromosome1.flatten()
        flat2 = chromosome2.flatten()
        
        for i in range(total_positions):
            if flat1[i] != flat2[i]:
                distance += 1
                
        return distance / total_positions

    def calculate_population_diversity(self):
        """Calculate average diversity of the population"""
        if self.pop_size <= 10:
            total_distance = 0
            comparisons = 0
            
            for i in range(self.pop_size):
                for j in range(i + 1, self.pop_size):
                    total_distance += self.hamming_distance(self.population[i], self.population[j])
                    comparisons += 1
            
            return total_distance / comparisons if comparisons > 0 else 0
        else:
            total_distance = 0
            comparisons = 10
            
            for _ in range(comparisons):
                i, j = random.sample(range(self.pop_size), 2)
                total_distance += self.hamming_distance(self.population[i], self.population[j])
            
            return total_distance / comparisons

    def find_clash(self, chromosome):
        """Find a timeslot where student group or lecturer clashes occur"""
        for timeslot_idx in range(len(self.timeslots)):
            student_groups_seen = set()
            lecturers_seen = set()
            
            for room_idx in range(len(self.rooms)):
                event_id = chromosome[room_idx, timeslot_idx]
                if event_id is not None:
                    event = self.events_map[event_id]
                    
                    if event.student_group.id in student_groups_seen:
                        return timeslot_idx
                    student_groups_seen.add(event.student_group.id)
                    
                    if event.faculty_id in lecturers_seen:
                        return timeslot_idx
                    lecturers_seen.add(event.faculty_id)
        
        return None

    def find_safe_empty_slot_for_event(self, chromosome, event, ignore_pos=None):
        """Find a safe empty slot for an event"""
        course = self.input_data.getCourse(event.course_id)
        if not course:
            return None

        possible_slots = []
        for r_idx, room in enumerate(self.rooms):
            if self.is_room_suitable(room, course):
                for t_idx in range(len(self.timeslots)):
                    if (r_idx, t_idx) == ignore_pos:
                        continue
                    
                    if chromosome[r_idx, t_idx] is None and self.is_slot_available_for_event(chromosome, r_idx, t_idx, event):
                        student_clash = self.constraints.check_student_group_clash_at_slot(
                            chromosome, event.student_group.id, t_idx, 
                            ignore_room_idx=ignore_pos[0] if ignore_pos else -1
                        )
                        lecturer_clash = self.constraints.check_lecturer_clash_at_slot(
                            chromosome, event.faculty_id, t_idx, 
                            ignore_room_idx=ignore_pos[0] if ignore_pos else -1
                        )
                        
                        if not student_clash and not lecturer_clash:
                            possible_slots.append((r_idx, t_idx))
        
        return random.choice(possible_slots) if possible_slots else None

    def mutate(self, target_idx):
        """COMPLETE mutation function with all repair calls - DEFENSIVE VERSION"""
        try:
            mutant_vector = self.population[target_idx].copy()
            
            mutation_attempts = random.randint(3, 8)

            for attempt in range(mutation_attempts):
                try:
                    strategy = random.choice(['resolve_clash', 'safe_swap', 'safe_move'])

                    if strategy == 'resolve_clash':
                        clash_timeslot = self.find_clash(mutant_vector)
                        if clash_timeslot is not None:
                            events_in_slot = []
                            for r in range(len(self.rooms)):
                                if mutant_vector[r, clash_timeslot] is not None:
                                    events_in_slot.append((r, mutant_vector[r, clash_timeslot]))
                            
                            if not events_in_slot:
                                continue
                            
                            room_to_move_from, event_id_to_move = random.choice(events_in_slot)
                            event_to_move = self.events_map.get(event_id_to_move)
                            if not event_to_move:
                                continue

                            new_pos = self.find_safe_empty_slot_for_event(
                                mutant_vector, event_to_move, 
                                ignore_pos=(room_to_move_from, clash_timeslot)
                            )
                            if new_pos:
                                new_r, new_t = new_pos
                                mutant_vector[new_r, new_t] = event_id_to_move
                                mutant_vector[room_to_move_from, clash_timeslot] = None

                    elif strategy == 'safe_swap':
                        occupied_slots = []
                        for r in range(len(self.rooms)):
                            for t in range(len(self.timeslots)):
                                if mutant_vector[r, t] is not None:
                                    occupied_slots.append((r, t))
                        
                        if len(occupied_slots) < 2:
                            continue
                        
                        pos1, pos2 = random.sample(occupied_slots, 2)
                        
                        event1_id = mutant_vector[pos1[0], pos1[1]]
                        event2_id = mutant_vector[pos2[0], pos2[1]]
                        event1 = self.events_map.get(event1_id)
                        event2 = self.events_map.get(event2_id)
                        
                        if not event1 or not event2:
                            continue

                        course1 = self.input_data.getCourse(event1.course_id)
                        course2 = self.input_data.getCourse(event2.course_id)
                        
                        if not course1 or not course2:
                            continue
                        
                        room1_ok_for_event2 = self.is_room_suitable(self.rooms[pos1[0]], course2)
                        room2_ok_for_event1 = self.is_room_suitable(self.rooms[pos2[0]], course1)

                        if room1_ok_for_event2 and room2_ok_for_event1:
                            clash_free_at_pos1 = not self.constraints.check_student_group_clash_at_slot(
                                mutant_vector, event2.student_group.id, pos1[1], ignore_room_idx=pos2[0]
                            ) and not self.constraints.check_lecturer_clash_at_slot(
                                mutant_vector, event2.faculty_id, pos1[1], ignore_room_idx=pos2[0]
                            )
                            
                            clash_free_at_pos2 = not self.constraints.check_student_group_clash_at_slot(
                                mutant_vector, event1.student_group.id, pos2[1], ignore_room_idx=pos1[0]
                            ) and not self.constraints.check_lecturer_clash_at_slot(
                                mutant_vector, event1.faculty_id, pos2[1], ignore_room_idx=pos1[0]
                            )

                            if clash_free_at_pos1 and clash_free_at_pos2:
                                mutant_vector[pos1[0], pos1[1]] = event2_id
                                mutant_vector[pos2[0], pos2[1]] = event1_id

                    elif strategy == 'safe_move':
                        occupied_slots = []
                        for r in range(len(self.rooms)):
                            for t in range(len(self.timeslots)):
                                if mutant_vector[r, t] is not None:
                                    occupied_slots.append((r, t))
                        
                        if not occupied_slots:
                            continue
                        
                        pos_to_move = random.choice(occupied_slots)
                        event_id_to_move = mutant_vector[pos_to_move[0], pos_to_move[1]]
                        event_to_move = self.events_map.get(event_id_to_move)
                        if not event_to_move:
                            continue

                        new_pos = self.find_safe_empty_slot_for_event(mutant_vector, event_to_move, ignore_pos=pos_to_move)
                        if new_pos:
                            new_r, new_t = new_pos
                            mutant_vector[new_r, new_t] = event_id_to_move
                            mutant_vector[pos_to_move[0], pos_to_move[1]] = None

                except Exception as strategy_error:
                    print(f"Error in mutation strategy {strategy}: {strategy_error}")
                    continue

            # CRITICAL: Apply repair functions after mutation
            try:
                mutant_vector = self.ensure_valid_solution(mutant_vector)
            except Exception as e:
                print(f"Error in ensure_valid_solution: {e}")
            
            try:
                mutant_vector = self.verify_and_repair_course_allocations(mutant_vector)
            except Exception as e:
                print(f"Error in verify_and_repair_course_allocations: {e}")
            
            try:
                mutant_vector = self.ensure_consecutive_slots(mutant_vector)
            except Exception as e:
                print(f"Error in ensure_consecutive_slots: {e}")
            
            try:
                mutant_vector = self.prevent_student_group_clashes(mutant_vector)
            except Exception as e:
                print(f"Error in prevent_student_group_clashes: {e}")

            return mutant_vector
            
        except Exception as e:
            print(f"CRITICAL ERROR in mutate(): {e}")
            import traceback
            traceback.print_exc()
            # Return unchanged population member as fallback
            return self.population[target_idx].copy()
    
    def ensure_valid_solution(self, mutant_vector):
        """Ensure same course on same day appears in same room"""
        course_day_room_mapping = {}
        
        # First pass: collect course-day-room mappings
        for room_idx in range(len(self.rooms)):
            for timeslot_idx in range(len(self.timeslots)):
                event_id = mutant_vector[room_idx, timeslot_idx]
                if event_id is not None:  # FIX: Changed != to is not
                    class_event = self.events_map.get(event_id)
                    if class_event:
                        day_idx = timeslot_idx // self.input_data.hours
                        course = self.input_data.getCourse(class_event.course_id)
                        course_id = getattr(course, 'course_id', None) or getattr(course, 'id', None) or getattr(course, 'code', None) if course else class_event.course_id
                        course_day_key = (course_id, day_idx, class_event.student_group.id)
                        
                        if course_day_key not in course_day_room_mapping:
                            course_day_room_mapping[course_day_key] = room_idx
        
        # Second pass: fix room violations
        events_to_move = []
        for room_idx in range(len(self.rooms)):
            for timeslot_idx in range(len(self.timeslots)):
                event_id = mutant_vector[room_idx, timeslot_idx]
                if event_id is not None:  # FIX: Changed != to is not
                    class_event = self.events_map.get(event_id)
                    if class_event:
                        day_idx = timeslot_idx // self.input_data.hours
                        course = self.input_data.getCourse(class_event.course_id)
                        course_id = getattr(course, 'course_id', None) or getattr(course, 'id', None) or getattr(course, 'code', None) if course else class_event.course_id
                        course_day_key = (course_id, day_idx, class_event.student_group.id)
                        expected_room = course_day_room_mapping.get(course_day_key)
                        
                        if expected_room is not None and room_idx != expected_room:
                            mutant_vector[room_idx, timeslot_idx] = None
                            events_to_move.append((event_id, expected_room, timeslot_idx))
        
        # Third pass: place moved events in correct rooms
        for event_id, correct_room, original_timeslot in events_to_move:
            placed = False
            for timeslot in range(len(self.timeslots)):
                if mutant_vector[correct_room, timeslot] is None:
                    mutant_vector[correct_room, timeslot] = event_id
                    placed = True
                    break
        
        return mutant_vector
    
    def verify_and_repair_course_allocations(self, chromosome):
        """AGGRESSIVE repair to ensure all events scheduled - FROM differential_evolution.py"""
        max_repair_passes = 5
        
        for pass_num in range(max_repair_passes):
            # Phase 1: Count scheduled events
            scheduled_event_counts = {}
            for r_idx in range(len(self.rooms)):
                for t_idx in range(len(self.timeslots)):
                    event_id = chromosome[r_idx, t_idx]
                    if event_id is not None:
                        scheduled_event_counts[event_id] = scheduled_event_counts.get(event_id, 0) + 1

            # Phase 2: Remove duplicates
            extra_event_ids = {event_id for event_id, count in scheduled_event_counts.items() if count > 1}
            
            if extra_event_ids:
                positions_to_clear = []
                for event_id in extra_event_ids:
                    locations = np.argwhere(chromosome == event_id)
                    for i in range(1, len(locations)):
                        positions_to_clear.append(tuple(locations[i]))
                
                for r_idx, t_idx in positions_to_clear:
                    chromosome[r_idx, t_idx] = None
            
            # Phase 3: AGGRESSIVELY place missing events
            scheduled_events = set(np.unique([e for e in chromosome.flatten() if e is not None]))
            all_required_events = set(range(len(self.events_list)))
            missing_events = list(all_required_events - scheduled_events)
            random.shuffle(missing_events)

            if not missing_events:
                break

            for event_id in missing_events:
                event = self.events_list[event_id]
                course = self.input_data.getCourse(event.course_id)
                if not course:
                    continue

                placed = False
                
                # Strategy 1: Find perfect slots (no conflicts)
                for r_idx, room in enumerate(self.rooms):
                    if self.is_room_suitable(room, course):
                        for t_idx in range(len(self.timeslots)):
                            if (chromosome[r_idx, t_idx] is None and
                                self.is_slot_available_for_event(chromosome, r_idx, t_idx, event) and
                                self._is_student_group_available(chromosome, event.student_group.id, t_idx) and
                                self._is_lecturer_available(chromosome, event.faculty_id, t_idx)):
                                chromosome[r_idx, t_idx] = event_id
                                placed = True
                                break
                        if placed:
                            break
                
                if placed:
                    continue
                
                # Strategy 2: Find slots with room type match only
                for r_idx, room in enumerate(self.rooms):
                    if self.is_room_suitable(room, course):
                        for t_idx in range(len(self.timeslots)):
                            if (chromosome[r_idx, t_idx] is None and
                                self.is_slot_available_for_event(chromosome, r_idx, t_idx, event)):
                                chromosome[r_idx, t_idx] = event_id
                                placed = True
                                break
                        if placed:
                            break
                
                if placed:
                    continue
                
                # Strategy 3: Try ANY room with capacity
                for r_idx, room in enumerate(self.rooms):
                    if event.student_group.no_students <= room.capacity:
                        for t_idx in range(len(self.timeslots)):
                            if chromosome[r_idx, t_idx] is None:
                                chromosome[r_idx, t_idx] = event_id
                                placed = True
                                break
                        if placed:
                            break
                
                if placed:
                    continue
                
                # Strategy 4: Force placement (displace if necessary)
                for r_idx, room in enumerate(self.rooms):
                    if self.is_room_suitable(room, course):
                        for t_idx in range(len(self.timeslots)):
                            if self.is_slot_available_for_event(chromosome, r_idx, t_idx, event):
                                displaced_event_id = chromosome[r_idx, t_idx]
                                chromosome[r_idx, t_idx] = event_id
                                placed = True
                                
                                if displaced_event_id is not None:
                                    self._try_quick_reschedule(chromosome, displaced_event_id)
                                break
                        if placed:
                            break

        return chromosome
    
    def _try_quick_reschedule(self, chromosome, displaced_event_id):
        """Helper to quickly reschedule a displaced event"""
        displaced_event = self.events_list[displaced_event_id]
        displaced_course = self.input_data.getCourse(displaced_event.course_id)
        
        for r_idx, room in enumerate(self.rooms):
            if self.is_room_suitable(room, displaced_course):
                for t_idx in range(len(self.timeslots)):
                    if (chromosome[r_idx, t_idx] is None and
                        self.is_slot_available_for_event(chromosome, r_idx, t_idx, displaced_event)):
                        chromosome[r_idx, t_idx] = displaced_event_id
                        return

    def ensure_consecutive_slots(self, chromosome):
        """COMPLETE implementation from differential_evolution.py"""
        course_events = {}
        
        for r_idx in range(len(self.rooms)):
            for t_idx in range(len(self.timeslots)):
                event_id = chromosome[r_idx, t_idx]
                if event_id is not None:  # FIX: Changed != to is not
                    event = self.events_map.get(event_id)
                    if event:
                        course_key = (event.student_group.id, event.course_id)
                        
                        if course_key not in course_events:
                            course_events[course_key] = []
                        
                        course_events[course_key].append({
                            'event_id': event_id,
                            'pos': (r_idx, t_idx),
                            'timeslot': t_idx
                        })
        
        # For each course with multiple hours, check if consecutive
        for course_key, events in course_events.items():
            if len(events) <= 1:
                continue
            
            events.sort(key=lambda x: x['timeslot'])
            
            is_consecutive = True
            base_room = events[0]['pos'][0]
            
            for i in range(len(events) - 1):
                current_timeslot = events[i]['timeslot']
                next_timeslot = events[i + 1]['timeslot']
                current_room = events[i]['pos'][0]
                next_room = events[i + 1]['pos'][0]
                
                if next_timeslot != current_timeslot + 1 or current_room != next_room:
                    is_consecutive = False
                    break
            
            if not is_consecutive:
                hours_required = len(events)
                student_group_id, course_id = course_key
                course = self.input_data.getCourse(course_id)
                
                possible_blocks = []
                for r_idx, room in enumerate(self.rooms):
                    if self.is_room_suitable(room, course):
                        for t_start in range(len(self.timeslots) - hours_required + 1):
                            is_block_valid = True
                            temp_chromosome = chromosome.copy()
                            
                            # Clear old positions
                            for event_info in events:
                                r, t = event_info['pos']
                                temp_chromosome[r, t] = None
                            
                            # Check if new block is valid
                            for i in range(hours_required):
                                t_check = t_start + i
                                event_to_place = self.events_list[events[i]['event_id']]
                                
                                if not (self.is_slot_available_for_event(temp_chromosome, r_idx, t_check, event_to_place) and
                                        self._is_student_group_available(temp_chromosome, student_group_id, t_check) and
                                        self._is_lecturer_available(temp_chromosome, event_to_place.faculty_id, t_check)):
                                    is_block_valid = False
                                    break
                            
                            if is_block_valid:
                                possible_blocks.append((r_idx, t_start))
                
                if possible_blocks:
                    new_r, new_t_start = random.choice(possible_blocks)
                    
                    # Clear old positions
                    for event_info in events:
                        r, t = event_info['pos']
                        chromosome[r, t] = None
                    
                    # Place in new consecutive block
                    for i in range(hours_required):
                        chromosome[new_r, new_t_start + i] = events[i]['event_id']

        return chromosome
    def prevent_student_group_clashes(self, chromosome):
        """COMPLETE implementation from differential_evolution.py"""
        max_clash_resolution_passes = 3
        
        for pass_num in range(max_clash_resolution_passes):
            clashes_found = False
            
            for t_idx in range(len(self.timeslots)):
                student_groups_in_slot = {}
                
                for r_idx in range(len(self.rooms)):
                    event_id = chromosome[r_idx, t_idx]
                    if event_id is not None:  # FIX: Changed != to is not
                        event = self.events_map.get(event_id)
                        if event:
                            sg_id = event.student_group.id
                            
                            if sg_id in student_groups_in_slot:
                                clashes_found = True
                                moved = False
                                
                                # Try to find a perfect alternative slot
                                for alt_r in range(len(self.rooms)):
                                    for alt_t in range(len(self.timeslots)):
                                        if (chromosome[alt_r, alt_t] is None and  # FIX
                                            self.is_slot_available_for_event(chromosome, alt_r, alt_t, event) and
                                            self._is_student_group_available(chromosome, sg_id, alt_t) and
                                            self._is_lecturer_available(chromosome, event.faculty_id, alt_t)):
                                            chromosome[r_idx, t_idx] = None
                                            chromosome[alt_r, alt_t] = event_id
                                            moved = True
                                            break
                                    if moved:
                                        break
                                
                                if not moved:
                                    course = self.input_data.getCourse(event.course_id)
                                    for alt_r_idx, room in enumerate(self.rooms):
                                        if self.is_room_suitable(room, course):
                                            for alt_t_idx in range(len(self.timeslots)):
                                                if (chromosome[alt_r_idx, alt_t_idx] is None and  # FIX
                                                    self.is_slot_available_for_event(chromosome, alt_r_idx, alt_t_idx, event)):
                                                    chromosome[r_idx, t_idx] = None
                                                    chromosome[alt_r_idx, alt_t_idx] = event_id
                                                    moved = True
                                                    break
                                            if moved:
                                                break
                            else:
                                student_groups_in_slot[sg_id] = r_idx
            
            if not clashes_found:
                break
        
        return chromosome
    def crossover(self, target_vector, mutant_vector):
        """Simple crossover that avoids creating student group clashes"""
        trial_vector = target_vector.copy()
        num_rooms, num_timeslots = target_vector.shape
        
        j_rand_r = random.randrange(num_rooms)
        j_rand_t = random.randrange(num_timeslots)

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
        """Evaluate fitness using cached results"""
        chromosome_key = str(chromosome.tobytes())
        if chromosome_key in self.fitness_cache:
            return self.fitness_cache[chromosome_key]
        
        fitness = self.constraints.evaluate_fitness(chromosome)
        
        # Cache management
        if len(self.fitness_cache) > 1000:
            keys_to_remove = list(self.fitness_cache.keys())[:-500]
            for key in keys_to_remove:
                del self.fitness_cache[key]
        
        self.fitness_cache[chromosome_key] = fitness
        return fitness

    def select(self, target_idx, trial_vector):
        """Selection with hard constraint prioritization"""
        trial_violations = self.constraints.get_constraint_violations(trial_vector)
        target_violations = self.constraints.get_constraint_violations(self.population[target_idx])

        hard_constraints = [
            'student_group_constraints', 
            'lecturer_availability', 
            'course_allocation_completeness',
            'room_time_conflict',
            'break_time_constraint',
            'room_constraints',
            'same_course_same_room_per_day',
            'lecturer_schedule_constraints',
            'lecturer_workload_constraints'
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
            self.population[target_idx] = trial_vector.copy()

    def run_enhanced(self, max_generations, job_id=None, progress_callback=None):
        """Run enhanced DE algorithm with COMPLETE repair sequence"""
        fitness_history = []
        diversity_history = []
        best_solution = self.population[0].copy()
        best_fitness = float('inf')
        stagnation_counter = 0
        
        # Calculate initial fitness
        initial_fitness = [self.evaluate_fitness(ind) for ind in self.population]
        best_idx = np.argmin(initial_fitness)
        best_solution = self.population[best_idx].copy()
        best_fitness = initial_fitness[best_idx]
        
        for generation in range(max_generations):
            generation_improved = False
            
            if progress_callback and job_id:
                progress = 10 + (generation / max_generations) * 80
                progress_callback(job_id, progress, f"Generation {generation+1}/{max_generations} - Best Fitness: {best_fitness:.2f}")
            
            for i in range(self.pop_size):
                # Step 1: Mutation (now includes all repair functions)
                mutant_vector = self.mutate(i)
                
                # Step 2: Crossover
                target_vector = self.population[i]
                trial_vector = self.crossover(target_vector, mutant_vector)
                
                # Step 3: Repair course allocations FIRST
                trial_vector = self.verify_and_repair_course_allocations(trial_vector)
                
                # Step 4: THEN handle clashes
                trial_vector = self.prevent_student_group_clashes(trial_vector)
                
                # Step 5: Final repair to catch missing events
                trial_vector = self.verify_and_repair_course_allocations(trial_vector)

                # Step 6: Selection
                old_fitness = self.evaluate_fitness(self.population[i])
                self.select(i, trial_vector)
                new_fitness = self.evaluate_fitness(self.population[i])
                
                if new_fitness < old_fitness:
                    generation_improved = True
                
            # Find best solution
            current_fitness = [self.evaluate_fitness(ind) for ind in self.population]
            current_best_idx = np.argmin(current_fitness)
            current_best_fitness = current_fitness[current_best_idx]
            
            if current_best_fitness < best_fitness:
                best_solution = self.population[current_best_idx].copy()
                best_fitness = current_best_fitness
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            fitness_history.append(best_fitness)

            # Calculate diversity periodically
            if generation % 20 == 0:
                population_diversity = self.calculate_population_diversity()
                diversity_history.append(population_diversity)
                
                # Adaptive parameter adjustment
                if population_diversity < self.population_diversity_threshold:
                    self.F = min(1.0, self.F * 1.1)
                    self.CR = max(0.1, self.CR * 0.9)

            print(f"Best solution for generation {generation+1}/{max_generations} has a fitness of: {best_fitness}")

            # Early termination
            if best_fitness <= self.desired_fitness:
                print(f"Solution with desired fitness found at Generation {generation}!")
                break
                
            if stagnation_counter >= 20:
                print(f"Early termination due to stagnation after {stagnation_counter} generations")
                break
            
            if stagnation_counter > 50 and best_fitness < 100:
                print(f"Early termination due to convergence at generation {generation+1}")
                break

        # CRITICAL: Complete post-algorithm repairs (4-step sequence)
        print(f"\n🔧 APPLYING POST-ALGORITHM REPAIRS...")
        pre_repair_fitness = self.evaluate_fitness(best_solution)
        print(f"   Fitness before repairs: {pre_repair_fitness:.4f}")
        
        best_solution = self.verify_and_repair_course_allocations(best_solution)
        best_solution = self.ensure_consecutive_slots(best_solution)
        best_solution = self.prevent_student_group_clashes(best_solution)
        
        # FINAL repair pass
        best_solution = self.verify_and_repair_course_allocations(best_solution)
        
        post_repair_fitness = self.evaluate_fitness(best_solution)
        repair_impact = post_repair_fitness - pre_repair_fitness
        print(f"   Fitness after repairs: {post_repair_fitness:.4f}")
        
        if abs(repair_impact) > 0.01:
            impact_direction = "improved" if repair_impact < 0 else "worsened"
            print(f"   🎯 Repair impact: {impact_direction} fitness by {abs(repair_impact):.4f} points")
        else:
            print(f"   ✅ Repair impact: minimal change ({repair_impact:.4f})")
        print(f"🔧 POST-ALGORITHM REPAIRS COMPLETE\n")

        # Final verification
        print("\n--- VERIFYING STUDENT GROUP CLASH PREVENTION ---")
        clash_free = self.verify_no_student_group_clashes(best_solution)
        if not clash_free:
            print("⚠️  APPLYING EMERGENCY CLASH PREVENTION...")
            best_solution = self.prevent_student_group_clashes(best_solution)
            print("✅ Emergency prevention applied!")
            self.verify_no_student_group_clashes(best_solution)
        print("--- VERIFICATION COMPLETE ---\n")

        # Get final constraint violations
        final_violations = self.constraints.get_constraint_violations(best_solution, debug=True)
        constraint_details = self.constraints.get_detailed_constraint_violations(best_solution)
        
        
        # Format timetables for frontend
        timetables = self.format_timetables_for_frontend(best_solution)
        
        return {
            'best_solution': best_solution,
            'best_fitness': post_repair_fitness,
            'fitness_history': fitness_history,
            'diversity_history': diversity_history,
            'constraint_violations': final_violations,
            'constraint_details': constraint_details,
            'generations_completed': generation + 1,
            'timetables': timetables
        }
        

    def verify_no_student_group_clashes(self, chromosome):
        """Verify no student group clashes exist"""
        print("🔍 CHECKING FOR STUDENT GROUP CLASHES...")
        
        for timeslot_idx in range(len(self.timeslots)):
            student_groups_seen = set()
            
            for room_idx in range(len(self.rooms)):
                event_id = chromosome[room_idx, timeslot_idx]
                if event_id is not None:
                    event = self.events_map[event_id]
                    sg_id = event.student_group.id
                    
                    if sg_id in student_groups_seen:
                        print(f"   ❌ CLASH DETECTED at timeslot {timeslot_idx}!")
                        print(f"   Student group {sg_id} appears multiple times at the same time!")
                        return False
                    student_groups_seen.add(sg_id)
        
        print("✅ VERIFIED: NO student group clashes detected!")
        return True

    def format_timetables_for_frontend(self, chromosome):
        """Format chromosome solution into frontend-compatible timetable structure"""
        timetables = []
        
        for student_group in self.student_groups:
            timetable_grid = []
            
            for hour_idx in range(self.input_data.hours):
                row = [f"{9 + hour_idx}:00-{10 + hour_idx}:00"]
                
                for day_idx in range(self.input_data.days):
                    timeslot_idx = day_idx * self.input_data.hours + hour_idx
                    cell_content = ""
                    
                    # Check break time
                    if hour_idx == 4 and day_idx in [0, 2, 4]:
                        cell_content = "BREAK"
                    else:
                        for room_idx in range(len(self.rooms)):
                            if timeslot_idx < len(self.timeslots):
                                event_id = chromosome[room_idx, timeslot_idx]
                                if event_id is not None:
                                    event = self.events_map[event_id]
                                    if event.student_group.id == student_group.id:
                                        course = self.input_data.getCourse(event.course_id)
                                        faculty = self.input_data.getFaculty(event.faculty_id)
                                        room = self.rooms[room_idx]
                                        
                                        course_name = course.name if course else "Unknown Course"
                                        faculty_name = faculty.name if faculty else "Unknown Faculty"
                                        room_name = getattr(room, 'name', getattr(room, 'Id', f"Room-{room_idx}"))
                                        
                                        cell_content = f"{course_name}\n{room_name}\n{faculty_name}"
                                        break
                    
                    row.append(cell_content)
                
                timetable_grid.append(row)
            
            # Calculate summary statistics
            total_courses = len(student_group.courseIDs)
            total_hours = sum(student_group.hours_required)
            filled_slots = sum(1 for row in timetable_grid for cell in row[1:] if cell and cell != "BREAK")
            
            timetables.append({
                'student_group': {
                    'id': student_group.id,
                    'name': student_group.name,
                    'department': getattr(student_group, 'department', ''),
                    'level': getattr(student_group, 'level', '')
                },
                'timetable': timetable_grid,
                'summary': {
                    'total_courses': total_courses,
                    'total_hours': total_hours,
                    'filled_slots': filled_slots
                }
            })
        
        return timetables

    def get_detailed_constraint_violations(self, chromosome):
        """Get detailed constraint violations for UI display"""
        return self.constraints.get_detailed_constraint_violations(chromosome)

    def get_break_time_violations(self, chromosome):
        """Get break time violations"""
        violations = []
        break_hour = 4
        
        try:
            days = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday"}
            
            for day_idx in [0, 2, 4]:  # Mon, Wed, Fri
                for hour_idx in range(self.input_data.hours):
                    if hour_idx == break_hour:
                        timeslot_idx = day_idx * self.input_data.hours + hour_idx
                        
                        for room_idx in range(len(self.rooms)):
                            if timeslot_idx < len(self.timeslots):
                                event_id = chromosome[room_idx, timeslot_idx]
                                if event_id is not None:
                                    event = self.events_map[event_id]
                                    course = self.input_data.getCourse(event.course_id)
                                    course_name = course.code if course else f"Course-{event.course_id}"
                                    
                                    violations.append({
                                        'group': event.student_group.name,
                                        'course': course_name,
                                        'location': f"{days[day_idx]} at {9 + break_hour}:00"
                                    })
        
        except Exception as e:
            print(f"Error getting break time violations: {e}")
        
        return violations
    
    def count_non_none(self, arr):
        """Count non-None elements in array - utility for debugging"""
        return np.count_nonzero(arr != None)

    def print_timetable(self, individual, student_group, days, hours_per_day, day_start_time=9):
        """Print timetable for a specific student group - for debugging"""
        timetable = [["" for _ in range(days)] for _ in range(hours_per_day)]
        
        # Fill break time slots on Mon, Wed, Fri
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
                    
                    # Check if it's a break slot that should be skipped
                    is_display_break = (hour == break_hour and day in [0, 2, 4])

                    if day < days and not is_display_break:
                        course = self.input_data.getCourse(class_event.course_id)
                        faculty = self.input_data.getFaculty(class_event.faculty_id)
                        course_code = course.code if course is not None else "Unknown"
                        
                        # Use faculty name if available
                        if faculty is not None:
                            faculty_name = faculty.name if faculty.name else faculty.faculty_id
                        else:
                            faculty_name = "Unknown"
                        
                        room_obj = self.input_data.rooms[room_idx]
                        room_display = getattr(room_obj, "name", getattr(room_obj, "Id", str(room_idx)))
                        
                        # Format as: Course Code\nRoom Name\nFaculty Name
                        timetable[hour][day] = f"{course_code}\n{room_display}\n{faculty_name}"
        
        return timetable

    def print_all_timetables(self, individual, days=5, hours_per_day=8, day_start_time=9):
        """Print timetables for all student groups - enhanced for API compatibility"""
        timetables = []
        
        for student_group in self.student_groups:
            timetable = self.print_timetable(individual, student_group, days, hours_per_day, day_start_time)
            rows = []
            
            for hour in range(hours_per_day):
                time_label = f"{day_start_time + hour}:00"
                row = [time_label] + [timetable[hour][day] for day in range(days)]
                rows.append(row)
            
            timetables.append({
                "student_group": student_group, 
                "timetable": rows
            })
        
        return timetables

    def get_student_groups(self):
        """Get list of student groups"""
        return self.student_groups

    def get_courses(self):
        """Get list of courses"""
        return self.courses

    def get_faculty(self):
        """Get list of faculty"""
        return self.input_data.faculty

    def get_rooms(self):
        """Get list of rooms"""
        return self.rooms

    def getCourse(self, course_id):
        """Get course by ID - wrapper method"""
        return self.input_data.getCourse(course_id)

    def getFaculty(self, faculty_id):
        """Get faculty by ID - wrapper method"""
        return self.input_data.getFaculty(faculty_id)

    def optimize_for_api(self, population_size=50, max_generations=100, mutation_factor=0.4, 
                        crossover_rate=0.9, break_start_time=12, break_duration=1,
                         job_id=None, progress_callback=None):
        """Enhanced optimize method that returns UI-compatible data structure"""
        try:
            print(f"🚀 Starting optimization for API with parameters:")
            print(f"   - Population size: {population_size}")
            print(f"   - Max generations: {max_generations}")
            print(f"   - Mutation factor: {mutation_factor}")
            print(f"   - Crossover rate: {crossover_rate}")
            
            # Update parameters
            self.pop_size = population_size
            self.F = mutation_factor
            self.CR = crossover_rate
            
            # Run the enhanced optimization
            result = self.run_enhanced(max_generations, job_id, progress_callback)
            
            best_solution = result['best_solution']
            final_fitness = result['best_fitness']
            
            print("📊 Generating UI-compatible timetables...")
            all_timetables = result['timetables']
            
            print("🔍 Getting detailed constraint violations...")
            constraint_details = result['constraint_details']
            
            return {
                'success': True,
                'best_solution': best_solution,
                'best_fitness': final_fitness,
                'fitness_history': result['fitness_history'],
                'diversity_history': result['diversity_history'],
                'constraint_violations': result['constraint_violations'],
                'constraint_details': constraint_details,
                'generations_completed': result['generations_completed'],
                'timetables': all_timetables,
                'break_time_violations': self.get_break_time_violations(best_solution)
            }
            
        except Exception as e:
            print(f"❌ Error in optimization: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'timetables': [],
                'best_fitness': float('inf')
            }