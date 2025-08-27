import random
from typing import List
import copy
from utils import Utility
from entitities.Class import Class
from input_data import input_data
import numpy as np
from constraints import Constraints

# population initialization using input_data
class DifferentialEvolution:
    def __init__(self, input_data, pop_size: int, F: float, CR: float):
        self.desired_fitness = 0
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
                                    
                                    if all(self.is_slot_available(chromosome, room_idx, timeslot_start + i) for i in range(block_hours)):
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
                                    if (self.is_slot_available(mutant_vector, r_idx, t_idx) and
                                        self._is_student_group_available(mutant_vector, event_to_move.student_group.id, t_idx) and
                                        self._is_lecturer_available(mutant_vector, event_to_move.faculty_id, t_idx)):
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
        trial_vector = target_vector.copy()

        # Identify all unique, non-None events in both parents
        target_events = set(e for e in target_vector.flatten() if e is not None)
        mutant_events = set(e for e in mutant_vector.flatten() if e is not None)

        # Find events to be added from mutant (genes in mutant but not in target)
        events_to_add = mutant_events - target_events
        
        # Find events that can be removed from trial (genes in target but not in mutant)
        # These are potential slots to be overwritten.
        events_to_remove = target_events - mutant_events
        
        # Create a list of positions for events that can be removed
        removable_positions = []
        for r in range(len(self.rooms)):
            for t in range(len(self.timeslots)):
                if trial_vector[r, t] in events_to_remove:
                    removable_positions.append((r, t))
        
        random.shuffle(removable_positions)

        # For each event that needs to be added, try to place it in a removable slot
        for event_id in events_to_add:
            if removable_positions:
                # Take a position of a gene that can be removed
                r, t = removable_positions.pop()
                # Replace it with the new gene
                trial_vector[r, t] = event_id
            else:
                # If we run out of removable positions, break. 
                # The repair function will handle any remaining inconsistencies.
                break
        
        # Apply a standard crossover element to maintain some DE characteristics
        # This adds diversity beyond just swapping missing/extra genes.
        num_rooms, num_timeslots = target_vector.shape
        j_rand = random.randrange(num_timeslots)
        for r_idx in range(num_rooms):
            for t_idx in range(num_timeslots):
                if random.random() < self.CR or t_idx == j_rand:
                    # Only crossover if the mutant gene is not None, to avoid creating empty slots unnecessarily
                    if mutant_vector[r_idx, t_idx] is not None:
                        trial_vector[r_idx, t_idx] = mutant_vector[r_idx, t_idx]

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
            'same_course_same_room_per_day'
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
            # The repair function is crucial here to ensure we don't lose events
            repaired_trial = self.verify_and_repair_course_allocations(trial_vector)
            self.population[target_idx] = repaired_trial


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
                                if (self.is_slot_available(chromosome, preferred_room, timeslot) and
                                    self._is_student_group_available(chromosome, event.student_group.id, timeslot) and
                                    self._is_lecturer_available(chromosome, event.faculty_id, timeslot)):
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
                                if (self.is_slot_available(chromosome, room_idx, timeslot_idx) and
                                    self._is_student_group_available(chromosome, event.student_group.id, timeslot_idx) and
                                    self._is_lecturer_available(chromosome, event.faculty_id, timeslot_idx)):
                                    valid_slots.append((room_idx, timeslot_idx))
                    if valid_slots:
                        room_idx, timeslot_idx = random.choice(valid_slots)
                        chromosome[room_idx][timeslot_idx] = missing_event_id
                        placed = True

                # Strategy 3 (Final Pass): Force placement by displacing another event if necessary
                if not placed and flexibility_level >= 2:
                    for room_idx, room in enumerate(self.rooms):
                        if self.is_room_suitable(room, course):
                            for timeslot_idx in range(len(self.timeslots)):
                                # Check for student/lecturer availability for the event we want to place
                                if (self._is_student_group_available(chromosome, event.student_group.id, timeslot_idx) and
                                    self._is_lecturer_available(chromosome, event.faculty_id, timeslot_idx)):
                                    
                                    # If the slot is empty, place it.
                                    if self.is_slot_available(chromosome, room_idx, timeslot_idx):
                                        chromosome[room_idx][timeslot_idx] = missing_event_id
                                        placed = True
                                        break
                            if placed: break
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
best_solution, fitness_history, generation, diversity_history = de.run(350)
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
    'lecturer_availability': "Lecturer Clashes",
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



import dash
from dash import dash_table
from dash import dcc, html, Input, Output
import pandas as pd

app = dash.Dash(__name__)

# Layout for the Dash app
app.layout = html.Div([
    html.H1("DE Timetable Output"),
    html.Div(id='tables-container')
])

# Callback to generate tables dynamically
@app.callback(
    Output('tables-container', 'children'),
    [Input('tables-container', 'n_clicks')]
)
def render_tables(n_clicks):
    all_timetables = de.print_all_timetables(best_solution, input_data.days, input_data.hours, 9)
    # print(all_timetables)
    tables = []
    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    
    for timetable_data in all_timetables:
        table = dash_table.DataTable(
            columns=[{"name": "Time", "id": "Time"}] + [{"name": day, "id": day} for day in days_of_week],
            data=[dict(zip(["Time"] + days_of_week, row)) for row in timetable_data["timetable"]],
            style_cell={
                'textAlign': 'center',
                'height': 'auto',
                'whiteSpace': 'normal',
            },
            style_data_conditional=[
                {
                    'if': {'column_id': 'Monday'},
                    'backgroundColor': 'lightblue',
                    'color': 'black',
                },
                {
                    'if': {'column_id': 'Tuesday'},
                    'backgroundColor': 'lightgreen',
                    'color': 'black',
                },
                  {
                    'if': {'column_id': 'Wednesday'},
                    'backgroundColor': 'lavender',
                    'color': 'black',
                },
                {
                    'if': {'column_id': 'Thursday'},
                    'backgroundColor': 'lightcyan',
                    'color': 'black',
                },
                {
                    'if': {'column_id': 'Friday'},
                    'backgroundColor': 'lightyellow',
                    'color': 'black',
                },
                # Add more styles for other days as needed
            ],
            tooltip_data=[
                {
                    day: {'value': 'Room info goes here', 'type': 'markdown'} for day in days_of_week
                } for row in timetable_data["timetable"]
            ],
            tooltip_duration=None
        )

        tables.append(html.Div([
            html.H3(f"Timetable for {timetable_data['student_group'].name}"), 
            table
        ]))
    
    return tables

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