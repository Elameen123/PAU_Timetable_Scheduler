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
                self.hourcount = 1 
                while self.hourcount <= student_group.hours_required[i]:
                    event = Class(student_group, student_group.teacherIDS[i], student_group.courseIDs[i])
                    events_list.append(event)
                    
                    # Add the event to the index map with the current index
                    event_map[idx] = event
                    idx += 1
                    self.hourcount += 1
                    
        return events_list, event_map

    def initialize_population(self):
        population = [] 
        for i in range(self.pop_size):
            chromosome = self.create_chromosome()
            population.append(chromosome)
        return np.array(population)

    def create_chromosome(self):
        # 2D NumPy array for chromosome where each row is a room and each column is a time slot
        chromosome = np.empty((len(self.rooms), len(self.timeslots)), dtype=object)

        # Optimization: Create smarter initial assignment to reduce high fitness
        unassigned_events = []
        course_day_room_mapping = {}  # Track room assignments for courses per day
        
        for idx, event in enumerate(self.events_list):
            valid_slots = []
            course = input_data.getCourse(event.course_id)
            student_group = event.student_group
            
            # Calculate which day this event might be scheduled on
            day_preference = None
            preferred_room = None
            
            # Check if this course already appeared today and get its room
            for potential_timeslot in range(len(self.timeslots)):
                day_idx = potential_timeslot // input_data.hours
                # Use the correct course identifier (try course_id first, then id, then other common names)
                course_id = getattr(course, 'course_id', None) or getattr(course, 'id', None) or getattr(course, 'code', None)
                course_day_key = (course_id, day_idx, student_group.id)  # Include student group for uniqueness
                
                if course_day_key in course_day_room_mapping:
                    day_preference = day_idx
                    preferred_room = course_day_room_mapping[course_day_key]
                    break
            
            # Optimization: Prioritize building assignments during initial generation
            for room_idx, room in enumerate(self.rooms):
                if self.is_room_suitable(room, course):
                    # Apply STRICT building preference logic using cached values
                    room_building = self.room_building_cache[room_idx]
                    is_engineering = student_group.id in self.engineering_groups
                    
                    # Check if this course needs a computer lab (exception to building rules)
                    needs_computer_lab = (
                        course.required_room_type.lower() in ['comp lab', 'computer_lab'] or
                        room.room_type.lower() in ['comp lab', 'computer_lab'] or
                        ('lab' in course.name.lower() and ('computer' in course.name.lower() or 
                                                          'programming' in course.name.lower() or
                                                          'software' in course.name.lower()))
                    )
                    
                    # FLEXIBLE Building assignment rules with computer lab exception
                    building_allowed = True
                    if needs_computer_lab:
                        # Computer labs can be in any building (exception rule)
                        building_allowed = True
                    elif is_engineering:
                        # Engineering groups PREFER SST but can use TYD for 1-2 courses
                        # Count how many non-SST courses this student group already has
                        non_sst_count = 0
                        for r_idx in range(len(self.rooms)):
                            for t_idx in range(len(self.timeslots)):
                                existing_event_id = chromosome[r_idx][t_idx]
                                if existing_event_id is not None:
                                    existing_event = self.events_map.get(existing_event_id)
                                    if (existing_event and 
                                        existing_event.student_group.id == student_group.id and
                                        self.room_building_cache[r_idx] != 'SST'):
                                        non_sst_count += 1
                        
                        # Allow up to 2 courses outside SST for engineering groups
                        if room_building != 'SST' and non_sst_count >= 2:
                            building_allowed = False
                    else:
                        # Non-engineering groups PREFER TYD but computer labs are allowed
                        if room_building == 'SST' and not needs_computer_lab:
                            building_allowed = False
                    
                    if building_allowed:
                        for i in range(len(self.timeslots)):
                            if self.is_slot_available(chromosome, room_idx, i):
                                day_idx = i // input_data.hours
                                
                                # Same course same day constraint
                                if day_preference is not None:
                                    if day_idx == day_preference and room_idx == preferred_room:
                                        valid_slots.insert(0, (room_idx, i))  # Prioritize same room
                                    elif day_idx == day_preference:
                                        continue  # Skip other rooms on same day
                                    else:
                                        valid_slots.append((room_idx, i))
                                else:
                                    valid_slots.append((room_idx, i))

            if len(valid_slots) > 0:
                row, col = random.choice(valid_slots)
                chromosome[row, col] = idx
                
                # Track course-day-room mapping
                day_idx = col // input_data.hours
                course_id = getattr(course, 'course_id', None) or getattr(course, 'id', None) or getattr(course, 'code', None)
                course_day_key = (course_id, day_idx, student_group.id)  # Include student group for uniqueness
                course_day_room_mapping[course_day_key] = row
            else:
                unassigned_events.append(idx)
                
        # Verify and repair missing course allocations - ensure ALL events are placed
        chromosome = self.verify_and_repair_course_allocations(chromosome)
        
        # Double-check: if still missing events, do one more aggressive repair
        scheduled_events = set()
        for room_idx in range(len(self.rooms)):
            for timeslot_idx in range(len(self.timeslots)):
                event_id = chromosome[room_idx][timeslot_idx]
                if event_id is not None:
                    scheduled_events.add(event_id)
        
        missing_count = len(self.events_list) - len(scheduled_events)
        if missing_count > 0:
            # Force place any remaining missing events
            for event_id in range(len(self.events_list)):
                if event_id not in scheduled_events:
                    event = self.events_list[event_id]
                    course = input_data.getCourse(event.course_id)
                    
                    # Find any available slot that meets basic requirements
                    for room_idx, room in enumerate(self.rooms):
                        if self.is_room_suitable(room, course):
                            for timeslot_idx in range(len(self.timeslots)):
                                if self.is_slot_available(chromosome, room_idx, timeslot_idx):
                                    chromosome[room_idx][timeslot_idx] = event_id
                                    break
                            else:
                                continue
                            break
        
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
        
        if hour_in_day == break_hour:
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
        indices = list(range(self.pop_size))
        indices.remove(target_idx)

        # Ensure population size is sufficient for mutation
        if len(indices) < 3:
            raise ValueError("Not enough population members to perform mutation.")
    
        r1, r2, r3 = random.sample(indices, 3)

        # Optimization: Use numpy copy instead of deepcopy for better performance
        x_r1 = self.population[r1].copy()
        x_r2 = self.population[r2]
        x_r3 = self.population[r3]

        for room_idx in range(len(self.rooms)):
            for timeslot_idx in range(len(self.timeslots)):
                # Check if the room-timeslot assignment is different between x_r2 and x_r3, and x_r1 has an open slot
                if x_r2[room_idx][timeslot_idx] != x_r3[room_idx][timeslot_idx] and x_r1[room_idx][timeslot_idx] == None:
                    mutant_gene = x_r2[room_idx][timeslot_idx] or x_r3[room_idx][timeslot_idx]
                    
                    # With probability F, adopt the assignment from x_r2 into x_r1
                    if random.random() < self.F:
                        self.remove_previous_event_assignment(x_r1, mutant_gene)
                        x_r1[room_idx][timeslot_idx] = mutant_gene
        
        mutant_vector = x_r1  # The mutated chromosome
        return self.ensure_valid_solution(mutant_vector)

    def ensure_valid_solution(self, mutant_vector):
        """Ensure same course on same day appears in same room"""
        course_day_room_mapping = {}
        violations_fixed = 0
        
        # First pass: collect course-day-room mappings
        for room_idx in range(len(self.rooms)):
            for timeslot_idx in range(len(self.timeslots)):
                event_id = mutant_vector[room_idx][timeslot_idx]
                if event_id is not None:
                    class_event = self.events_map.get(event_id)
                    if class_event:
                        day_idx = timeslot_idx // input_data.hours
                        # Use the correct course identifier - must include student group for uniqueness
                        course = input_data.getCourse(class_event.course_id)
                        course_id = getattr(course, 'course_id', None) or getattr(course, 'id', None) or getattr(course, 'code', None) if course else class_event.course_id
                        course_day_key = (course_id, day_idx, class_event.student_group.id)  # Include student group
                        
                        if course_day_key not in course_day_room_mapping:
                            course_day_room_mapping[course_day_key] = room_idx
        
        # Second pass: fix violations
        events_to_move = []
        for room_idx in range(len(self.rooms)):
            for timeslot_idx in range(len(self.timeslots)):
                event_id = mutant_vector[room_idx][timeslot_idx]
                if event_id is not None:
                    class_event = self.events_map.get(event_id)
                    if class_event:
                        day_idx = timeslot_idx // input_data.hours
                        # Use the correct course identifier - must include student group for uniqueness
                        course = input_data.getCourse(class_event.course_id)
                        course_id = getattr(course, 'course_id', None) or getattr(course, 'id', None) or getattr(course, 'code', None) if course else class_event.course_id
                        course_day_key = (course_id, day_idx, class_event.student_group.id)  # Include student group
                        expected_room = course_day_room_mapping[course_day_key]
                        
                        if room_idx != expected_room:
                            # This event is in wrong room for this course on this day
                            mutant_vector[room_idx][timeslot_idx] = None
                            events_to_move.append((event_id, expected_room, timeslot_idx))
                            violations_fixed += 1
        
        # Third pass: place moved events in correct rooms
        for event_id, correct_room, original_timeslot in events_to_move:
            day_idx = original_timeslot // input_data.hours
            # Try to find available slot in correct room on same day
            day_start = day_idx * input_data.hours
            day_end = (day_idx + 1) * input_data.hours
            
            placed = False
            for timeslot in range(day_start, day_end):
                if timeslot < len(self.timeslots) and self.is_slot_available(mutant_vector, correct_room, timeslot):
                    mutant_vector[correct_room][timeslot] = event_id
                    placed = True
                    break
            
            if not placed:
                # If no slot available in correct room on same day, try other days
                for timeslot in range(len(self.timeslots)):
                    if self.is_slot_available(mutant_vector, correct_room, timeslot):
                        mutant_vector[correct_room][timeslot] = event_id
                        placed = True
                        break
        
        # Verify and repair missing course allocations after moving events
        # Apply repair every time to ensure all courses are properly allocated
        mutant_vector = self.verify_and_repair_course_allocations(mutant_vector)
        
        return mutant_vector
    
    def count_non_none(self, arr):
        # Flatten the 2D array and count elements that are not None
        return np.count_nonzero(arr != None)
    
    def crossover(self, target_vector, mutant_vector):
        donor_vector = mutant_vector
        # Optimization: Use numpy copy instead of deepcopy
        trial_vector = target_vector.copy()

        # Perform crossover based on a crossover rate (CR)
        for room_idx in range(len(self.rooms)):
            for timeslot_idx in range(len(self.timeslots)):
                crossover_gene = donor_vector[room_idx][timeslot_idx]
                if crossover_gene is not None and self.is_slot_available(trial_vector, room_idx, timeslot_idx):
                    if random.random() < self.CR:  # With probability CR, adopt from mutant
                        self.remove_previous_event_assignment(trial_vector, crossover_gene)
                        trial_vector[room_idx][timeslot_idx] = crossover_gene

        # Verify and repair missing course allocations after crossover
        # Apply repair every time to ensure course completeness
        trial_vector = self.verify_and_repair_course_allocations(trial_vector)
        
        return trial_vector
    
    def remove_previous_event_assignment(self, chromosome, gene):
        for room_idx in range(len(self.rooms)):
            for timeslot_idx in range(len(self.timeslots)):
                if chromosome[room_idx][timeslot_idx] is not None and chromosome[room_idx][timeslot_idx] == gene:
                    chromosome[room_idx][timeslot_idx] = None
                    return

    
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
        # Evaluate the fitness of both the trial vector and the target vector
        trial_fitness = self.evaluate_fitness(trial_vector)
        target_fitness = self.evaluate_fitness(self.population[target_idx])
        
        # If the trial vector is better, it replaces the target in the population
        if trial_fitness <= target_fitness:  # Allow equal fitness to increase diversity
            # Ensure the selected solution has all events properly allocated
            trial_vector = self.verify_and_repair_course_allocations(trial_vector)
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
        
        # First, fill break time slots
        break_hour = 4  # 13:00 is the 5th hour (index 4) starting from 9:00
        if break_hour < hours_per_day:
            for day in range(days):
                timetable[break_hour][day] = "BREAK"
        
        for room_idx, room_slots in enumerate(individual):
            for timeslot_idx, event in enumerate(room_slots):
                class_event = self.events_map.get(event)
                if class_event is not None and class_event.student_group.id == student_group.id:
                    day = timeslot_idx // hours_per_day
                    hour = timeslot_idx % hours_per_day
                    if day < days and hour != break_hour:  # Don't overwrite break slots
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
        # Multiple repair passes to ensure completeness
        max_repair_passes = 3
        
        for repair_pass in range(max_repair_passes):
            # Check for missing events
            scheduled_events = set()
            for room_idx in range(len(self.rooms)):
                for timeslot_idx in range(len(self.timeslots)):
                    event_id = chromosome[room_idx][timeslot_idx]
                    if event_id is not None:
                        scheduled_events.add(event_id)
            
            # Find truly missing events (events that should exist but aren't scheduled anywhere)
            missing_events = []
            for event_id in range(len(self.events_list)):
                if event_id not in scheduled_events:
                    missing_events.append(event_id)
            
            # If no missing events, we're done
            if not missing_events:
                break
            
            # Get more aggressive with each pass
            flexibility_level = repair_pass
        
            # Get more aggressive with each pass
            flexibility_level = repair_pass
        
            # Track course-day room assignments to maintain consistency
            course_day_room_mapping = {}
            for room_idx in range(len(self.rooms)):
                for timeslot_idx in range(len(self.timeslots)):
                    event_id = chromosome[room_idx][timeslot_idx]
                    if event_id is not None:
                        event = self.events_map.get(event_id)
                        if event:
                            course = input_data.getCourse(event.course_id)
                            if course:
                                day_idx = timeslot_idx // input_data.hours
                                course_id = getattr(course, 'course_id', None) or getattr(course, 'id', None) or getattr(course, 'code', None)
                                course_day_key = (course_id, day_idx, event.student_group.id)
                                course_day_room_mapping[course_day_key] = room_idx
            
            # Try to place missing events with increasing aggression
            for missing_event_id in missing_events:
                event = self.events_list[missing_event_id]
                course = input_data.getCourse(event.course_id)
                
                if not course:
                    continue
                    
                course_id = getattr(course, 'course_id', None) or getattr(course, 'id', None) or getattr(course, 'code', None)
                placed = False
                
                # Strategy 1: Try to place in same room as other instances (only on first pass)
                if flexibility_level == 0:
                    preferred_slots = []
                    for day_idx in range(input_data.days):
                        course_day_key = (course_id, day_idx, event.student_group.id)
                        if course_day_key in course_day_room_mapping:
                            preferred_room = course_day_room_mapping[course_day_key]
                            day_start = day_idx * input_data.hours
                            day_end = (day_idx + 1) * input_data.hours
                            for timeslot in range(day_start, day_end):
                                if (timeslot < len(self.timeslots) and 
                                    self.is_slot_available(chromosome, preferred_room, timeslot) and
                                    timeslot % input_data.hours != 4):  # Avoid break time
                                    preferred_slots.append((preferred_room, timeslot))
                    
                    # If we have preferred slots, use them
                    if preferred_slots:
                        room_idx, timeslot_idx = preferred_slots[0]  # Take first available
                        chromosome[room_idx][timeslot_idx] = missing_event_id
                        placed = True
                
                # Strategy 2: Find valid slots with building constraints (flexibility increases each pass)
                if not placed:
                    valid_slots = []
                    for room_idx, room in enumerate(self.rooms):
                        if self.is_room_suitable(room, course):
                            # Check building constraints with increasing flexibility
                            room_building = self.room_building_cache[room_idx]
                            is_engineering = event.student_group.id in self.engineering_groups
                            
                            # Check if this course needs a computer lab
                            needs_computer_lab = (
                                course.required_room_type.lower() in ['comp lab', 'computer_lab'] or
                                room.room_type.lower() in ['comp lab', 'computer_lab'] or
                                ('lab' in course.name.lower() and ('computer' in course.name.lower() or 
                                                                  'programming' in course.name.lower() or
                                                                  'software' in course.name.lower()))
                            )
                            
                            # Apply building assignment rules with increasing flexibility
                            building_allowed = True
                            if needs_computer_lab:
                                building_allowed = True  # Always allow computer labs
                            elif flexibility_level >= 2:
                                building_allowed = True  # On final pass, ignore building constraints
                            elif is_engineering:
                                # Count existing non-SST courses for this student group
                                non_sst_count = 0
                                for r_idx in range(len(self.rooms)):
                                    for t_idx in range(len(self.timeslots)):
                                        existing_event_id = chromosome[r_idx][t_idx]
                                        if existing_event_id is not None:
                                            existing_event = self.events_map.get(existing_event_id)
                                            if (existing_event and 
                                                existing_event.student_group.id == event.student_group.id and
                                                self.room_building_cache[r_idx] != 'SST'):
                                                non_sst_count += 1
                                
                                # Increase flexibility with each pass
                                max_non_sst = 2 + flexibility_level  # 2, 3, 4 for passes 0, 1, 2
                                if room_building != 'SST' and non_sst_count >= max_non_sst:
                                    building_allowed = False
                            else:
                                # Non-engineering: prefer TYD but be flexible on later passes
                                if room_building == 'SST' and not needs_computer_lab and flexibility_level == 0:
                                    building_allowed = False
                            
                            if building_allowed:
                                for timeslot_idx in range(len(self.timeslots)):
                                    if (self.is_slot_available(chromosome, room_idx, timeslot_idx) and
                                        timeslot_idx % input_data.hours != 4):  # Avoid break time
                                        valid_slots.append((room_idx, timeslot_idx))
                    
                    # Place in first valid slot
                    if valid_slots:
                        room_idx, timeslot_idx = valid_slots[0]
                        chromosome[room_idx][timeslot_idx] = missing_event_id
                        placed = True
                    
                # Strategy 3: Emergency placement (only room type constraint)
                if not placed and flexibility_level >= 1:
                    for room_idx, room in enumerate(self.rooms):
                        if self.is_room_suitable(room, course):  # Keep room type constraint
                            for timeslot_idx in range(len(self.timeslots)):
                                if (self.is_slot_available(chromosome, room_idx, timeslot_idx) and
                                    timeslot_idx % input_data.hours != 4):  # Still avoid break time
                                    chromosome[room_idx][timeslot_idx] = missing_event_id
                                    placed = True
                                    break
                            if placed:
                                break
                    
                # Strategy 4: Force placement with displacement (final pass only)
                if not placed and flexibility_level >= 2:
                    for room_idx, room in enumerate(self.rooms):
                        if self.is_room_suitable(room, course):
                            for timeslot_idx in range(len(self.timeslots)):
                                if timeslot_idx % input_data.hours != 4:  # Only avoid break time
                                    # If slot is occupied, try to move the existing event
                                    if chromosome[room_idx][timeslot_idx] is not None:
                                        existing_event_id = chromosome[room_idx][timeslot_idx]
                                        # Try to find another slot for the existing event
                                        moved_existing = False
                                        for alt_room in range(len(self.rooms)):
                                            for alt_time in range(len(self.timeslots)):
                                                if (chromosome[alt_room][alt_time] is None and
                                                    alt_time % input_data.hours != 4):
                                                    chromosome[alt_room][alt_time] = existing_event_id
                                                    moved_existing = True
                                                    break
                                            if moved_existing:
                                                break
                                        
                                        if moved_existing:
                                            chromosome[room_idx][timeslot_idx] = missing_event_id
                                            placed = True
                                            break
                                    else:
                                        # Slot is empty, place here
                                        chromosome[room_idx][timeslot_idx] = missing_event_id
                                        placed = True
                                        break
                            if placed:
                                break
        
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
    
    for timetable_data in all_timetables:
        table = dash_table.DataTable(
            columns=[{"name": "Time", "id": "Time"}] + [{"name": f"Day {d+1}", "id": f"Day {d+1}"} for d in range(input_data.days)],
            data=[dict(zip(["Time"] + [f"Day {d+1}" for d in range(input_data.days)], row)) for row in timetable_data["timetable"]],
            style_cell={
                'textAlign': 'center',
                'height': 'auto',
                'whiteSpace': 'normal',
            },
            style_data_conditional=[
                {
                    'if': {'column_id': 'Day 1'},
                    'backgroundColor': 'lightblue',
                    'color': 'black',
                },
                {
                    'if': {'column_id': 'Day 2'},
                    'backgroundColor': 'lightgreen',
                    'color': 'black',
                },
                  {
                    'if': {'column_id': 'Day 3'},
                    'backgroundColor': 'lavender',
                    'color': 'black',
                },
                {
                    'if': {'column_id': 'Day 4'},
                    'backgroundColor': 'lightcyan',
                    'color': 'black',
                },
                {
                    'if': {'column_id': 'Day 5'},
                    'backgroundColor': 'lightyellow',
                    'color': 'black',
                },
                # Add more styles for other days as needed
            ],
            tooltip_data=[
                {
                    f"Day {d+1}": {'value': 'Room info goes here', 'type': 'markdown'} for d in range(input_data.days)
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