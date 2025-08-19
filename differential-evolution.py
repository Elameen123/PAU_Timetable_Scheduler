import random
from typing import List
import copy
# from Chromosome import Chromosome
from Gene import Gene
from utils import Utility
from entitities.Class import Class
# from entitities.course import Course
# from timetable import Timetable
from input_data import input_data
import numpy as np

# Updated population initialization using input_data
class DifferentialEvolution:
    def __init__(self, input_data, pop_size: int, F: float, CR: float):
        self.rooms = input_data.rooms
        self.timeslots = input_data.create_time_slots(no_hours_per_day=input_data.hours, no_days_per_week=input_data.days, day_start_time=9)
        self.student_groups = input_data.student_groups
        self.courses = input_data.courses
        self.events_list, self.events_map = self.create_events()
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.population = self.initialize_population()  # List to hold all chromosomes

    def create_events(self):
        events_list = []
        event_map = {}

        idx = 0
        for student_group in self.student_groups:
            # periods = 0
            for i in range(student_group.no_courses):
                self.hourcount = 1 
                while self.hourcount <= student_group.hours_required[i]:
                    event = Class(student_group, student_group.teacherIDS[i], student_group.courseIDs[i])
                    events_list.append(event)
                    
                    # Add the event to the index map with the current index
                    event_map[idx] = event
                    idx += 1

                    self.hourcount += 1
                    # self.cnt += 1
                    # periods += 1
        # print(events_list, len(events_list))
        return events_list, event_map

    def initialize_population(self):
        population = [] 
        for _ in range(self.pop_size):
            chromosome = self.create_chromosome()
            population.append(chromosome)
        return np.array(population)

    def create_chromosome(self):
        # 2D NumPy array for chromosome where each row is a room and each column is a time slot
        chromosome = np.empty((len(self.rooms), len(self.timeslots)), dtype=object)
        # rows, cols = chromosome.shape
        
        # # Assign classes to rooms and time slots
        # for student_group in self.student_groups:
        #     for course_id in student_group.courseIDs:
        #         course = input_data.getCourse(course_id)
                
        #         # Assign timeslots based on course's required hours
        #         self.assign_class_to_timeslots(chromosome, student_group, course)

        # return chromosome
        unassigned_events = []
        for idx, event in enumerate(self.events_list):
            valid_slots = []
            course = input_data.getCourse(event.course_id)
            for room_idx, room in enumerate(self.rooms):
                if self.is_room_suitable(room, course):
                    for i in range(len(self.timeslots)):
                        if self.is_slot_available(chromosome, room_idx, i):
                            valid_slots.append((room_idx, i))

            if len(valid_slots) > 0:
                row, col = random.choice(valid_slots)
                chromosome[row, col] = idx
            else:
                unassigned_events.append(idx) #TODO
        # print("unassigned events", unassigned_events)
        return chromosome

    # def assign_room_and_timeslot(self, class_event: Class):
        course = input_data.getCourse(class_event.course_id)
        # Randomly assign a room and time slot, ensuring the room matches the required room type
        valid_rooms = [room for room in self.rooms if room.room_type == course.required_room_type]
        if not valid_rooms:
            print('no room (or timeslot) found for', class_event)
            return None, None

        room = random.choice(valid_rooms)
        timeslot = random.choice(self.timeslots)
        return self.rooms.index(room), self.timeslots.index(timeslot)

    # def assign_class_to_timeslots(self, chromosome, student_group, course):
        required_hours = course.credits  # course credits map to required hours
        two_hour_blocks = required_hours // 2  # Number of 2-hour blocks
        remaining_hours = required_hours % 2  # Remaining 1 hour if required_hours is odd
        
        # Find consecutive slots for each 2-hour block
        for _ in range(two_hour_blocks):
            room, timeslot1, timeslot2 = self.find_consecutive_slots(chromosome, course)
            if room is not None:
                # Assign the two consecutive slots to the course
                class_event = Class(student_group=student_group, faculty_id=course.facultyId, course_id=course.code)
                chromosome[room][timeslot1] = class_event
                chromosome[room][timeslot2] = class_event
        
        # Handle the remaining hour (if odd)
        if remaining_hours > 0:
            room, timeslot = self.find_single_slot(chromosome, course)
            if room is not None:
                class_event = Class(student_group=student_group, faculty_id=course.facultyId, course_id=course.code)
                chromosome[room][timeslot] = class_event

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
        # Find a single available slot (for the odd remaining hour)
        # valid_rooms = [room for room in self.rooms if self.is_room_suitable(room, course)]
        # if not valid_rooms:
        #     return None, None
        
        # room_idx, timeslot_idx = self.rooms.index(random.choice(valid_rooms)), self.timeslots.index(random.choice(self.timeslots))

        # while self.is_slot_available(room_idx, timeslot_idx) == False:
        #     room_idx, timeslot_idx = self.rooms.index(random.choice(valid_rooms)), self.timeslots.index(random.choice(self.timeslots))
        
        # return room_idx, timeslot_idx

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
        return chromosome[room_idx][timeslot_idx] is None

    def is_room_suitable(self, room, course):
        return room.room_type == course.required_room_type
    
    # def mutate(self, target_idx):
        # select three random indices from the population excluding the target
        indices = list(range(self.pop_size))
        indices.remove(target_idx)
        r1, r2, r3 = random.sample(indices, 3)

        # retrieve the three individuals
        x_r1 = self.population[r1]
        x_r2 = self.population[r2]
        x_r3 = self.population[r3]

        # create the mutatant vector
        mutant_vector = x_r1 + self.F * (x_r2 - x_r3)

        # Ensure the mnutant vector is valid
        mutant_vector = self.ensure_valid_mutant(mutant_vector)

        return mutant_vector
    

    def mutate(self, target_idx):
        indices = list(range(self.pop_size))
        indices.remove(target_idx)

            # Ensure population size is sufficient for mutation
        if len(indices) < 3:
            raise ValueError("Not enough population members to perform mutation.")
    
        r1, r2, r3 = random.sample(indices, 3)

        x_r1 = copy.deepcopy(self.population[r1])
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
        # for now
        return mutant_vector
    
    def count_non_none(self, arr):
        # Flatten the 2D array and count elements that are not None
        return np.count_nonzero(arr != None)
    
    def crossover(self, target_vector, mutant_vector):
        donor_vector = mutant_vector
        trial_vector = copy.deepcopy(target_vector)

        # Perform crossover based on a crossover rate (CR)
        for room_idx in range(len(self.rooms)):
            for timeslot_idx in range(len(self.timeslots)):
                crossover_gene = donor_vector[room_idx][timeslot_idx]
                if crossover_gene is not None and trial_vector[room_idx][timeslot_idx] == None:
                    if random.random() < self.CR:  # With probability CR, adopt from mutant
                        self.remove_previous_event_assignment(trial_vector, crossover_gene)
                        trial_vector[room_idx][timeslot_idx] = crossover_gene

        return trial_vector
    
    def remove_previous_event_assignment(self, chromosome, gene):
        for room_idx in range(len(self.rooms)):
            for timeslot_idx in range(len(self.timeslots)):
                if chromosome[room_idx][timeslot_idx] is not None and chromosome[room_idx][timeslot_idx] == gene:
                    chromosome[room_idx][timeslot_idx] = None
                    return

    
    def evaluate_fitness(self, chromosome):
        penalty = 0
        cost = 0
        
        # Check for hard constraint violations (H1-H5)
        penalty += self.check_room_constraints(chromosome)  # H1
        penalty += self.check_student_group_constraints(chromosome)  # H2
        # penalty += self.check_room_time_conflict(chromosome)  # H3
        penalty += self.check_lecturer_availability(chromosome)  # H4
        # penalty += self.check_valid_timeslot(chromosome)  # H5
        
        # Check for soft constraint violations (S1-S3)
        # cost += self.check_single_event_per_day(chromosome)  # S1
        # cost += self.check_consecutive_timeslots(chromosome)  # S2

        # Fitness is a combination of penalties and costs
        return penalty + cost   
    
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
                    # std_grp = input_data.getStudentGroup(class_event.student_grp)
                    # H1: Room capacity and type constraints
                    if room.room_type != course.required_room_type or class_event.student_group.no_students > room.capacity:
                        # print("failed - rooms must meet the capacity and type of the scheduled event")
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
                        # print(f"failed - {student_group.id} has more than one event in a timeslot")
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
                        # print(f"failed - {faculty_id} has more than one event in a timeslot")
                    else:
                        lecturer_watch.add(faculty_id)

        return penalty


    def check_single_event_per_day(self, chromosome):
        penalty = 0
        
        # Create a dictionary to track events per day for each student group
        events_per_day = {group.id: [0] * len(self.timeslots) for group in self.student_groups}

        for room_idx, room_schedule in enumerate(chromosome):
            for timeslot_idx, class_event in enumerate(room_schedule):
                if class_event is not None:  # Event scheduled
                    student_group = class_event.student_group
                    day_idx = timeslot_idx // input_data.hours  # Calculate which day this timeslot falls on
                    
                    # S1: Try to avoid scheduling more than one event per day for each student group
                    events_per_day[student_group.id][day_idx] += 1
                    if events_per_day[student_group.id][day_idx] == 1:
                        penalty += 0.05  # Soft penalty for multiple events on the same day for a group
                        print("failed - {student_group.id} has only one event per day")

        return penalty

    def check_consecutive_timeslots(self, chromosome):
        penalty = 0

        for room_idx, room_schedule in enumerate(chromosome):
            for timeslot_idx, class_event in enumerate(room_schedule):
                if class_event is not None:
                    course = input_data.getCourse(class_event.course_id)
                    
                    # S2: Multi-hour lectures should be scheduled in consecutive timeslots
                    required_hours = course.credits
                    if required_hours > 1:
                        # Check for consecutive timeslots for multi-hour courses
                        if timeslot_idx + 1 < len(self.timeslots) and room_schedule[timeslot_idx + 1] != class_event:
                            penalty += 0.05  # Soft penalty for non-consecutive timeslots in multi-hour lectures

        return penalty

    # Optional: Spread events over the week
    def check_spread_events(self, chromosome):
        penalty = 0
        group_event_days = {group.group_id: set() for group in self.student_groups}
        
        # S3: Try to spread the events throughout the week
        for room_idx, room_schedule in enumerate(chromosome):
            for timeslot_idx, class_event in enumerate(room_schedule):
                if class_event is not None:  # Event scheduled
                    student_group = class_event.student_group
                    day_idx = timeslot_idx // input_data.hours_per_day
                    
                    # Track which days each student group has events
                    group_event_days[student_group.group_id].add(day_idx)

        # Penalize student groups that have events tightly clustered in the week
        for group_id, event_days in group_event_days.items():
            if len(event_days) < len(self.timeslots) // input_data.days_per_week:
                penalty += 0.025  # Small penalty for clustering events

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
    # def check_lecturer_availability(self, chromosome):
    #     penalty = 0
    #     lecturer_schedule = {}

    #     for room_idx, room_schedule in enumerate(chromosome):
    #         for timeslot_idx, class_event in enumerate(room_schedule):
    #             if class_event is not None:  # Event scheduled
    #                 lecturer_id = class_event.faculty_id
                    
    #                 # H4: Ensure no lecturer is assigned to more than one event at the same time
    #                 if lecturer_id not in lecturer_schedule:
    #                     lecturer_schedule[lecturer_id] = [None] * len(self.timeslots)
                    
    #                 if lecturer_schedule[lecturer_id][timeslot_idx] is not None:
    #                     penalty += 1000  # Penalty for lecturer conflict (more than one event)

    #                 lecturer_schedule[lecturer_id][timeslot_idx] = class_event

    #     return penalty

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
        if trial_fitness < target_fitness:
            self.population[target_idx] = trial_vector


    def run(self, max_generations):
        # print(self.population[0])
        # self.print_all_timetables(self.population[0], input_data.days, input_data.hours)

        for generation in range(max_generations):
            for i in range(self.pop_size):
                # Step 1: Mutation
                # print(f"Generation {generation}, Individual {i}: Before mutation: {self.count_non_none(self.population[i])}")
                mutant_vector = self.mutate(i)
                # print(f"Generation {generation}, Individual {i}: After mutation: {self.count_non_none(mutant_vector)}")
                
                # Step 2: Crossover
                target_vector = self.population[i]
                # print(f"Generation {generation}, Individual {i}: Before crossover: {self.count_non_none(self.population[i])}")
                trial_vector = self.crossover(target_vector, mutant_vector)
                # print(f"Generation {generation}, Individual {i}: After crossover: {self.count_non_none(trial_vector)}")
                
                # Step 3: Evaluation and Selection
                # print(f"Generation {generation}, Individual {i}: Before selection: {self.count_non_none(self.population[i])}")
                self.select(i, trial_vector)
                # print(f"Generation {generation}, Individual {i}: After selection: {self.count_non_none(self.population[i])}")
                
            # Optional: Check if the best solution is good enough
            best_solution = min(self.population, key=self.evaluate_fitness)
            best_fitness = self.evaluate_fitness(best_solution)
            print(f"Best solution for generation {generation+1}/{max_generations} has a fitness of: {best_fitness}")
            # print(f"Best solution {best_solution}, count {self.count_non_none(best_solution)}")
            # self.print_all_timetables(best_solution, input_data.days, input_data.hours)
            
            if self.evaluate_fitness(best_solution) == 0:
                break  # Stop if the best solution has no constraint violations

        # print and return best individual for last generation
        best_solution = min(self.population, key=self.evaluate_fitness)
        print(f"Best solution {best_solution}, count {self.count_non_none(best_solution)}")
        self.print_all_timetables(best_solution, input_data.days, input_data.hours)
        return best_solution
        # print("")

    
    def print_timetable(self, individual, student_group, days, hours_per_day, day_start_time=9):
        # Create a blank timetable grid for the student group
        timetable = [['' for _ in range(days)] for _ in range(hours_per_day)]

        # Loop through the individual's chromosome to populate the timetable
        for room_idx, room_slots in enumerate(individual):
            # print(room_slots, "room slots")
            for timeslot_idx, event in enumerate(room_slots):
                # print(event, "event found")
                class_event = self.events_map.get(event)
                if class_event is not None and class_event.student_group.id == student_group.id:
                    
                    day = timeslot_idx // hours_per_day
                    hour = timeslot_idx % hours_per_day
                    if day < days:
                        timetable[hour][day] = f"Course: {class_event.course_id}, Lecturer: {class_event.faculty_id}, Room: {room_idx}"
        
        # Print the timetable for the student group
        print(f"Timetable for Student Group: {student_group.name}")
        print(" " * 15 + " | ".join([f"Day {d+1}" for d in range(days)]))
        print("-" * (20 + days * 15))
        
        # print(timetable)
        for hour in range(hours_per_day):
            time_label = f"{day_start_time + hour}:00"
            row = [timetable[hour][day] if timetable[hour][day] else "Free" for day in range(days)]
            print(f"{time_label:<15} | " + " | ".join(row))
        print("\n")

    def print_all_timetables(self, individual, days, hours_per_day, day_start_time=9):
        # Find all unique student groups in the individual
        student_groups = input_data.student_groups
        
        # Print timetable for each student group
        for student_group in student_groups:
            self.print_timetable(individual, student_group, days, hours_per_day, day_start_time)
            

pop_size = 50
F = 0.5
max_generations = 300
CR = 0.8

de = DifferentialEvolution(input_data, pop_size, F, CR)

de.initialize_population()

# print(de.population[0])
# print(de.evaluate_fitness(de.population[0]), "fitness of initial solution")
# de.print_all_timetables(de.population[0], input_data.days, input_data.hours)

de.run(max_generations)

