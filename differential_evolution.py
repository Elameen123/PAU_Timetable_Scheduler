import random
from typing import List
import copy
from utils import Utility
from entitities.Class import Class
from input_data import input_data
import numpy as np
from constraints import Constraints  # Import the Constraints class

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
        self.constraints = Constraints()  # Initialize constraints
        self.constraints.rooms = self.rooms
        self.constraints.timeslots = self.timeslots
        self.constraints.student_groups = self.student_groups
        self.constraints.events_map = self.events_map
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
        for _ in range(self.pop_size):
            chromosome = self.create_chromosome()
            population.append(chromosome)
        return np.array(population)

    def create_chromosome(self):
        # 2D NumPy array for chromosome where each row is a room and each column is a time slot
        chromosome = np.empty((len(self.rooms), len(self.timeslots)), dtype=object)

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
        return chromosome[room_idx][timeslot_idx] is None

    def is_room_suitable(self, room, course):
        if course is None:
            return False
        return room.room_type == course.required_room_type
    
    def hamming_distance(self, chromosome1, chromosome2):
        return np.sum(chromosome1.flatten() != chromosome2.flatten())

    def calculate_population_diversity(self):
        total_distance = 0
        comparisons = 0
        
        for i in range(self.pop_size):
            for j in range(i + 1, self.pop_size):
                total_distance += self.hamming_distance(self.population[i], self.population[j])
                comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0


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
        # for now TODO
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
        
        # Use the constraints class for evaluation
        penalty += self.constraints.check_room_constraints(chromosome)  # H1
        penalty += self.constraints.check_student_group_constraints(chromosome)  # H2
        penalty += self.constraints.check_lecturer_availability(chromosome)  # H4
        
        # Fitness is a combination of penalties and costs
        return penalty + cost



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
        if trial_fitness < target_fitness:
            self.population[target_idx] = trial_vector


    def run(self, max_generations):
        # np.random.seed(seed)
        self.initialize_population()
        fitness_history = []
        best_solution = self.population[0]
        diversity_history = []

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
                self.select(i, trial_vector)
                # print(f"Generation {generation}, Individual {i}: After selection: {self.count_non_none(self.population[i])}")
                
            # Optional: Check if the best solution is good enough
            best_solution = min(self.population, key=self.evaluate_fitness)
            best_fitness = self.evaluate_fitness(best_solution)
            fitness_history.append(best_fitness)

            # Measure and track diversity
            population_diversity = self.calculate_population_diversity()
            diversity_history.append(population_diversity)

            print(f"Best solution for generation {generation+1}/{max_generations} has a fitness of: {best_fitness}")

            if self.evaluate_fitness(best_solution) == self.desired_fitness:
                print(f"Solution with desired fitness of {self.desired_fitness} found at Generation {generation}! 🎉")
                break  # Stop if the best solution has no constraint violations

        # print and return best individual for last generation
        best_solution = min(self.population, key=self.evaluate_fitness)
        # self.print_all_timetables(best_solution, input_data.days, input_data.hours)
        return best_solution, fitness_history, generation, diversity_history


    def print_timetable(self, individual, student_group, days, hours_per_day, day_start_time=9):
        timetable = [["" for _ in range(days)] for _ in range(hours_per_day)]
        for room_idx, room_slots in enumerate(individual):
            for timeslot_idx, event in enumerate(room_slots):
                class_event = self.events_map.get(event)
                if class_event is not None and class_event.student_group.id == student_group.id:
                    day = timeslot_idx // hours_per_day
                    hour = timeslot_idx % hours_per_day
                    if day < days:
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

# Run the algorithm only when this file is executed directly
if __name__ == '__main__':
    print("Starting Differential Evolution Algorithm...")
    de = DifferentialEvolution(input_data, 50, 0.4, 0.9)
    best_solution, fitness_history, generation, diversity_history = de.run(200)
    print("Algorithm completed!")


    # Import Dash components only when needed
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

    # Start the Dash app
    print("Starting Dash web application...")
    app.run(debug=True)


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