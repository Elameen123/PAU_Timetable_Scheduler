from input_data import input_data
import random

class Constraints:
    def __init__(self, input_data):
        self.rooms = input_data.rooms
        self.timeslots = input_data.create_time_slots(no_hours_per_day=input_data.hours, no_days_per_week=input_data.days, day_start_time=9)
        self.student_groups = input_data.student_groups
        self.courses = input_data.courses
        self.events_list, self.events_map = self.create_events()
    
    def create_events(self):
        """Create events list and mapping similar to genetic algorithm"""
        events_list = []
        event_map = {}
        
        from entitities.Class import Class
        
        idx = 0
        for student_group in self.student_groups:
            for i in range(student_group.no_courses):
                hourcount = 1 
                while hourcount <= student_group.hours_required[i]:
                    event = Class(student_group, student_group.teacherIDS[i], student_group.courseIDs[i])
                    events_list.append(event)
                    
                    # Add the event to the index map with the current index
                    event_map[idx] = event
                    idx += 1
                    hourcount += 1
                    
        return events_list, event_map    
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
                    # H1a: Room type constraints
                    if room.room_type != course.required_room_type:
                        point += 1
                    # H1b: Room capacity constraints - student group must fit in room
                    if class_event.student_group.no_students > room.capacity:
                        point += 1

        return point
       
    
    def check_student_group_constraints(self, chromosome):
        """
        No student group can have overlapping classes at the same time
        """
        penalty = 0
        for i in range(len(self.timeslots)):
            simultaneous_class_events = chromosome[:, i]
            student_group_watch = set()
            for class_event_idx in simultaneous_class_events:
                if class_event_idx is not None:
                    class_event = self.events_map.get(class_event_idx)
                    if class_event is not None:  # Added safety check
                        student_group = class_event.student_group
                        if student_group.id in student_group_watch:
                            penalty += 1  # Student group has overlapping classes
                        else:
                            student_group_watch.add(student_group.id)

        return penalty
    
    def check_lecturer_availability(self, chromosome):
        """
        No lecturer can have overlapping classes at the same time
        """
        penalty = 0
        for i in range(len(self.timeslots)):
            simultaneous_class_events = chromosome[:, i]
            lecturer_watch = set()
            for class_event_idx in simultaneous_class_events:
                if class_event_idx is not None:
                    class_event = self.events_map.get(class_event_idx)
                    if class_event is not None:  # Added safety check
                        faculty_id = class_event.faculty_id
                        if faculty_id is not None:  # Check if faculty_id exists
                            if faculty_id in lecturer_watch:
                                penalty += 1  # Lecturer has overlapping classes
                            else:
                                lecturer_watch.add(faculty_id)

        return penalty

    def check_room_time_conflict(self, chromosome):
        """
        Ensure only one event is scheduled per room per timeslot
        """
        penalty = 0
        for room_idx in range(len(self.rooms)):
            for timeslot_idx in range(len(self.timeslots)):
                event = chromosome[room_idx][timeslot_idx]
                if event is not None:
                    # Check if event is somehow a list (multiple events in same slot)
                    if isinstance(event, list) and len(event) > 1:
                        penalty += 100  # High penalty for multiple events in same room-time slot
                    
                    # Additional check: count non-None values to ensure only one event per slot
                    # This constraint is inherently satisfied by the chromosome structure,
                    # but we check for any data corruption
                    
        return penalty

    def check_break_time_constraint(self, chromosome):
        """
        Ensure no classes are scheduled during break time (13:00 - 14:00)
        Break time corresponds to timeslot index 4 on each day (9:00, 10:00, 11:00, 12:00, 13:00)
        """
        penalty = 0
        break_hour = 4  # 13:00 is the 5th hour (index 4) starting from 9:00
        
        for day in range(input_data.days):  # For each day
            break_timeslot = day * input_data.hours + break_hour  # Calculate break timeslot index
            
            for room_idx in range(len(self.rooms)):
                if chromosome[room_idx][break_timeslot] is not None:
                    penalty += 1000  # Very high penalty for scheduling during break time
                    
        return penalty

    def check_building_assignments(self, chromosome):
        """
        STRICT building assignment rules:
        - Engineering/Computer Science/Software Engineering groups MUST be in SST building
        - All other groups MUST be in TYD building  
        - Exception: Computer lab courses can use any computer lab regardless of building
          (since there are only 2 computer labs in SST but more courses may need them)
        """
        penalty = 0
        
        # Identify engineering groups more comprehensively
        engineering_groups = []
        for student_group in self.student_groups:
            group_name = student_group.name.lower()
            # Check for engineering, computer science, software engineering keywords
            if any(keyword in group_name for keyword in [
                'engineering', 'eng', 'computer science', 'software engineering',
                'mechatronics', 'electrical', 'mechanical', 'csc', 'sen'
            ]):
                engineering_groups.append(student_group.id)
        
        for room_idx in range(len(self.rooms)):
            for timeslot_idx in range(len(self.timeslots)):
                event_id = chromosome[room_idx][timeslot_idx]
                if event_id is not None:
                    class_event = self.events_map.get(event_id)
                    room = self.rooms[room_idx]
                    course = input_data.getCourse(class_event.course_id)
                    
                    if class_event and course:
                        # Computer lab exception: Check if this needs a computer lab
                        needs_computer_lab = (
                            course.required_room_type.lower() in ['comp lab', 'computer_lab'] or
                            room.room_type.lower() in ['comp lab', 'computer_lab'] or
                            'lab' in course.name.lower() and ('computer' in course.name.lower() or 
                                                             'programming' in course.name.lower() or
                                                             'software' in course.name.lower())
                        )
                        
                        # Get room's building
                        room_building = None
                        if hasattr(room, 'building'):
                            room_building = room.building.upper()
                        elif hasattr(room, 'name') and room.name:
                            room_name = room.name.upper()
                            if 'SST' in room_name:
                                room_building = 'SST'
                            elif 'TYD' in room_name:
                                room_building = 'TYD'
                        elif hasattr(room, 'room_id'):
                            room_id = str(room.room_id).upper()
                            if 'SST' in room_id:
                                room_building = 'SST'
                            elif 'TYD' in room_id:
                                room_building = 'TYD'
                        
                        # Apply STRICT building assignment rules
                        if class_event.student_group.id in engineering_groups:
                            # Engineering groups should be in SST but allow some flexibility
                            if not needs_computer_lab and room_building != 'SST':
                                penalty += 10  # Moderate penalty for engineering in TYD
                        else:
                            # Non-engineering groups MUST NEVER be in SST (except computer labs)
                            if not needs_computer_lab and room_building == 'SST':
                                penalty += 200  # Very high penalty for non-engineering in SST
                        
        return penalty

    def check_same_course_same_room_per_day(self, chromosome):
        """
        Same course appearing multiple times on same day must be in same room.
        """
        penalty = 0
        course_day_rooms = {}  # {(course_id, day): set_of_rooms}
        
        for room_idx in range(len(self.rooms)):
            for timeslot_idx in range(len(self.timeslots)):
                event_id = chromosome[room_idx][timeslot_idx]
                if event_id is not None:
                    class_event = self.events_map.get(event_id)
                    if class_event:
                        day_idx = timeslot_idx // input_data.hours
                        # Use the correct course identifier
                        course = input_data.getCourse(class_event.course_id)
                        course_id = getattr(course, 'course_id', None) or getattr(course, 'id', None) or getattr(course, 'code', None) if course else class_event.course_id
                        course_day_key = (course_id, day_idx)
                        
                        if course_day_key not in course_day_rooms:
                            course_day_rooms[course_day_key] = set()
                        course_day_rooms[course_day_key].add(room_idx)
        
        # Penalize courses that appear in multiple rooms on same day
        for course_day_key, rooms_used in course_day_rooms.items():
            if len(rooms_used) > 1:
                penalty += 5 * (len(rooms_used) - 1)  # Reduced penalty for room inconsistency
        
        return penalty


    def check_single_event_per_day(self, chromosome):
        penalty = 0
        
        # Create a dictionary to track events per day for each student group
        events_per_day = {group.id: [0] * input_data.days for group in self.student_groups}

        for room_idx in range(len(self.rooms)):
            for timeslot_idx in range(len(self.timeslots)):
                class_event_idx = chromosome[room_idx][timeslot_idx]
                if class_event_idx is not None:  # Event scheduled
                    class_event = self.events_map.get(class_event_idx)
                    if class_event is not None:
                        student_group = class_event.student_group
                        day_idx = timeslot_idx // input_data.hours  # Calculate which day this timeslot falls on
                        
                        # S1: Try to avoid scheduling more than one event per day for each student group
                        events_per_day[student_group.id][day_idx] += 1
                        if events_per_day[student_group.id][day_idx] > 1:
                            penalty += 0.05  # Soft penalty for multiple events on the same day for a group

        return penalty

    def check_consecutive_timeslots(self, chromosome):
        penalty = 0

        for room_idx in range(len(self.rooms)):
            for timeslot_idx in range(len(self.timeslots)):
                class_event_idx = chromosome[room_idx][timeslot_idx]
                if class_event_idx is not None:
                    class_event = self.events_map.get(class_event_idx)
                    if class_event is not None:
                        course = input_data.getCourse(class_event.course_id)
                        
                        # S2: Multi-hour lectures should be scheduled in consecutive timeslots
                        if course and course.credits > 1:
                            # Check if next timeslot for same course is consecutive
                            # This is a simplified check - you may need more complex logic
                            # based on your specific requirements
                            penalty += 0.05

        return penalty

    # Optional: Spread events over the week
    def check_spread_events(self, chromosome):
        penalty = 0
        group_event_days = {group.id: set() for group in self.student_groups}
        
        # S3: Try to spread the events throughout the week
        for room_idx in range(len(self.rooms)):
            for timeslot_idx in range(len(self.timeslots)):
                class_event_idx = chromosome[room_idx][timeslot_idx]
                if class_event_idx is not None:  # Event scheduled
                    class_event = self.events_map.get(class_event_idx)
                    if class_event is not None:
                        student_group = class_event.student_group
                        day_idx = timeslot_idx // input_data.hours
                        
                        # Track which days each student group has events
                        group_event_days[student_group.id].add(day_idx)

        # Penalize student groups that have events tightly clustered in the week
        for group_id, event_days in group_event_days.items():
            if len(event_days) < input_data.days // 2:  # If events are clustered in less than half the week
                penalty += 0.025  # Small penalty for clustering events

        return penalty

    def check_course_allocation_completeness(self, chromosome):
        """
        Check that all courses appear the correct number of times for each student group
        based on their credit hours/hours_required.
        """
        penalty = 0
        
        for student_group in self.student_groups:
            # Count actual course occurrences
            course_counts = {}
            for room_idx in range(len(self.rooms)):
                for timeslot_idx in range(len(self.timeslots)):
                    event_id = chromosome[room_idx][timeslot_idx]
                    if event_id is not None:
                        class_event = self.events_map.get(event_id)
                        if class_event and class_event.student_group.id == student_group.id:
                            course_id = class_event.course_id
                            course_counts[course_id] = course_counts.get(course_id, 0) + 1
            
            # Check expected vs actual course occurrences
            for i, course_id in enumerate(student_group.courseIDs):
                expected_hours = student_group.hours_required[i]
                actual_hours = course_counts.get(course_id, 0)
                
                if actual_hours != expected_hours:
                    # Apply stronger penalty for missing courses to force proper allocation
                    difference = abs(expected_hours - actual_hours)
                    if actual_hours == 0:
                        penalty += difference * 20  # Strong penalty for completely missing courses
                    else:
                        penalty += difference * 5   # Moderate penalty for imbalances
        
        return penalty

    def evaluate_fitness(self, chromosome):
        """
        Evaluate the overall fitness of a chromosome by checking all constraints.
        Lower values indicate better fitness.
        """
        penalty = 0
        cost = 0
        
        # Check for hard constraint violations (H1-H8)
        penalty += self.check_room_constraints(chromosome)  # H1: Room capacity and type
        penalty += self.check_student_group_constraints(chromosome)  # H2: No student overlaps
        penalty += self.check_lecturer_availability(chromosome)  # H3: No lecturer overlaps
        penalty += self.check_room_time_conflict(chromosome)  # H4: One event per room-time slot
        penalty += self.check_building_assignments(chromosome)  # H5: Building assignments
        penalty += self.check_same_course_same_room_per_day(chromosome)  # H6: Same course same room per day
        penalty += self.check_break_time_constraint(chromosome)  # H7: No classes during break time
        penalty += self.check_course_allocation_completeness(chromosome)  # H8: All courses allocated correctly
        
        # Check for soft constraint violations (S1-S3)
        cost += self.check_single_event_per_day(chromosome)  # S1
        cost += self.check_consecutive_timeslots(chromosome)  # S2
        cost += self.check_spread_events(chromosome)  # S3

        # Fitness is a combination of penalties and costs
        return penalty + cost
        
    def get_constraint_violations(self, chromosome):
        """
        Get detailed information about constraint violations for debugging.
        """
        violations = {
            'room_constraints': self.check_room_constraints(chromosome),
            'student_group_constraints': self.check_student_group_constraints(chromosome),
            'lecturer_availability': self.check_lecturer_availability(chromosome),
            'room_time_conflict': self.check_room_time_conflict(chromosome),
            'building_assignments': self.check_building_assignments(chromosome),
            'same_course_same_room_per_day': self.check_same_course_same_room_per_day(chromosome),
            'break_time_constraint': self.check_break_time_constraint(chromosome),
            'course_allocation_completeness': self.check_course_allocation_completeness(chromosome),
            'single_event_per_day': self.check_single_event_per_day(chromosome),
            'consecutive_timeslots': self.check_consecutive_timeslots(chromosome),
            'spread_events': self.check_spread_events(chromosome),
            'course_allocation_completeness': self.check_course_allocation_completeness(chromosome)
        }
        violations['total'] = sum(violations.values())
        return violations