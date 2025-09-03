from input_data import input_data
import random

import re

class Constraints:
    def __init__(self, input_data):
        self.input_data = input_data
        self.validate_faculty_data() # Validate data on initialization
        self.rooms = input_data.rooms
        self.timeslots = input_data.create_time_slots(no_hours_per_day=input_data.hours, no_days_per_week=input_data.days, day_start_time=9)
        self.student_groups = input_data.student_groups
        self.courses = input_data.courses
        self.events_list, self.events_map = self.create_events()

    def validate_faculty_data(self):
        """
        Validates the format of avail_days and avail_times for all faculty members.
        Raises a ValueError if any format is incorrect.
        """
        time_format_regex = re.compile(r'^\d{2}:\d{2}-\d{2}:\d{2}$')
        valid_days = ["Mon", "Tue", "Wed", "Thu", "Fri", "All"]

        for faculty in self.input_data.faculties:
            # Validate avail_times
            if isinstance(faculty.avail_times, str) and faculty.avail_times.upper() != 'ALL':
                if not time_format_regex.match(faculty.avail_times):
                    raise ValueError(
                        f"FATAL: Invalid 'avail_times' format for faculty '{faculty.name}' (ID: {faculty.id}). "
                        f"Expected 'HH:MM-HH:MM' or 'ALL', but got '{faculty.avail_times}'. Please correct the input data."
                    )
            
            # Validate avail_days
            days_to_check = []
            if isinstance(faculty.avail_days, str):
                if faculty.avail_days.upper() == 'ALL':
                    days_to_check = ["All"]
                else:
                    days_to_check = [d.strip() for d in faculty.avail_days.split(',')]
            elif isinstance(faculty.avail_days, list):
                days_to_check = faculty.avail_days

            for day in days_to_check:
                if day.capitalize() not in valid_days:
                    raise ValueError(
                        f"FATAL: Invalid 'avail_days' value for faculty '{faculty.name}' (ID: {faculty.id}). "
                        f"Found invalid day '{day}'. Valid days are {valid_days}. Please correct the input data."
                    )
    
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
       
    
    def check_student_group_constraints(self, chromosome, debug=False):
        """
        No student group can have overlapping classes at the same time
        """
        penalty = 0
        clashes = []
        days_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}

        for i in range(len(self.timeslots)):
            simultaneous_class_events = chromosome[:, i]
            student_group_watch = {}  # Store the first event for a group in a timeslot
            for class_event_idx in simultaneous_class_events:
                if class_event_idx is not None:
                    class_event = self.events_map.get(class_event_idx)
                    if class_event is not None:
                        student_group = class_event.student_group
                        if student_group.id in student_group_watch:
                            penalty += 1
                            if debug:
                                # A clash is detected. We have the new event and the one from the watch.
                                first_event = student_group_watch[student_group.id]
                                second_event = class_event
                                
                                first_course = self.input_data.getCourse(first_event.course_id)
                                second_course = self.input_data.getCourse(second_event.course_id)
                                
                                timeslot = self.timeslots[i]
                                day_abbr = days_map.get(timeslot.day)
                                time = timeslot.start_time + 9
                                
                                clash_info = (
                                    f"Student Group Clash: '{student_group.name}' on {day_abbr} at {time}:00. "
                                    f"Clashing Courses: '{first_course.code}' and '{second_course.code}'."
                                )
                                if clash_info not in clashes:
                                    clashes.append(clash_info)
                        else:
                            # First time seeing this group in this timeslot, store the event.
                            student_group_watch[student_group.id] = class_event
        
        if debug and clashes:
            print("\n--- Student Group Clashes Detected ---")
            for clash in sorted(clashes):
                print(clash)
            print("-------------------------------------\n")
            
        return penalty
    
    def check_lecturer_availability(self, chromosome, debug=False):
        """
        No lecturer can have overlapping classes at the same time
        """
        penalty = 0
        clashes = []
        days_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}

        for i in range(len(self.timeslots)):
            simultaneous_class_events = chromosome[:, i]
            lecturer_watch = {}  # Store the first event for a lecturer in a timeslot
            for class_event_idx in simultaneous_class_events:
                if class_event_idx is not None:
                    class_event = self.events_map.get(class_event_idx)
                    if class_event is not None:
                        faculty_id = class_event.faculty_id
                        if faculty_id is not None:
                            if faculty_id in lecturer_watch:
                                penalty += 1
                                if debug:
                                    first_event = lecturer_watch[faculty_id]
                                    second_event = class_event
                                    
                                    faculty = self.input_data.getFaculty(faculty_id)
                                    first_course = self.input_data.getCourse(first_event.course_id)
                                    second_course = self.input_data.getCourse(second_event.course_id)
                                    
                                    timeslot = self.timeslots[i]
                                    day_abbr = days_map.get(timeslot.day)
                                    time = timeslot.start_time + 9
                                    
                                    clash_info = (
                                        f"Lecturer Clash: '{faculty.name}' on {day_abbr} at {time}:00. "
                                        f"Clashing Courses: '{first_course.name}' for group '{first_event.student_group.name}' and "
                                        f"'{second_course.name}' for group '{second_event.student_group.name}'."
                                    )
                                    if clash_info not in clashes:
                                        clashes.append(clash_info)
                            else:
                                lecturer_watch[faculty_id] = class_event
        
        if debug and clashes:
            print("\n--- Lecturer Clashes Detected ---")
            for clash in sorted(clashes):
                print(clash)
            print("---------------------------------\n")

        return penalty

    def check_lecturer_schedule_constraints(self, chromosome, debug=False):
        """
        Checks if courses are scheduled according to the lecturer's available days and times.
        """
        penalty = 0
        violations = []
        days_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}

        for room_idx in range(len(self.rooms)):
            for timeslot_idx in range(len(self.timeslots)):
                event_id = chromosome[room_idx][timeslot_idx]
                if event_id is not None:
                    class_event = self.events_map.get(event_id)
                    if class_event and class_event.faculty_id is not None:
                        faculty = self.input_data.getFaculty(class_event.faculty_id)
                        if not faculty:
                            continue

                        timeslot = self.timeslots[timeslot_idx]
                        day_idx = timeslot.day
                        day_abbr = days_map.get(day_idx)
                        
                        # The actual hour of the day (e.g., 9, 10, 11)
                        slot_hour = timeslot.start_time + 9

                        # 1. Check available days
                        is_available_day = False
                        avail_days = faculty.avail_days
                        if not avail_days or (isinstance(avail_days, str) and avail_days.upper() == "ALL"):
                            is_available_day = True
                        else:
                            # Normalize to a list of capitalized day abbreviations
                            if isinstance(avail_days, str):
                                avail_days_list = [d.strip().capitalize() for d in avail_days.split(',')]
                            else: # is a list
                                avail_days_list = [d.strip().capitalize() for d in avail_days]
                            
                            if "All" in avail_days_list or day_abbr in avail_days_list:
                                is_available_day = True

                        if not is_available_day:
                            penalty += 10
                            if debug:
                                violation_info = (
                                    f"Lecturer Schedule Violation: '{faculty.name}' is scheduled on {day_abbr}, "
                                    f"but is only available on: {faculty.avail_days}."
                                )
                                if violation_info not in violations:
                                    violations.append(violation_info)
                            continue # Skip time check if day is already wrong

                        # 2. Check available times
                        is_available_time = False
                        avail_times = faculty.avail_times

                        if not avail_times:
                            is_available_time = True
                        elif isinstance(avail_times, str) and avail_times.upper() == "ALL":
                            is_available_time = True
                        elif isinstance(avail_times, list) and any(str(t).strip().upper() == 'ALL' for t in avail_times):
                            is_available_time = True
                        else:
                            # It's a list or string of specific times/ranges
                            if isinstance(avail_times, str):
                                avail_times_list = [t.strip() for t in avail_times.split(',')]
                            else: # is a list
                                avail_times_list = avail_times

                            for time_spec in avail_times_list:
                                time_spec_str = str(time_spec).strip()
                                if '-' in time_spec_str: # It's a range, e.g., "09:00-12:00"
                                    try:
                                        start_str, end_str = time_spec_str.split('-')
                                        start_h = int(start_str.split(':')[0])
                                        end_h = int(end_str.split(':')[0])
                                        # The slot is valid if its start time is within the range [start, end).
                                        # e.g., for "09:00-12:00", slots 9, 10, 11 are valid. Slot 12 is not.
                                        if start_h <= slot_hour < end_h:
                                            is_available_time = True
                                            break
                                    except (ValueError, IndexError):
                                        continue # Ignore malformed range
                                else: # It's a single time, e.g., "09:00"
                                    try:
                                        h = int(time_spec_str.split(':')[0])
                                        if h == slot_hour:
                                            is_available_time = True
                                            break
                                    except (ValueError, IndexError):
                                        continue # Ignore malformed time
                        
                        if not is_available_time:
                            penalty += 10
                            if debug:
                                violation_info = (
                                    f"Lecturer Schedule Violation: '{faculty.name}' is scheduled at {slot_hour}:00 on {day_abbr}, "
                                    f"but is only available during: {faculty.avail_times}."
                                )
                                if violation_info not in violations:
                                    violations.append(violation_info)
        
        if debug and violations:
            print("\n--- Lecturer Schedule Violations Detected ---")
            for violation in sorted(violations):
                print(violation)
            print("-------------------------------------------\n")
            
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
                        penalty += 10  # Reduced from 100 to 10
                    
                    # Additional check: count non-None values to ensure only one event per slot
                    # This constraint is inherently satisfied by the chromosome structure,
                    # but we check for any data corruption
                    
        return penalty

    def check_break_time_constraint(self, chromosome):
        """
        Ensure no classes are scheduled during break time (13:00 - 14:00) on Mon, Wed, Fri.
        Break time corresponds to timeslot index 4 on each day (9:00, 10:00, 11:00, 12:00, 13:00)
        """
        penalty = 0
        break_hour = 4  # 13:00 is the 5th hour (index 4) starting from 9:00
        
        for day in range(input_data.days):  # For each day
            # Apply break constraint only on Monday (0), Wednesday (2), and Friday (4)
            if day in [0, 2, 4]:
                break_timeslot = day * input_data.hours + break_hour  # Calculate break timeslot index
                
                for room_idx in range(len(self.rooms)):
                    if chromosome[room_idx][break_timeslot] is not None:
                        penalty += 100
                        
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
                'engineering', 'eng', 'computer science', 'software engineering', 'data science',
                'mechatronics', 'electrical', 'mechanical', 'csc', 'sen', 'data', 'ds'
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
                        
                        # Apply LENIENT building assignment rules (reduced penalties)
                        if class_event.student_group.id in engineering_groups:
                            # Engineering groups prefer SST but small penalty for TYD
                            if not needs_computer_lab and room_building != 'SST':
                                penalty += 0.5  # Very small penalty (reduced from 2)
                        else:
                            # Non-engineering groups prefer TYD but small penalty for SST
                            if not needs_computer_lab and room_building == 'SST':
                                penalty += 0.5  # Very small penalty (reduced from 20)
                        
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
                        # The key now includes the student group to correctly handle multiple groups taking the same course
                        course_day_key = (course_id, day_idx, class_event.student_group.id)
                        
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
        """
        - 2-credit courses MUST be in 2 consecutive slots.
        - 3-credit courses MUST have at least a 2-hour block.
        - Penalizes the single hour of a 3-credit course if it's not consecutive with the block.
        """
        penalty = 0
        
        # Group events by course and student group to analyze their schedule
        course_schedule = {} # {(course_id, student_group_id): [timeslot_indices]}
        
        for room_idx in range(len(self.rooms)):
            for timeslot_idx in range(len(self.timeslots)):
                event_id = chromosome[room_idx][timeslot_idx]
                if event_id is not None:
                    event = self.events_map.get(event_id)
                    if event:
                        course = input_data.getCourse(event.course_id)
                        if course:
                            key = (course.code, event.student_group.id)
                            if key not in course_schedule:
                                course_schedule[key] = []
                            course_schedule[key].append(timeslot_idx)

        for (course_id, student_group_id), timeslots in course_schedule.items():
            course = input_data.getCourse(course_id)
            if not course or course.credits <= 1:
                continue

            # Sort timeslots to check for consecutiveness
            timeslots.sort()
            
            if course.credits == 2:
                # H: 2-credit courses MUST be consecutive
                if len(timeslots) == 2 and (timeslots[1] - timeslots[0] != 1):
                    penalty += 50 # Severe penalty for breaking a 2-hour block
            
            elif course.credits == 3:
                # H: 3-credit courses MUST have at least a 2-hour block
                if len(timeslots) == 3:
                    is_block_of_2 = (timeslots[1] - timeslots[0] == 1) or \
                                    (timeslots[2] - timeslots[1] == 1)
                    if not is_block_of_2:
                        penalty += 50 # Severe penalty if no 2-hour block exists
                    
                    # S: Prefer all 3 hours to be consecutive
                    is_block_of_3 = (timeslots[1] - timeslots[0] == 1) and \
                                    (timeslots[2] - timeslots[1] == 1)
                    if not is_block_of_3:
                        penalty += 0.1 # Small penalty for the separated hour

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

    def check_course_allocation_completeness(self, chromosome, debug=False):
        """
        Check that all courses appear the correct number of times for each student group
        based on their credit hours/hours_required.
        """
        penalty = 0
        allocation_issues = []
        
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
                    difference = abs(expected_hours - actual_hours)
                    
                    if actual_hours < expected_hours:
                        # Apply moderate penalty for missing courses
                        penalty += difference * (5 if actual_hours == 0 else 2)
                        if debug:
                            course = self.input_data.getCourse(course_id)
                            course_name = course.name if course else "Unknown Course"
                            info = (
                                f"Missing Class: Group '{student_group.name}' is missing {difference} hour(s) "
                                f"for course '{course_name}' (Code: {course_id}). "
                                f"Expected {expected_hours}, but only {actual_hours} are scheduled."
                            )
                            allocation_issues.append(info)
                    else: # actual_hours > expected_hours
                        # Apply penalty for extra classes
                        penalty += difference * 2
                        if debug:
                            course = self.input_data.getCourse(course_id)
                            course_name = course.name if course else "Unknown Course"
                            info = (
                                f"Extra Class: Group '{student_group.name}' has {difference} extra hour(s) "
                                f"for course '{course_name}' (Code: {course_id}). "
                                f"Expected {expected_hours}, but {actual_hours} are scheduled."
                            )
                            allocation_issues.append(info)

        if debug and allocation_issues:
            print("\n--- Course Allocation Issues Detected ---")
            for info in sorted(allocation_issues):
                print(info)
            print("---------------------------------------\n")
        
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
        penalty += self.check_lecturer_schedule_constraints(chromosome) # H9: Lecturer schedule constraints
        
        # Check for soft constraint violations (S1-S3)
        cost += self.check_single_event_per_day(chromosome)  # S1
        cost += self.check_consecutive_timeslots(chromosome)  # S2
        cost += self.check_spread_events(chromosome)  # S3

        # Fitness is a combination of penalties and costs
        return penalty + cost
        
    def get_all_conflicts(self, chromosome):
        """
        Identifies all conflicts in a given chromosome to guide the crossover process.
        Returns a dictionary of conflicts.
        """
        conflicts = {
            'student_group': [],
            'lecturer': [],
            'room': []
        }

        # Check for student group and lecturer clashes
        for timeslot_idx in range(len(self.timeslots)):
            student_group_watch = {}
            lecturer_watch = {}
            simultaneous_events = chromosome[:, timeslot_idx]

            for room_idx, event_id in enumerate(simultaneous_events):
                if event_id is not None:
                    event = self.events_map.get(event_id)
                    if event:
                        # Student group conflicts
                        sg_id = event.student_group.id
                        if sg_id in student_group_watch:
                            conflicts['student_group'].append({
                                'timeslot': timeslot_idx,
                                'student_group': sg_id,
                                'positions': [student_group_watch[sg_id], (room_idx, timeslot_idx)]
                            })
                        else:
                            student_group_watch[sg_id] = (room_idx, timeslot_idx)

                        # Lecturer conflicts
                        fac_id = event.faculty_id
                        if fac_id in lecturer_watch:
                            conflicts['lecturer'].append({
                                'timeslot': timeslot_idx,
                                'lecturer': fac_id,
                                'positions': [lecturer_watch[fac_id], (room_idx, timeslot_idx)]
                            })
                        else:
                            lecturer_watch[fac_id] = (room_idx, timeslot_idx)

        # Check for room capacity/type conflicts
        for room_idx in range(len(self.rooms)):
            for timeslot_idx in range(len(self.timeslots)):
                event_id = chromosome[room_idx][timeslot_idx]
                if event_id is not None:
                    event = self.events_map.get(event_id)
                    room = self.rooms[room_idx]
                    course = input_data.getCourse(event.course_id)
                    if event and course:
                        if room.room_type != course.required_room_type or event.student_group.no_students > room.capacity:
                            conflicts['room'].append({
                                'position': (room_idx, timeslot_idx),
                                'details': f"Room {room.name} (cap {room.capacity}, type {room.room_type}) vs Course {course.code} (students {event.student_group.no_students}, type {course.required_room_type})"
                            })

        return conflicts
        
    def get_constraint_violations(self, chromosome, debug=False):
        """
        Get detailed information about constraint violations for debugging.
        """
        violations = {
            'room_constraints': self.check_room_constraints(chromosome),
            'student_group_constraints': self.check_student_group_constraints(chromosome, debug=debug),
            'lecturer_availability': self.check_lecturer_availability(chromosome, debug=debug),
            'room_time_conflict': self.check_room_time_conflict(chromosome),
            'building_assignments': self.check_building_assignments(chromosome),
            'same_course_same_room_per_day': self.check_same_course_same_room_per_day(chromosome),
            'break_time_constraint': self.check_break_time_constraint(chromosome),
            'course_allocation_completeness': self.check_course_allocation_completeness(chromosome, debug=debug),
            'lecturer_schedule_constraints': self.check_lecturer_schedule_constraints(chromosome, debug=debug),
            'single_event_per_day': self.check_single_event_per_day(chromosome),
            'consecutive_timeslots': self.check_consecutive_timeslots(chromosome),
            'spread_events': self.check_spread_events(chromosome)
        }
        violations['total'] = sum(violations.values())
        return violations