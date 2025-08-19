    
class Constraints:    
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