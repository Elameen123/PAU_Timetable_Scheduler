# input_data_api.py
"""
API-compatible version of input_data that can be initialized from JSON data
instead of static files.
"""

from typing import List, Dict, Any
from entitities.course import Course
from entitities.faculty import Faculty
from entitities.room import Room
from entitities.student_group import StudentGroup
from entitities.Class import Class
from entitities.time_slot import TimeSlot


class InputData:
    def __init__(self) -> None:
        self.courses = []
        self.rooms = []
        self.student_groups = []    
        self.faculties = []
        self.classes = []
        self.nostudentgroup = 0
        # Keep API defaults aligned with the non-API scheduler (see input_data.py).
        # Can be overridden via initialize_input_data_from_json(...).
        self.hours = 10
        self.days = 5

    def addCourse(self, name: str, code: str, credits: int, student_groupsID: List[str], facultyId, required_room_type: str):
        self.courses.append(Course(name, code, credits, student_groupsID, facultyId, required_room_type))

    def addRoom(self, Id: str, name: str, capacity: int, room_type: str, building: str):
        self.rooms.append(Room(Id, name, capacity, room_type, building))

    def addStudentGroup(self, id: str, name: str, no_students: int, courseIDs: str, teacherIDS: str, hours_required: List[int], building: str = None):
        normalized_teachers = []
        for t in (teacherIDS or []):
            try:
                ts = str(t).strip()
                if '@' in ts:
                    ts = ts.lower()
                normalized_teachers.append(ts)
            except Exception:
                normalized_teachers.append(t)
        self.student_groups.append(StudentGroup(id, name, no_students, courseIDs, normalized_teachers, hours_required, building=building))

    def addFaculty(self, id: str, name: str, department: str, courseID: str, avail_days=None, avail_times=None):
        # If not provided, default to ALL (matches the static JSON dataset and core scheduler).
        if not avail_days:
            avail_days = ["ALL"]
        if not avail_times:
            avail_times = ["ALL"]
        normalized_id = str(id).strip()
        if '@' in normalized_id:
            normalized_id = normalized_id.lower()
        self.faculties.append(Faculty(normalized_id, name, department, courseID, avail_days, avail_times))

    def getCourse(self, code: str) -> Course:
        for course in self.courses:
            if course.code == code:
                return course
        return None
    
    def getRoom(self, Id: str) -> Room:
        for room in self.rooms:
            if room.Id == Id:
                return room
        return None
    
    def getStudentGroup(self, id: str) -> StudentGroup:
        for student_group in self.student_groups:
            if student_group.id == id:
                return student_group
        return None
    
    def getFaculty(self, id: str) -> Faculty:
        target = str(id).strip()
        target_norm = target.lower() if '@' in target else target
        for faculty in self.faculties:
            fid = getattr(faculty, 'faculty_id', None)
            if fid is None:
                continue
            fid_s = str(fid).strip()
            fid_norm = fid_s.lower() if '@' in fid_s else fid_s
            if fid_norm == target_norm:
                return faculty
        return None
    
    def create_time_slots(self, no_hours_per_day, no_days_per_week, day_start_time):
        """
        Create TimeSlot entries.
        Note: start_time is stored as an integer hour index within the day (0-based),
        e.g., 0 => 9:00, 1 => 10:00 when day_start_time=9.
        This matches constraints.py which uses arithmetic like `timeslot.start_time + 9`.
        """
        time_slots = []
        for day in range(no_days_per_week):
            for hour in range(no_hours_per_day):
                # Store 0-based hour index for numeric arithmetic in constraints
                start_time_index = hour
                time_slots.append(TimeSlot(
                    id=len(time_slots), 
                    day=day, 
                    start_time=start_time_index, 
                    available=True
                ))
        return time_slots
    
    def assign_class_to_course_and_faculty(self, student_group: StudentGroup):
        for course_x in student_group.courseIDs:
            for course in self.courses:
                if course_x == course.code:
                    facultyId = course.facultyId
                    self.classes.append(Class(student_group.id, facultyId, course.code))

    def get_data_summary(self) -> Dict[str, Any]:
        """Return summary statistics for API responses"""
        return {
            'courses': len(self.courses),
            'rooms': len(self.rooms),
            'student_groups': len(self.student_groups),
            'faculties': len(self.faculties),
            'total_student_capacity': sum(sg.no_students for sg in self.student_groups),
            'total_room_capacity': sum(r.capacity for r in self.rooms)
        }


def initialize_input_data_from_json(json_data: Dict[str, Any]) -> InputData:
    """
    Initialize InputData instance from transformer JSON output.
    This replaces the static file loading approach.
    """
    input_data = InputData()

    # Allow the caller to override calendar configuration.
    # If not present, we keep the API defaults (aligned with the core scheduler).
    try:
        if 'hours' in json_data and json_data['hours'] is not None:
            input_data.hours = int(json_data['hours'])
        if 'days' in json_data and json_data['days'] is not None:
            input_data.days = int(json_data['days'])
    except (TypeError, ValueError):
        # Fall back to defaults if invalid values were provided.
        pass
    
    # Load courses
    for course_data in json_data.get('courses', []):
        input_data.addCourse(
            name=course_data['name'],
            code=course_data['code'],
            credits=course_data['credits'],
            student_groupsID=course_data['student_groupsID'],
            facultyId=course_data.get('facultyId'),
            required_room_type=course_data['required_room_type']
        )
    
    # Load rooms
    for room_data in json_data.get('rooms', []):
        input_data.addRoom(
            Id=room_data['Id'],
            name=room_data['name'],
            capacity=room_data['capacity'],
            room_type=room_data['room_type'],
            building=room_data.get('building', '')
        )
    
    # Load student groups (support both 'studentgroups' and 'student_groups' keys)
    student_groups_data = json_data.get('studentgroups', json_data.get('student_groups', []))
    for sg_data in student_groups_data:
        input_data.addStudentGroup(
            id=sg_data['id'],
            name=sg_data['name'],
            building=sg_data.get('building', ''),
            no_students=sg_data['no_students'],
            courseIDs=sg_data['courseIDs'],
            teacherIDS=sg_data['teacherIDS'],
            hours_required=sg_data['hours_required']
        )
    
    # Load faculties
    for faculty_data in json_data.get('faculties', []):
        input_data.addFaculty(
            id=faculty_data['id'],
            name=faculty_data['name'],
            department=faculty_data.get('department', ''),
            courseID=faculty_data.get('courseID', []),
            avail_days=faculty_data.get('avail_days', []),
            avail_times=faculty_data.get('avail_times', [])
        )
    
    # Create classes for each student group
    for student_group in input_data.student_groups:
        input_data.assign_class_to_course_and_faculty(student_group)
    
    input_data.nostudentgroup = len(input_data.student_groups)
    
    return input_data