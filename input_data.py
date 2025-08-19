from typing import List
from entitities.course import Course
from entitities.faculty import Faculty
from entitities.room import Room
from entitities.student_group import StudentGroup
from entitities.Class import Class
import json
from enums import Size, RoomType
from entitities.time_slot import TimeSlot


class inputData():
    def __init__(self) -> None:
        self.courses = []
        self.rooms = []
        self.student_groups = []    
        self.faculties = []
        self.constraints = []
        self.classes = []
        self.nostudentgroup = len(self.student_groups)
        self.hours = 8
        self.days = 5

    def addCourse(self, name: str, code: str, credits: int, student_groupsID: List[str], facultyId, required_room_type: str ):
        self.courses.append(Course(name, code, credits, student_groupsID, facultyId, required_room_type))
        # print(Course(name, code, credits, student_groupsID).name)

    def addRoom(self, Id: str, name:str, capacity:int, room_type:str):
        self.rooms.append(Room(Id, name, capacity, room_type))

    def addStudentGroup(self, id: str, name:str, no_students: int, courseIDs: str, teacherIDS: str, hours_required:List[int]):
        self.student_groups.append(StudentGroup(id, name, no_students, courseIDs, teacherIDS, hours_required))

    def addFaculty(self, id:str, name:str, department:str, courseID: str):
        self.faculties.append(Faculty(id, name, department, courseID))

    # def addConstraint(self, constraint: Constraint):
    #     self.constraints.append(constraint)

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
        for faculty in self.faculties:
            if faculty.faculty_id == id:
                return faculty
        return None
    
    def create_time_slots(self, no_hours_per_day, no_days_per_week, day_start_time):
        time_slots = []
        for day in range(no_days_per_week):
            for hour in range(no_hours_per_day):
                time_slots.append(TimeSlot(id=len(time_slots), day=day, start_time=hour, available=True))
        # print(time_slots, len(time_slots))
        return time_slots
    
    
    def assign_class_to_course_and_faculty(self, student_group: StudentGroup):
        for course_x in student_group.courseIDs:
            for course in input_data.courses:
                if course_x == course.code:
                    facultyId = course.facultyId
                    self.classes.append(Class(student_group.id, facultyId, course.code))

    # def __repr__(self):
    #     return f"inputData(courses={self.courses}, rooms={self.rooms}, student_groups={self.student_groups}, faculties={self.faculties}, constraints={self.constraints}, classes={self.classes})"
        
    

    # def getConstraints(self, constraint_type: str) -> List[Constraint]:
    #     constraints = []
    #     for constraint in self.constraints:
    #         if constraint.constraint_type == constraint_type:
    #             constraints.append(constraint)
    #     return constraints
    
    # def getConstraintsByCourse(self, course_code: str) -> List[Constraint]:
    #     constraints = []
    #     for constraint in self.constraints:
    #         if constraint.course_code == course_code:
    #             constraints.append(constraint)
    #     return constraints
    

input_data = inputData()

# Read course data from JSON file
with open('data/course-data.json') as file:
    course_data = json.load(file)
    for course in course_data:
        input_data.addCourse(course['name'], course['code'], course['credits'], course['student_groupsID'], course['facultyId'], course['required_room_type'])

# Read room data from JSON file
with open('data/rooms-data.json') as file:
    room_data = json.load(file)
    for room in room_data:
        input_data.addRoom(room['Id'], room['name'], room['capacity'], room['room_type'])

# Read student group data from JSON file
with open('data/studentgroup-data.json') as file:
    student_group_data = json.load(file)
    for student_group in student_group_data:
        input_data.addStudentGroup(student_group['id'], student_group['name'], student_group['no_students'], student_group['courseIDs'], student_group['teacherIDS'], student_group['hours_required'])

# Read faculty data from JSON file
with open('data/faculty-data.json') as file:
    faculty_data = json.load(file)
    for faculty in faculty_data:
        input_data.addFaculty(faculty['id'], faculty['name'], faculty['department'], faculty['courseID'])

# timeslot
# [print(time_slot.day, time_slot.start_time) for time_slot in input_data.create_time_slots(7, 5, 9)]

for student_group in input_data.student_groups:
    input_data.assign_class_to_course_and_faculty(student_group)

input_data.nostudentgroup = len(input_data.student_groups)

# print(repr(input_data))
# print("\n")

# for course in input_data.courses:
#     print(Course.__repr__(course))

# print("\n")

# for room in input_data.rooms:
#     print(Room.__repr__(room))

# print("\n")

# for student_group in input_data.student_groups:
#     print(StudentGroup.__repr__(student_group))

# print("\n")

# for faculty in input_data.faculties:
#     print(Faculty.__repr__(faculty))

# print("\n")

# for class_obj in input_data.classes:
#     print(Class.__repr__(class_obj))

# print("\n")

# for time_slot in input_data.create_time_slots(7, 5, 9):
#     print(TimeSlot.__repr__(time_slot))

        