from typing import List
from enums import RoomType

class Course:
    def __init__(self, name: str, code: str, credits: int, student_groupsID: List[str], facultyId: str, required_room_type: str):
        self.code = code
        self.name = name
        self.credits = credits
        self.student_groupsID = student_groupsID
        self.facultyId = facultyId
        self.required_room_type = required_room_type

    def __repr__(self):
        return f"Course(name={self.name}, code={self.code}, credits={self.credits}, student_groupsID={self.student_groupsID})"
