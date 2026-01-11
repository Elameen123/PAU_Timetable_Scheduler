from typing import List
# from entitities.course import Course
# from entitities.faculty import Faculty
from enums import Size

class StudentGroup:
    def __init__(self, id: str, name:str, no_students: int, courseIDs: List[str], teacherIDS: List[str], hours_required:List[int]):
        self.id = id
        self.name = name
        self.no_students= no_students
        self.courseIDs = courseIDs
        self.no_courses = len(courseIDs)
        self.teacherIDS = teacherIDS    # consider teacherID
        self.hours_required = hours_required

    def __repr__(self):
        return f"StudentGroup(id={self.id}, name={self.name}, courseIDs={self.courseIDs}, teacherIDS={self.teacherIDS}, hours_required={self.hours_required})"

    def categorize_group_size(self):
        if self.no_students <= 20:
            return Size.SMALL
        elif 21 <= self.no_students <= 50:
            return Size.MEDIUM
        else:
            return Size.LARGE

    @property
    def is_sst(self):
        """
        Determines if the student group belongs to the School of Science and Technology (SST).
        Used for building constraints (SST students are preferred in SST building, others in TYD).
        """
        # SST Department Prefixes
        sst_prefixes = {
            'EEE',   # Electrical Engineering
            'MEE',   # Mechanical Engineering
            'CSC',   # Computer Science
            'SEN',   # Software Engineering
            'MCT',   # Mechatronics
            'DTS'    # Data Science
        }
        
        # Keywords that strongly indicate SST
        sst_keywords = [
            'engineering', 'computer science', 'data science', 
            'mechatronics', 'software', 'technology', 'mechanical', 'electrical'
        ]
        
        # 1. Check ID Prefix
        if self.id:
            prefix = self.id.split(' ')[0].upper()
            if prefix in sst_prefixes:
                return True
                
        # 2. Check Name Keywords
        if self.name:
            name_lower = self.name.lower()
            if any(k in name_lower for k in sst_keywords):
                return True
                
        return False