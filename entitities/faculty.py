
class Faculty:
    def __init__(self, id:str, name:str, department:str, courseID: str):
        self.faculty_id = id
        self.name = name
        self.department = department
        self.courseID = courseID

    def __repr__(self):
        return f"Faculty(id={self.faculty_id}, name={self.name}, department={self.department}, courseID={self.courseID})"
       