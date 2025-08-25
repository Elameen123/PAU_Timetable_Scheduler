
class Faculty:
    def __init__(self, id:str, name:str, department:str, courseID: str, avail_days: list, avail_times: list):
        self.faculty_id = id
        self.name = name
        self.department = department
        self.courseID = courseID
        self.avail_days = avail_days
        self.avail_times = avail_times

    def __repr__(self):
        return f"Faculty(id={self.faculty_id}, name={self.name}, department={self.department}, courseID={self.courseID}, avail_days={self.avail_days}, avail_times={self.avail_times})"
