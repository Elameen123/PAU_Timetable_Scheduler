# transformer_api.py
"""
API-ready transformer module for converting the Timetable Excel template
into a JSON structure suitable for the timetable scheduling input system.

Provides:
- validate_excel_structure(file_or_path) -> (bool, message)
- transform_excel_to_json(file_or_path) -> dict

file_or_path may be:
- a filesystem path (str or pathlib.Path)
- a file-like object (has .read()) such as Werkzeug FileStorage.stream
- bytes / bytearray / io.BytesIO

This module does NOT write JSON files to disk; it returns dictionaries.
"""

from pathlib import Path
import pandas as pd
import json
import re
from collections import OrderedDict
from typing import Tuple, Union, Any, Optional, IO
import io

# Opt-in to explicit behavior to avoid silent downcasting warnings
try:
    pd.set_option('future.no_silent_downcasting', True)
except Exception:
    pass

# Types accepted for file_or_path
FileInput = Union[str, Path, bytes, bytearray, IO]

# Required sheet names expected in the template
REQUIRED_SHEETS = ["Classrooms", "Lecturers", "Student Groups", "Courses"]

# Accept common aliases (case-insensitive) for sheet names
SHEET_ALIASES = {
    "Classrooms": ["classrooms", "rooms", "room", "venues", "classroom"],
    "Lecturers": ["lecturers", "faculty", "faculties", "teachers", "instructors"],
    "Student Groups": ["student groups", "studentgroups", "groups", "students", "cohorts"],
    "Courses": ["courses", "course list", "modules", "subjects"],
}

def _normalize(s: str) -> str:
    return str(s or '').strip().lower()

def _resolve_required_sheets(sheet_names):
    """Return a mapping of logical required sheet name -> actual sheet name in workbook.
    Uses case-insensitive matching and common aliases.
    """
    normalized_map = {_normalize(name): name for name in sheet_names}
    resolved = {}
    for logical in REQUIRED_SHEETS:
        # direct match
        if _normalize(logical) in normalized_map:
            resolved[logical] = normalized_map[_normalize(logical)]
            continue
        # alias match
        for alias in SHEET_ALIASES.get(logical, []):
            if alias in normalized_map:
                resolved[logical] = normalized_map[alias]
                break
        # fuzzy: allow replacing hyphens/extra spaces
        if logical not in resolved:
            for cand_norm, real in normalized_map.items():
                if cand_norm.replace('-', ' ').replace('_', ' ') == _normalize(logical).replace('-', ' ').replace('_', ' '):
                    resolved[logical] = real
                    break
    return resolved

def _open_excel(file_or_path: FileInput) -> pd.ExcelFile:
    """
    Return a pandas.ExcelFile from a path, bytes, or file-like object.
    Raises helpful errors if reading fails.
    """
    try:
        if isinstance(file_or_path, (bytes, bytearray)):
            bio = io.BytesIO(file_or_path)
            return pd.ExcelFile(bio)
        if hasattr(file_or_path, "read") and not isinstance(file_or_path, (str, Path)):
            # file-like object (e.g., request.files['file'].stream or FileStorage)
            content = file_or_path.read()
            # If stream returns bytes, wrap; if returns str, attempt encoding
            if isinstance(content, str):
                content = content.encode("utf-8")
            return pd.ExcelFile(io.BytesIO(content))
        # else assume path-like
        path = Path(file_or_path)
        if not path.exists():
            raise FileNotFoundError(f"Excel file not found at path: {path}")
        return pd.ExcelFile(str(path))
    except Exception as e:
        raise RuntimeError(f"Failed to open Excel file: {e}") from e


def validate_excel_structure(file_or_path: FileInput) -> Tuple[bool, str]:
    """
    Validate the workbook contains required sheets and basic expected columns.
    Returns (True, "") when valid, otherwise (False, "reason").
    """
    try:
        xls = _open_excel(file_or_path)
    except Exception as e:
        return False, f"Could not open Excel: {e}"

    # Resolve required sheets with aliases
    sheet_map = _resolve_required_sheets(xls.sheet_names)
    missing = [s for s in REQUIRED_SHEETS if s not in sheet_map]
    if missing:
        return False, (
            "Workbook is missing required sheet(s): "
            + ", ".join(missing)
            + ". Found sheets: "
            + ", ".join(map(str, xls.sheet_names))
        )

    # Basic checks for Courses sheet: must have at least Course Code and a lecturer column
    try:
        courses_df = pd.read_excel(xls, sheet_name=sheet_map["Courses"], dtype=object)
    except Exception as e:
        return False, f"Failed to read 'Courses' sheet: {e}"

    # check Course Code header
    cols_lower = [str(c).strip().lower() for c in courses_df.columns]
    if not any(c in cols_lower for c in ("course code", "code", "course_code")):
        return False, "Courses sheet missing a 'Course Code' column."

    # Try to find an assigned lecturer column (heuristic)
    assigned_col = None
    for key in ("assigned lecturer emails", "assigned lecturers", "assigned lecturer", "lecturer email", "lecturer", "lecturer emails", "assigned lecturer email"):
        for col in courses_df.columns:
            if key in str(col).strip().lower():
                assigned_col = col
                break
        if assigned_col:
            break
    # fallback: any column containing 'lectur' substring
    if assigned_col is None:
        for col in courses_df.columns:
            if "lectur" in str(col).lower():
                assigned_col = col
                break

    if assigned_col is None:
        return False, "Courses sheet does not contain a detectable 'Assigned Lecturer' column."

    return True, ""


# ----------------------- helper functions -----------------------
def slugify_id(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r'[^A-Za-z0-9\-_]', '_', str(s)).strip('_')


def normalize_list_cell(raw: Any):
    """
    Convert a cell that may contain lists (strings separated by commas/;/ /) into a list of strings.
    """
    if raw is None:
        return []
    if pd.isna(raw):
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]
    s = str(raw).strip()
    if not s:
        return []
    # split on common separators
    parts = re.split(r'[;,/]|[ \t]+', s)
    return [p.strip() for p in parts if p.strip()]


def find_student_group_columns(columns):
    """
    Detect columns in Courses sheet that correspond to Student Group 1, Student Group 2, ...
    Fallback: any column containing 'group' (first few).
    """
    pattern = re.compile(r'(?i)^\s*student\s*group\s*(\d+)\s*$')
    matches = []
    for c in columns:
        m = pattern.match(str(c))
        if m:
            matches.append((int(m.group(1)), c))
    if matches:
        matches.sort(key=lambda x: x[0])
        return [col for _, col in matches]
    # fallback
    candidate = [c for c in columns if 'group' in str(c).lower()]
    return candidate[:3]


def find_assigned_lecturer_column(columns):
    """
    Return the best guess column name in columns for assigned lecturers.
    """
    low = {str(c).strip().lower(): c for c in columns}
    for key in ("assigned lecturer emails", "assigned lecturers", "assigned lecturer", "lecturer email", "lecturer emails", "email"):
        if key in low:
            return low[key]
    # look for column containing 'lectur' and optionally 'email'
    for c in columns:
        s = str(c).lower()
        if 'lectur' in s and 'email' in s:
            return c
    for c in columns:
        if 'lectur' in str(c).lower():
            return c
    return None


# ----------------------- main transform function -----------------------
def transform_excel_to_json(file_or_path: FileInput) -> dict:
    """
    Parse the timetable Excel template and return a dict containing parsed data:
    {
      "courses": [...],
      "rooms": [...],
      "studentgroups": [...],
      "faculties": [...]
    }

    Accepts file path, bytes, or file-like (see _open_excel).
    Raises RuntimeError on parsing problems (missing sheets/columns).
    """
    xls = _open_excel(file_or_path)

    # Resolve names using aliases
    sheet_map = _resolve_required_sheets(xls.sheet_names)
    for required in REQUIRED_SHEETS:
        if required not in sheet_map:
            raise RuntimeError(
                f"Required sheet '{required}' not found in workbook. Found: {xls.sheet_names}"
            )

    # Read required sheets into dataframes (as object dtype)
    sheets = {}
    for logical in REQUIRED_SHEETS:
        real_name = sheet_map[logical]
        df = pd.read_excel(xls, sheet_name=real_name, dtype=object).fillna("")
        # Infer object dtypes explicitly to avoid future warning (no silent downcasting)
        try:
            df = df.infer_objects(copy=False)
        except Exception:
            pass
        sheets[logical] = df

    # Build faculty map from Lecturers sheet
    lect_df = sheets["Lecturers"]
    email_col = None
    name_col = None
    for col in lect_df.columns:
        lc = str(col).strip().lower()
        if lc in ("faculty email", "lecturer email", "email", "email address"):
            email_col = col
        if lc in ("faculty name", "lecturer name", "name"):
            name_col = col
    # fallback heuristics
    if email_col is None:
        for col in lect_df.columns:
            if 'email' in str(col).lower():
                email_col = col
                break
    if name_col is None:
        for col in lect_df.columns:
            if 'name' in str(col).lower():
                name_col = col
                break

    faculty_by_lower = {}
    for _, r in lect_df.iterrows():
        raw_email = str(r.get(email_col) or "").strip() if email_col else ""
        raw_name = str(r.get(name_col) or "").strip() if name_col else ""
        dept = str(r.get("Department") or "").strip() if "Department" in lect_df.columns else ""
        status = str(r.get("Status") or "").strip() if "Status" in lect_df.columns else ""
        avail_days = []
        if "Available Days" in lect_df.columns:
            aval = str(r.get("Available Days") or "").strip()
            avail_days = [d.strip() for d in re.split(r'[ ,;]+', aval) if d.strip()] if aval else []
        avail_times = []
        if "Available Times" in lect_df.columns:
            aval_t = str(r.get("Available Times") or "").strip()
            avail_times = [t.strip() for t in re.split(r'[ ,;]+', aval_t) if t.strip()] if aval_t else []
        if raw_email:
            key = raw_email.lower()
            faculty_by_lower[key] = {"id": raw_email, "name": raw_name or raw_email, "department": dept, "status": status, "avail_days": avail_days, "avail_times": avail_times, "courseID": []}
        else:
            synthetic = slugify_id(raw_name) or f"lect_{len(faculty_by_lower)+1}"
            faculty_by_lower[synthetic.lower()] = {"id": synthetic, "name": raw_name or synthetic, "department": dept, "status": status, "avail_days": avail_days, "avail_times": avail_times, "courseID": []}

    # Rooms
    rooms = []
    rooms_df = sheets["Classrooms"]
    for idx, r in rooms_df.iterrows():
        name = str(r.get("Room Name") or "").strip()
        building = str(r.get("Building") or "").strip() if "Building" in rooms_df.columns else ""
        cap_raw = r.get("Capacity")
        try:
            capacity = int(cap_raw) if str(cap_raw).strip() else 0
        except Exception:
            capacity = 0
        room_type = str(r.get("Classroom Type") or "").strip() if "Classroom Type" in rooms_df.columns else "Classroom"
        notes = str(r.get("Location Notes") or "").strip() if "Location Notes" in rooms_df.columns else ""
        Id = slugify_id(name) or f"room_{idx+1}"
        rooms.append({"Id": Id, "name": name or Id, "capacity": capacity, "room_type": room_type, "building": building, "notes": notes})

    # Student groups
    groups = OrderedDict()
    groups_df = sheets["Student Groups"]
    for _, r in groups_df.iterrows():
        gid = str(r.get("Group ID") or "").strip()
        gname = str(r.get("Group Name") or "").strip()
        level = str(r.get("Level") or "").strip() if "Level" in groups_df.columns else ""
        dept = str(r.get("Department") or "").strip() if "Department" in groups_df.columns else ""
        size_raw = r.get("Size") if "Size" in groups_df.columns else ""
        try:
            size = int(size_raw) if str(size_raw).strip() else 0
        except Exception:
            size = 0
        if not gid:
            gid = slugify_id(gname) or f"group_{len(groups)+1}"
        groups[gid] = {"id": gid, "name": gname or gid, "level": level, "dept": dept, "no_students": size, "courseIDs": [], "teacherIDS": [], "hours_required": []}

    # Courses
    course_df = sheets["Courses"]
    sg_cols = find_student_group_columns(list(course_df.columns))
    assigned_col = find_assigned_lecturer_column(list(course_df.columns))
    if assigned_col is None:
        raise RuntimeError("Could not detect the 'Assigned Lecturer' column in the Courses sheet.")

    courses = []
    for _, r in course_df.iterrows():
        code = str(r.get("Course Code") or "").strip()
        if not code:
            continue
        name = str(r.get("Course Name") or "").strip()
        credits_raw = r.get("Credit Units") if "Credit Units" in course_df.columns else r.get("Credits")
        try:
            credits = int(credits_raw) if str(credits_raw).strip() else 0
        except Exception:
            credits = 0
        room_type = str(r.get("Classroom Type") or "").strip() if "Classroom Type" in course_df.columns else "Classroom"
        lecturers = normalize_list_cell(r.get(assigned_col))

        # Build student_groupsID strictly from sg_cols
        student_groups = []
        for col in sg_cols:
            val = str(r.get(col) or "").strip()
            if val:
                student_groups.append(val)
        seen = set()
        student_groups = [x for x in student_groups if not (x in seen or seen.add(x))]

        dept = str(r.get("Department") or "").strip() if "Department" in course_df.columns else ""
        req_raw = str(r.get("Special Requirements") or "").strip() if "Special Requirements" in course_df.columns else ""
        req_list = [p.strip() for p in re.split(r'[;,/]|[ \t]+', req_raw) if p.strip()] if req_raw else []

        facultyId = lecturers[0] if lecturers else None
        courses.append({"name": name, "code": code, "credits": credits, "student_groupsID": student_groups, "facultyId": facultyId, "required_room_type": room_type, "lecturers": lecturers, "dept": dept, "req": req_list})

        for g in student_groups:
            if g not in groups:
                groups[g] = {"id": g, "name": g, "level": "", "dept": dept, "no_students": 0, "courseIDs": [], "teacherIDS": [], "hours_required": []}
            groups[g]["courseIDs"].append(code)
            # Keep duplicates aligned with courseIDs (as in original)
            groups[g]["teacherIDS"].append(lecturers[0] if lecturers else None)
            groups[g]["hours_required"].append(credits)

        # Append course code to each lecturer's courseID list
        for lect in lecturers:
            key = lect.strip().lower()
            if key in faculty_by_lower:
                if code not in faculty_by_lower[key]["courseID"]:
                    faculty_by_lower[key]["courseID"].append(code)
                if not faculty_by_lower[key].get("department") and dept:
                    faculty_by_lower[key]["department"] = dept
            else:
                faculty_by_lower[key] = {"id": lect.strip(), "name": lect.strip(), "department": dept, "status": "", "avail_days": [], "avail_times": [], "courseID": [code]}

    # finalize groups (coerce hours_required to ints)
    for gobj in groups.values():
        gobj["hours_required"] = [int(h) if str(h).strip() else 0 for h in (gobj.get("hours_required") or [])]

    # Build JSON-compatible structures
    course_json = [
        {
            "name": c["name"],
            "code": c["code"],
            "credits": int(c["credits"]),
            "student_groupsID": c["student_groupsID"],
            "facultyId": c["facultyId"],
            "required_room_type": c["required_room_type"],
            "lecturers": c.get("lecturers", []),
            "dept": c.get("dept", ""),
            "req": c.get("req", [])
        } for c in courses
    ]
    rooms_json = [
        {"Id": r["Id"], "name": r["name"], "capacity": int(r["capacity"]), "room_type": r["room_type"], "building": r["building"], "notes": r.get("notes", "")}
        for r in rooms
    ]
    studentgroups_json = [
        {
            "id": g["id"],
            "name": g["name"],
            "no_students": int(g.get("no_students") or 0),
            "courseIDs": g.get("courseIDs") or [],
            "teacherIDS": g.get("teacherIDS") or [],
            "hours_required": g.get("hours_required") or []
        } for g in groups.values()
    ]
    faculties_json = list(faculty_by_lower.values())

    result = {
        "courses": course_json,
        "rooms": rooms_json,
        "studentgroups": studentgroups_json,
        # Keep alternative key for compatibility
        "student_groups": studentgroups_json,
        "faculties": faculties_json,
        # include simple counts for convenience
        "_meta": {
            "course_count": len(course_json),
            "room_count": len(rooms_json),
            "studentgroup_count": len(studentgroups_json),
            "faculty_count": len(faculties_json),
        }
    }

    return result


# If module executed directly, allow quick local test (not used by app)
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python transformer_api.py <path-to-Excel>")
        sys.exit(1)
    path = sys.argv[1]
    ok, msg = validate_excel_structure(path)
    if not ok:
        print("Validation failed:", msg)
        sys.exit(2)
    data = transform_excel_to_json(path)
    print(json.dumps(data.get("_meta", {}), indent=2))
    # Optionally print sample course count
    print(f"Courses parsed: {len(data['courses'])}")
