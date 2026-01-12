import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

# For app.py runs, we want to build the exact same InputData instance used by the API pipeline.
from input_data_api import initialize_input_data_from_json


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


_DAYS_MAP = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}


def _time_to_minutes(time_str: str) -> int | None:
    s = str(time_str or "").strip()
    m = re.match(r"^(\d{1,2}):(\d{2})$", s)
    if not m:
        return None
    hh = int(m.group(1))
    mm = int(m.group(2))
    return hh * 60 + mm


def _is_available_day(avail_days: list, day_abbr: str) -> bool:
    if not avail_days:
        return True
    normalized = {str(d).strip().upper() for d in avail_days if d is not None}
    if "ALL" in normalized:
        return True
    return day_abbr.strip().upper() in normalized


def _is_available_time(avail_times: list, start_minutes: int) -> bool:
    if not avail_times:
        return True
    normalized = [str(t).strip() for t in avail_times if t is not None]
    if any(t.upper() == "ALL" for t in normalized):
        return True
    for t in normalized:
        m = re.match(r"^(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})$", t)
        if not m:
            continue
        start = _time_to_minutes(m.group(1))
        end = _time_to_minutes(m.group(2))
        if start is None or end is None:
            continue
        # END-EXCLUSIVE: 9:00-14:00 means hours 9,10,11,12,13 (not 14).
        if start <= start_minutes < end:
            return True
    return False


def _parse_cell(cell: str) -> tuple[str | None, str | None, str | None]:
    """Return (course_code, room_name, lecturer_name) from a timetable cell.

    Supports:
    - Newline format: "CODE\nROOM\nLECTURER"
    - Labeled format: "Course: CODE, Lecturer: X, Room: Y"
    """
    if not cell:
        return None, None, None

    s = str(cell).strip()
    if not s or s in {"BREAK", "FREE"}:
        return None, None, None

    if "\n" in s:
        parts = [p.strip() for p in s.split("\n") if p.strip()]
        if not parts:
            return None, None, None
        course_code = parts[0]
        room_name = parts[1] if len(parts) > 1 else None
        lecturer = parts[2] if len(parts) > 2 else None
        return course_code, room_name, lecturer

    m_course = re.search(r"\bCourse\s*:\s*([^,]+)", s, flags=re.IGNORECASE)
    m_lect = re.search(r"\bLecturer\s*:\s*([^,]+)", s, flags=re.IGNORECASE)
    m_room = re.search(r"\bRoom\s*:\s*(.+)$", s, flags=re.IGNORECASE)
    course_code = m_course.group(1).strip() if m_course else None
    lecturer = m_lect.group(1).strip() if m_lect else None
    room_name = m_room.group(1).strip() if m_room else None
    return course_code, room_name, lecturer


def _build_room_lookup(input_json: dict) -> dict[str, dict]:
    rooms = input_json.get("rooms") or []
    lookup = {}
    for r in rooms:
        if not isinstance(r, dict):
            continue
        name = str(r.get("name") or "").strip()
        rid = str(r.get("Id") or r.get("id") or "").strip()
        if name:
            lookup[name] = r
        if rid:
            lookup[rid] = r
    return lookup


def verify(schedule_path: Path, input_path: Path):
    try:
        if input_path.exists():
            input_json = json.loads(input_path.read_text(encoding="utf-8"))
        else:
            # Fallback: build a transformer-shaped input JSON from the repo's static datasets.
            data_dir = Path("data")
            courses = json.loads((data_dir / "course-data.json").read_text(encoding="utf-8"))
            rooms = json.loads((data_dir / "rooms-data.json").read_text(encoding="utf-8"))
            studentgroups = json.loads((data_dir / "studentgroup-data.json").read_text(encoding="utf-8"))
            faculties = json.loads((data_dir / "faculty-data.json").read_text(encoding="utf-8"))

            input_json = {
                "hours": 9,
                "days": 5,
                "courses": courses,
                "rooms": rooms,
                "studentgroups": studentgroups,
                "faculties": faculties,
            }

        input_data = initialize_input_data_from_json(input_json)
    except Exception as e:
        logger.error(f"Failed to load/initialize input data from {input_path}: {e}")
        return 2

    try:
        timetable_data = json.loads(schedule_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"Failed to load schedule from {schedule_path}: {e}")
        return 2

    rooms_lookup = _build_room_lookup(input_json)

    course_credits = {c.code: getattr(c, "credits", 0) for c in input_data.courses}
    course_lookup = {c.code: c for c in input_data.courses}
    group_by_id = {g.id: g for g in input_data.student_groups}
    faculty_lookup = {}
    for f in getattr(input_data, 'faculties', []) or []:
        fid = str(getattr(f, 'faculty_id', '') or '').strip()
        name = str(getattr(f, 'name', '') or '').strip()
        if fid:
            faculty_lookup[fid.casefold()] = f
        if name:
            faculty_lookup[name.casefold()] = f

    # Map: (day_idx, hour_idx) -> list of {group_id, course_code, room_name}
    global_schedule = defaultdict(list)

    # Map: group_id -> course_code -> list[(day_idx, hour_idx, room_name)]
    group_course_schedule = defaultdict(lambda: defaultdict(list))

    violations = {
        "MissingOrExtra": [],
        "Hard_WrongBuilding": [],
        "Hard_RoomTypeMismatch": [],
        "Hard_ConsecutiveSlots": [],
        "Hard_LecturerClash": [],
        "Hard_StudentGroupClash": [],
        "Hard_LecturerAvailability": [],
        "Hard_SameCourseMultipleRooms": [],
        "Hard_Capacity": [],
    }

    # iterate
    for entry in timetable_data:
        group_info = entry.get("student_group") or {}
        group_id = str(group_info.get("id") or "").strip()
        timetable_matrix = entry.get("timetable") or []

        sg = group_by_id.get(group_id)
        is_sst = bool(getattr(sg, "is_sst", False)) if sg else False

        for h_idx, row in enumerate(timetable_matrix):
            if not isinstance(row, list) or len(row) < 6:
                continue

            start_minutes = _time_to_minutes(row[0])
            # If the time label isn't parseable, fall back to the conventional 9:00 start.
            if start_minutes is None:
                start_minutes = (9 + h_idx) * 60

            for d_idx in range(5):
                cell = row[d_idx + 1]
                course_code, room_name, lecturer = _parse_cell(cell)
                if not course_code:
                    continue

                course_code = course_code.strip()
                room_name = room_name.strip() if room_name else "Unknown"
                lecturer_s = str(lecturer or "").strip()
                lecturer_norm = lecturer_s.casefold() if lecturer_s else None

                global_schedule[(d_idx, h_idx)].append(
                    {"group_id": group_id, "course": course_code, "room": room_name, "lecturer": lecturer_s}
                )
                group_course_schedule[group_id][course_code].append((d_idx, h_idx, room_name))

                # Room type mismatch
                course_obj = course_lookup.get(course_code)
                required_type = str(getattr(course_obj, 'required_room_type', '') or '').strip()
                room_info = rooms_lookup.get(room_name) or {}
                actual_type = str(room_info.get('room_type') or '').strip()
                if required_type and actual_type and required_type != actual_type:
                    violations["Hard_RoomTypeMismatch"].append(
                        f"{group_id} {course_code}: required {required_type}, got {actual_type} in {room_name} at {_DAYS_MAP.get(d_idx,d_idx)} hourIndex={h_idx}"
                    )

                # Lecturer availability (based on the lecturer string in the cell)
                if lecturer_norm:
                    faculty = faculty_lookup.get(lecturer_norm)
                    if faculty is None:
                        # Sometimes the cell contains an email/id instead of a name.
                        faculty = faculty_lookup.get(lecturer_s.casefold())
                    day_abbr = _DAYS_MAP.get(d_idx, "")
                    if faculty is not None:
                        if not _is_available_day(getattr(faculty, 'avail_days', None), day_abbr):
                            violations["Hard_LecturerAvailability"].append(
                                f"{lecturer_s}: not available on {day_abbr} (avail_days={getattr(faculty,'avail_days',None)}, avail_times={getattr(faculty,'avail_times',None)})"
                            )
                        elif not _is_available_time(getattr(faculty, 'avail_times', None), start_minutes):
                            violations["Hard_LecturerAvailability"].append(
                                f"{lecturer_s}: not available at {row[0]} on {day_abbr} (avail_days={getattr(faculty,'avail_days',None)}, avail_times={getattr(faculty,'avail_times',None)})"
                            )

                # Wrong building (TYD in SST)
                if not is_sst:
                    room_info = rooms_lookup.get(room_name)
                    building = str((room_info or {}).get("building") or "").strip().upper()
                    if building == "SST":
                        violations["Hard_WrongBuilding"].append(
                            f"Group {group_id} in {room_name} (SST) at {_DAYS_MAP.get(d_idx,d_idx)} hourIndex={h_idx}"
                        )

    # Lecturer clashes & same-student-group clashes (global checks)
    for (d_idx, h_idx), event_list in global_schedule.items():
        # Lecturer clash
        by_lect = defaultdict(list)
        by_group = defaultdict(list)
        for e in event_list:
            lect = str(e.get('lecturer') or '').strip()
            if lect:
                by_lect[lect.casefold()].append(e)
            gid = str(e.get('group_id') or '').strip()
            if gid:
                by_group[gid].append(e)

        for lect_norm, items in by_lect.items():
            if len(items) > 1:
                lect_name = items[0].get('lecturer')
                details = "; ".join([f"{it['group_id']}:{it['course']}@{it['room']}" for it in items[:4]])
                if len(items) > 4:
                    details += f"; ...(+{len(items)-4})"
                violations["Hard_LecturerClash"].append(
                    f"{lect_name} double-booked at {_DAYS_MAP.get(d_idx,d_idx)} hourIndex={h_idx}: {details}"
                )

        for gid, items in by_group.items():
            if len(items) > 1:
                details = "; ".join([f"{it['course']}@{it['room']}" for it in items[:4]])
                if len(items) > 4:
                    details += f"; ...(+{len(items)-4})"
                violations["Hard_StudentGroupClash"].append(
                    f"{gid} has multiple events at {_DAYS_MAP.get(d_idx,d_idx)} hourIndex={h_idx}: {details}"
                )

    # Completeness check: per group, per course.
    for sg in input_data.student_groups:
        gid = sg.id
        actual_counts = defaultdict(int)
        for course_code, events in group_course_schedule.get(gid, {}).items():
            actual_counts[course_code] += len(events)

        for i, course_code in enumerate(sg.courseIDs or []):
            expected = 0
            credits = course_credits.get(course_code, 0)
            if credits == 1:
                expected = 3
            else:
                # hours_required is already aligned with credits in transformer output
                try:
                    expected = int((sg.hours_required or [])[i])
                except Exception:
                    expected = int(credits or 0)

            actual = actual_counts.get(course_code, 0)
            if actual != expected:
                violations["MissingOrExtra"].append(
                    f"{gid} course {course_code}: expected {expected}, got {actual}"
                )

    # Consecutive-slot checks (same as before)
    for group_id, courses in group_course_schedule.items():
        for course_code, events in courses.items():
            credits = course_credits.get(course_code, 0)
            events = sorted(events)

            if credits == 2 and len(events) == 2:
                d1, h1, _ = events[0]
                d2, h2, _ = events[1]
                if not (d1 == d2 and (h2 - h1) == 1):
                    violations["Hard_ConsecutiveSlots"].append(
                        f"{group_id} {course_code} (2cr): not consecutive ({d1},{h1}) ({d2},{h2})"
                    )

            if credits == 3 and len(events) >= 3:
                has_block = any(events[i][0] == events[i + 1][0] and (events[i + 1][1] - events[i][1]) == 1 for i in range(len(events) - 1))
                if not has_block:
                    violations["Hard_ConsecutiveSlots"].append(
                        f"{group_id} {course_code} (3cr): no 2-hr consecutive block"
                    )

    # Same course multiple rooms on same day
    for group_id, courses in group_course_schedule.items():
        for course_code, events in courses.items():
            rooms_per_day = defaultdict(set)
            for d, _, r in events:
                if r and r != "Unknown":
                    rooms_per_day[d].add(r)
            for d, rooms in rooms_per_day.items():
                if len(rooms) > 1:
                    violations["Hard_SameCourseMultipleRooms"].append(
                        f"{group_id} {course_code} on {_DAYS_MAP.get(d,d)}: rooms={sorted(rooms)}"
                    )

    # Capacity
    for (d_idx, h_idx), event_list in global_schedule.items():
        room_to_groups = defaultdict(set)
        for e in event_list:
            room_to_groups[e["room"]].add(e["group_id"])

        for room_name, groups in room_to_groups.items():
            room_info = rooms_lookup.get(room_name) or {}
            try:
                capacity = int(room_info.get("capacity") or 0)
            except Exception:
                capacity = 0

            for gid in groups:
                sg = group_by_id.get(gid)
                if not sg:
                    continue
                if getattr(sg, "no_students", 0) > capacity and capacity > 0:
                    violations["Hard_Capacity"].append(
                        f"{room_name} cap={capacity} overloaded by {gid} size={getattr(sg,'no_students',0)}"
                    )

    # Report
    print("\n=== APP PIPELINE VERIFICATION REPORT ===")
    total = 0
    for k, vals in violations.items():
        print(f"[{k}]: {len(vals)}")
        total += len(vals)
        for v in vals[:5]:
            print(f"  - {v}")
        if len(vals) > 5:
            print(f"  ... (+{len(vals)-5} more)")

    if total == 0:
        print("\nSUCCESS: No issues detected.")
        return 0

    print(f"\nFAILED: {total} total issues detected.")
    return 1


def main():
    parser = argparse.ArgumentParser(description="Verify fresh_timetable_data.json against the app.py (transformer) input dataset")
    parser.add_argument("--schedule", default="data/fresh_timetable_data.json", help="Path to schedule JSON")
    parser.add_argument("--input", default="data/last_input_data.json", help="Path to last transformer input JSON saved by app.py")
    args = parser.parse_args()

    return verify(Path(args.schedule), Path(args.input))


if __name__ == "__main__":
    raise SystemExit(main())
