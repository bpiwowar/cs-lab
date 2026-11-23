"""Hand a course's practicals to a student, and keep their folder up to date.

``cs-lab course init <url>`` sets a folder up from the release a course
publishes (``release.json`` next to its zip); ``cs-lab course update`` brings
it to the latest release without losing the student's work. See
:mod:`cs_lab.course.commands` for what an update does.
"""

from .commands import Chooser, Course, CourseError
from .folder import CHOICES, Config, Folder, FolderError
from .release import Release, ReleaseError

__all__ = [
    "CHOICES",
    "Chooser",
    "Config",
    "Course",
    "CourseError",
    "Folder",
    "FolderError",
    "Release",
    "ReleaseError",
]
