"""What an update does to each file: a three-way comparison.

For every file, three versions: *base* (the release the student was given),
*local* (what is in their folder now) and *new* (the release being installed),
each a hash or ``None`` when the file is absent.

    new == base      the course did not change it: leave the student's copy
    local == new     already up to date
    local == base    the student did not touch it: take the new one
                     (add it, or delete it when the course removed it)
    otherwise        both changed it: a conflict, for the student to decide

Two conflicts need no decision: a file the course removed but the student
changed is kept, and so is anything outside the release.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

#: Kinds of action, in the order they are reported.
ADD = "add"
UPDATE = "update"
DELETE = "delete"
KEEP_REMOVED = "kept (removed from the course)"
CONFLICT = "conflict"


@dataclass
class FileAction:
    path: str
    kind: str
    base: Optional[str]
    local: Optional[str]
    new: Optional[str]

    @property
    def has_base(self) -> bool:
        return self.base is not None

    @property
    def deleted_locally(self) -> bool:
        return self.local is None


def classify(
    path: str, base: Optional[str], local: Optional[str], new: Optional[str]
) -> Optional[FileAction]:
    """The action for one file, or None when there is nothing to do."""
    if new == base or local == new:
        return None
    if local == base:
        if new is None:
            return FileAction(path, DELETE, base, local, new)
        return FileAction(path, ADD if local is None else UPDATE, base, local, new)
    if new is None:
        return FileAction(path, KEEP_REMOVED, base, local, new)
    return FileAction(path, CONFLICT, base, local, new)


def plan(
    base: Dict[str, str],
    local: Dict[str, Optional[str]],
    new: Dict[str, str],
) -> List[FileAction]:
    """Actions for every path of the base and the new release.

    ``local`` maps the same paths to the hash of the student's copy (None
    when absent).
    """
    actions = []
    for path in sorted(set(base) | set(new)):
        action = classify(path, base.get(path), local.get(path), new.get(path))
        if action is not None:
            actions.append(action)
    return actions
