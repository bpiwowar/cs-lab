"""``cs-lab course init / update / status``.

An update never merges silently into what the student wrote unless they ask:
files they did not touch are replaced, files the course did not change are
left alone, and for a file both changed the student chooses —

    merge    notebooks only: nbdime, the student's side winning conflicts; their
             previous copy is kept as <name>.mine.ipynb
    new      leave their file, and put the course's next to it: <name>.new.ipynb
    replace  take the course's; theirs is kept as <name>.mine-<date>.ipynb
    keep     do nothing

— per file when asked in a terminal, or once for all with ``--on-conflict``
(or ``on_conflict`` in ``.course/config.toml``). Without a terminal and without
a choice, ``new``: it never touches the student's file.
"""

from __future__ import annotations

import datetime
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import env as course_env
from . import notebooks
from .folder import CHOICES, Config, Folder, FolderError, free_sibling, sibling
from .plan import ADD, CONFLICT, DELETE, KEEP_REMOVED, UPDATE, FileAction, plan
from .release import Release, ReleaseError

Ask = Callable[[str], str]


class CourseError(Exception):
    pass


# ---- choosing what to do with a conflict ------------------------------------


def _mergeable(action: FileAction) -> bool:
    return (
        action.path.endswith(".ipynb")
        and action.has_base
        and not action.deleted_locally
        and notebooks.can_merge()
    )


def _describe(action: FileAction) -> str:
    if action.deleted_locally:
        return "you deleted it, and the course changed it"
    if not action.has_base:
        return "yours differs from the course's (no record of the version you got)"
    return "changed by you and by the course"


class Chooser:
    """Decides, for each conflict, what to do."""

    def __init__(self, policy: str = "", ask: Optional[Ask] = None, show=print):
        self.policy = policy
        self.ask = ask
        self.show = show

    def choose(self, action: FileAction, root: Path, release_dir: Path) -> str:
        allowed = [c for c in CHOICES if c != "merge" or _mergeable(action)]
        if self.policy:
            return self.policy if self.policy in allowed else "new"
        if self.ask is None:
            return "new"
        return self._prompt(action, allowed, root, release_dir)

    def _prompt(self, action, allowed, root: Path, release_dir: Path) -> str:
        keys = {choice[0]: choice for choice in allowed}
        menu = " / ".join(f"[{c[0]}]{c[1:]}" for c in allowed) + " / [d]iff"
        self.show(f"\n{action.path}: {_describe(action)}")
        while True:
            answer = self.ask(f"  {menu} (capital: same for all the rest) ? ").strip()
            if answer.lower() == "d":
                text = notebooks.diff(
                    root / action.path if not action.deleted_locally else None,
                    release_dir / action.path,
                    ("yours", "course"),
                )
                self.show(text or "  (no difference in the cells' text)")
                continue
            if answer[:1].lower() in keys:
                choice = keys[answer[:1].lower()]
                if answer[:1].isupper():
                    self.policy = choice
                return choice


# ---- applying ------------------------------------------------------------


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _apply_conflict(
    action: FileAction, choice: str, folder: Folder, release_dir: Path
) -> str:
    source, target = release_dir / action.path, folder.root / action.path
    rel = folder.root
    if choice == "keep":
        return "kept yours"
    if choice == "new":
        copy = sibling(target, "new")
        _copy(source, copy)
        return f"course's version in {copy.relative_to(rel)}"
    if choice == "replace":
        note = "replaced"
        if target.exists():
            date = datetime.date.today().strftime("%Y%m%d")
            mine = free_sibling(target, f"mine-{date}")
            target.rename(mine)
            note += f", yours kept as {mine.relative_to(rel)}"
        _copy(source, target)
        return note
    return _merge(action, folder, release_dir)


def _merge(action: FileAction, folder: Folder, release_dir: Path) -> str:
    source, target = release_dir / action.path, folder.root / action.path
    base = folder.base / action.path
    rel = folder.root
    with tempfile.TemporaryDirectory() as tmp:
        merged = Path(tmp) / target.name
        try:
            notebooks.merge(base, target, source, merged)
        except notebooks.MergeError as exc:
            copy = sibling(target, "new")
            _copy(source, copy)
            return (
                f"could not merge ({exc}); course's version in {copy.relative_to(rel)}"
            )
        mine = free_sibling(target, "mine")
        shutil.copy2(target, mine)
        shutil.copyfile(merged, target)
    note = f"merged, yours kept as {mine.relative_to(rel)}"
    lost = notebooks.missing_cells(base, source, target)
    if lost:
        copy = sibling(target, "new")
        _copy(source, copy)
        note += (
            f"; {len(lost)} new cell(s) of the course could not be placed, "
            f"see {copy.relative_to(rel)}"
        )
    return note


def _apply(action: FileAction, choice: str, folder: Folder, release_dir: Path) -> str:
    target = folder.root / action.path
    if action.kind in (ADD, UPDATE):
        _copy(release_dir / action.path, target)
        return "updated" if action.kind == UPDATE else "added"
    if action.kind == DELETE:
        target.unlink(missing_ok=True)
        return "removed (the course dropped it)"
    if action.kind == KEEP_REMOVED:
        return "kept: the course dropped it, but you changed it"
    return _apply_conflict(action, choice, folder, release_dir)


# ---- git: a snapshot before and after, when the folder is a repository ----


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True
    )


def _git_snapshot(root: Path, message: str) -> bool:
    if shutil.which("git") is None or not (root / ".git").exists():
        return False
    if not _git(root, "status", "--porcelain").stdout.strip():
        return False
    identity = []
    if not _git(root, "config", "user.email").stdout.strip():
        identity = ["-c", "user.name=cs-lab", "-c", "user.email=cs-lab@localhost"]
    _git(root, "add", "-A")
    return (
        _git(root, *identity, "commit", "--no-verify", "-q", "-m", message).returncode
        == 0
    )


# ---- the commands ---------------------------------------------------------


class Course:
    def __init__(self, show=print, ask: Optional[Ask] = None):
        self.show = show
        self.ask = ask

    def init(
        self,
        url: str,
        directory: Optional[Path] = None,
        shared_env: Optional[Path] = None,
        on_conflict: str = "",
        sync: bool = True,
    ) -> int:
        release = Release.fetch(url)
        folder = Folder(Path(directory) if directory else Path(release.name))
        if folder.exists():
            raise CourseError(
                f"{folder.root} is already a course folder: `cs-lab course update`"
            )
        if shared_env is not None and not Path(shared_env).is_dir():
            raise CourseError(f"no shared environment at {shared_env}")
        config = Config(
            url=release.url,
            name=release.name,
            mode="shared" if shared_env else "own",
            shared_env=str(Path(shared_env).resolve()) if shared_env else "",
            on_conflict=on_conflict,
        )
        folder.root.mkdir(parents=True, exist_ok=True)
        self.show(f"Setting up {folder.root} for {release.name}")
        return self._install(folder, config, release, on_conflict, sync=sync)

    def update(
        self,
        directory: Path = Path("."),
        on_conflict: str = "",
        dry_run: bool = False,
        reset_env: bool = False,
        sync: bool = True,
    ) -> int:
        folder = Folder(directory)
        config = folder.load()
        release = Release.fetch(config.url)
        if release.name != config.name:
            raise CourseError(
                f"{config.url} now serves {release.name!r}, not {config.name!r}"
            )
        if release.id == config.release and not reset_env:
            self.show(f"Up to date (release {release.id}, {release.created}).")
            return 0
        return self._install(
            folder,
            config,
            release,
            on_conflict or config.on_conflict,
            dry_run=dry_run,
            reset_env=reset_env,
            sync=sync,
        )

    def status(self, directory: Path = Path("."), offline: bool = False) -> int:
        folder = Folder(directory)
        config = folder.load()
        self.show(f"{config.name}: release {config.release or '(none)'}")
        self.show(f"  from    {config.url}")
        mode = config.mode + (f" ({config.shared_env})" if config.shared_env else "")
        self.show(f"  env     {mode}")
        changed = [
            path
            for path, digest in folder.base_files().items()
            if folder.local_digest(path) != digest
            and not (
                config.mode == "shared"
                and path in (course_env.LOCK, course_env.PYPROJECT)
            )
        ]
        self.show(
            f"  yours   {len(changed)} file(s) changed since: " + ", ".join(changed)
        )
        if not offline:
            try:
                latest = Release.fetch(config.url)
            except ReleaseError as exc:
                self.show(f"  latest  unknown ({exc})")
            else:
                news = (
                    "up to date"
                    if latest.id == config.release
                    else "an update is available"
                )
                self.show(f"  latest  {latest.id} ({latest.created}): {news}")
        return 0

    # -- the shared part of init and update --

    def _install(
        self,
        folder: Folder,
        config: Config,
        release: Release,
        policy: str,
        dry_run: bool = False,
        reset_env: bool = False,
        sync: bool = True,
    ) -> int:
        with tempfile.TemporaryDirectory() as tmp:
            release_dir = release.download(Path(tmp) / "release")
            actions = self._plan(folder, release)
            chooser = Chooser(policy, self.ask, self.show)
            choices: Dict[str, str] = {
                a.path: chooser.choose(a, folder.root, release_dir)
                for a in actions
                if a.kind == CONFLICT
            }
            if dry_run:
                self._report_plan(actions, choices)
                return 0

            _git_snapshot(folder.root, "cs-lab: my work, before the course update")
            for action in actions:
                note = _apply(action, choices.get(action.path, ""), folder, release_dir)
                self.show(f"  {action.path}: {note}")
            status = self._environment(
                folder, config, release, release_dir, reset_env, sync
            )
            folder.replace_base(release_dir, release.files)
            config.release = release.id
            folder.save(config)
        _git_snapshot(folder.root, f"cs-lab: course release {release.id}")
        if not actions:
            self.show("  (no file to change)")
        self.show(f"Now at release {release.id} ({release.created}).")
        return status

    def _plan(self, folder: Folder, release: Release) -> List[FileAction]:
        content = release.content_files()
        base = {p: h for p, h in folder.base_files().items() if p not in release.env}
        local = {p: folder.local_digest(p) for p in set(base) | set(content)}
        return plan(base, local, {p: release.files[p] for p in content})

    def _report_plan(self, actions: List[FileAction], choices: Dict[str, str]) -> None:
        if not actions:
            self.show("Nothing would change.")
        for action in actions:
            what = choices.get(action.path, action.kind)
            self.show(f"  would {what:8} {action.path}")

    def _environment(
        self,
        folder: Folder,
        config: Config,
        release: Release,
        release_dir: Path,
        reset_env: bool,
        sync: bool,
    ) -> int:
        if config.mode == "shared":
            match = course_env.shared_lock_matches(
                Path(config.shared_env), release.lock
            )
            if match is False:
                self.show(
                    "  warning: the shared environment was built for another "
                    "release; ask your teacher to update it"
                )
            return 0
        result = course_env.update_own(folder, release_dir, reset=reset_env)
        if not result.changed:
            return 0
        self.show("  environment: the course's pyproject.toml / uv.lock")
        if result.replayed.added:
            self.show("    + yours again: " + ", ".join(result.replayed.added))
        if result.replayed.removed:
            self.show("    - removed again: " + ", ".join(result.replayed.removed))
        if result.backup:
            self.show(
                f"    your previous one: {result.backup.relative_to(folder.root)}"
            )
        if result.error:
            self.show(f"  warning: {result.error}")
            return 1
        if sync:
            error = course_env.sync(folder.root)
            if error:
                self.show(f"  warning: uv sync failed:\n{error}")
                return 1
        return 0


def interactive_ask() -> Optional[Ask]:
    """``input``, when there is a terminal to ask in."""
    return input if sys.stdin.isatty() and sys.stdout.isatty() else None


__all__ = ["Course", "CourseError", "Chooser", "FolderError", "ReleaseError"]
