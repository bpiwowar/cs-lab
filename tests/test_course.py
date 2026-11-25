"""Tests for `cs-lab course`: releases, the three-way plan, conflicts, env."""

import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from cs_lab.course import (
    Chooser,
    Course,
    CourseError,
    Folder,
    Release,
    ReleaseError,
    notebooks,
)
from cs_lab.course import env as course_env
from cs_lab.course.plan import ADD, CONFLICT, DELETE, UPDATE, classify


def nb(*cells):
    """A minimal notebook; each cell is ("code"|"markdown", source)."""
    return json.dumps(
        {
            "cells": [
                {
                    "cell_type": kind,
                    "id": f"c{i}",
                    "metadata": {},
                    "source": source,
                    **(
                        {"outputs": [], "execution_count": None}
                        if kind == "code"
                        else {}
                    ),
                }
                for i, (kind, source) in enumerate(cells)
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        },
        indent=1,
    )


def pyproject(*deps):
    listed = ",\n".join(f'    "{d}"' for d in deps)
    return (
        f'[project]\nname = "tp"\nversion = "0.1.0"\nrequires-python = ">=3.11"\n'
        f"dependencies = [\n{listed},\n]\n\n[tool.uv]\npackage = false\n"
    )


V1 = {
    "README.md": "# TP\n",
    "pyproject.toml": pyproject("six"),
    "uv.lock": "version = 1\n# v1\n",
    "notebooks/a.ipynb": nb(
        ("markdown", "# Attention\nThis practicle shows attention."),
        ("code", "x = 1"),
        ("code", "# Exercise\n# hint: y = x * ?\nassert False"),
        ("code", "print(x)"),
    ),
    "notebooks/b.ipynb": nb(("markdown", "# B"), ("code", "b = 1")),
    "notebooks/old.ipynb": nb(("code", "old = 1")),
}


def v2():
    files = dict(V1)
    files["notebooks/a.ipynb"] = nb(
        ("markdown", "# Attention\nThis practical shows attention."),
        ("code", "x = 2"),
        ("code", "# Exercise\n# hint: y = x * 3\nassert False"),
        ("code", "print(x)"),
    )
    files["notebooks/b.ipynb"] = nb(("markdown", "# B"), ("code", "b = 10"))
    files["notebooks/c.ipynb"] = nb(("code", "c = 1"))
    files["pyproject.toml"] = pyproject("six", "attrs")
    files["uv.lock"] = "version = 1\n# v2\n"
    del files["notebooks/old.ipynb"]
    return files


def publish(
    site: Path,
    files: dict,
    name: str = "tp-test",
    shared_env: str = "",
    shared_cache: str = "",
    rooms: list = (),
) -> str:
    """Write a release (zip + release.json) into ``site``; return its URL."""
    site.mkdir(parents=True, exist_ok=True)
    bundle = site / "tp.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        for path, content in sorted(files.items()):
            archive.writestr(path, content)
    hashes = {
        path: "sha256:" + hashlib.sha256(content.encode()).hexdigest()
        for path, content in sorted(files.items())
    }
    listing = "".join(f"{p}\0{h}\n" for p, h in hashes.items())
    release = {
        "format": 1,
        "name": name,
        "id": hashlib.sha256(listing.encode()).hexdigest()[:16],
        "created": "2026-11-22T00:00:00Z",
        "bundle": {
            "path": "tp.zip",
            "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
        },
        "env": [p for p in ("pyproject.toml", "uv.lock") if p in files],
        "files": hashes,
    }
    if "uv.lock" in hashes:
        release["lock"] = hashes["uv.lock"]
    if shared_env:
        release["shared_env"] = shared_env
    if shared_cache:
        release["shared_cache"] = shared_cache
    if rooms:
        release["rooms"] = list(rooms)
    (site / "release.json").write_text(json.dumps(release))
    return site.as_uri()


def cells(path: Path):
    return ["".join(c["source"]) for c in json.loads(path.read_text())["cells"]]


def student_edits(folder: Path):
    """Fill the exercise of a.ipynb, add a note, and run it (outputs)."""
    path = folder / "notebooks" / "a.ipynb"
    data = json.loads(path.read_text())
    data["cells"][2]["source"] = "# Exercise\n# hint: y = x * ?\ny = x * 3"
    for cell in data["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = [
                {"output_type": "stream", "name": "stdout", "text": "out\n"}
            ]
    data["cells"].insert(
        3, {"cell_type": "markdown", "id": "mine", "metadata": {}, "source": "my note"}
    )
    path.write_text(json.dumps(data, indent=1))


@pytest.fixture
def course():
    lines = []
    return Course(show=lines.append), lines


@pytest.fixture
def site(tmp_path):
    return tmp_path / "site"


def setup_folder(tmp_path, site, course_obj, **kwargs):
    url = publish(site, V1)
    folder = tmp_path / "work"
    assert course_obj.init(url, folder, sync=False, **kwargs) == 0
    return folder


# ---- the plan --------------------------------------------------------------


@pytest.mark.parametrize(
    "base,local,new,kind",
    [
        ("b", "b", "b", None),  # nothing changed
        ("b", "l", "b", None),  # only the student changed it
        ("b", "b", "n", UPDATE),  # only the course changed it
        ("b", "n", "n", None),  # both made the same change
        (None, None, "n", ADD),  # a new file
        ("b", "b", None, DELETE),  # the course dropped an untouched file
        ("b", "l", "n", CONFLICT),
        ("b", None, "n", CONFLICT),  # the student deleted it, the course changed it
        (None, "l", "n", CONFLICT),  # no base: a folder not set up by cs-lab
    ],
)
def test_classify(base, local, new, kind):
    action = classify("f", base, local, new)
    assert (action.kind if action else None) == kind


# ---- init / update ----------------------------------------------------------


def test_init_installs_the_release_and_records_it(tmp_path, site, course):
    course_obj, _ = course
    folder = setup_folder(tmp_path, site, course_obj)
    assert (folder / "notebooks" / "a.ipynb").read_text() == V1["notebooks/a.ipynb"]
    assert (folder / "pyproject.toml").is_file()
    state = Folder(folder)
    config = state.load()
    assert config.name == "tp-test" and config.mode == "own" and config.release
    assert sorted(state.base_files()) == sorted(V1)
    assert (folder / ".course" / ".gitignore").read_text() == "base/\n"


def test_update_takes_what_the_student_did_not_touch(tmp_path, site, course):
    course_obj, lines = course
    folder = setup_folder(tmp_path, site, course_obj)
    (folder / "README.md").write_text("my readme\n")  # course did not change it
    publish(site, v2())

    assert course_obj.update(folder, on_conflict="keep", sync=False) == 0
    assert (folder / "notebooks" / "b.ipynb").read_text() == v2()["notebooks/b.ipynb"]
    assert (folder / "notebooks" / "c.ipynb").is_file()  # new practical
    assert not (folder / "notebooks" / "old.ipynb").exists()  # dropped, untouched
    assert (folder / "README.md").read_text() == "my readme\n"  # theirs
    assert Folder(folder).load().release == Release.fetch(site.as_uri()).id

    lines.clear()
    assert course_obj.update(folder, sync=False) == 0
    assert any("Up to date" in line for line in lines)


def test_a_dropped_file_the_student_changed_is_kept(tmp_path, site, course):
    course_obj, _ = course
    folder = setup_folder(tmp_path, site, course_obj)
    (folder / "notebooks" / "old.ipynb").write_text(nb(("code", "mine")))
    publish(site, v2())
    course_obj.update(folder, sync=False)
    assert "mine" in cells(folder / "notebooks" / "old.ipynb")


def test_dry_run_changes_nothing(tmp_path, site, course):
    course_obj, lines = course
    folder = setup_folder(tmp_path, site, course_obj)
    before = Folder(folder).load().release
    publish(site, v2())
    course_obj.update(folder, dry_run=True, sync=False)
    assert not (folder / "notebooks" / "c.ipynb").exists()
    assert Folder(folder).load().release == before
    assert any("notebooks/c.ipynb" in line for line in lines)


# ---- conflicts --------------------------------------------------------------


def test_new_leaves_theirs_and_writes_the_course_version_next_to_it(
    tmp_path, site, course
):
    course_obj, _ = course
    folder = setup_folder(tmp_path, site, course_obj)
    student_edits(folder)
    mine = (folder / "notebooks" / "a.ipynb").read_text()
    publish(site, v2())

    course_obj.update(folder, on_conflict="new", sync=False)
    assert (folder / "notebooks" / "a.ipynb").read_text() == mine
    assert (folder / "notebooks" / "a.new.ipynb").read_text() == v2()[
        "notebooks/a.ipynb"
    ]


def test_replace_keeps_theirs_aside(tmp_path, site, course):
    course_obj, _ = course
    folder = setup_folder(tmp_path, site, course_obj)
    student_edits(folder)
    mine = (folder / "notebooks" / "a.ipynb").read_text()
    publish(site, v2())

    course_obj.update(folder, on_conflict="replace", sync=False)
    assert (folder / "notebooks" / "a.ipynb").read_text() == v2()["notebooks/a.ipynb"]
    (aside,) = (folder / "notebooks").glob("a.mine-*.ipynb")
    assert aside.read_text() == mine


def test_keep_changes_nothing_but_moves_the_base(tmp_path, site, course):
    course_obj, _ = course
    folder = setup_folder(tmp_path, site, course_obj)
    student_edits(folder)
    mine = (folder / "notebooks" / "a.ipynb").read_text()
    publish(site, v2())
    course_obj.update(folder, on_conflict="keep", sync=False)
    assert (folder / "notebooks" / "a.ipynb").read_text() == mine
    base = Folder(folder).base / "notebooks" / "a.ipynb"
    assert base.read_text() == v2()["notebooks/a.ipynb"]


needs_nbdime = pytest.mark.skipif(not notebooks.can_merge(), reason="no nbdime, no uvx")


@needs_nbdime
def test_merge_brings_the_fixes_into_the_students_notebook(tmp_path, site, course):
    course_obj, _ = course
    folder = setup_folder(tmp_path, site, course_obj)
    student_edits(folder)
    publish(site, v2())

    course_obj.update(folder, on_conflict="merge", sync=False)
    merged = cells(folder / "notebooks" / "a.ipynb")
    assert merged[0] == "# Attention\nThis practical shows attention."  # typo fixed
    assert merged[1] == "x = 2"
    assert "y = x * 3" in merged[2]  # their answer stays
    assert "my note" in merged
    assert (folder / "notebooks" / "a.mine.ipynb").is_file()
    assert not (folder / "notebooks" / "a.new.ipynb").exists()


@needs_nbdime
def test_a_course_cell_the_merge_loses_is_kept_in_a_new_copy(tmp_path, site, course):
    course_obj, lines = course
    folder = setup_folder(tmp_path, site, course_obj)
    path = folder / "notebooks" / "b.ipynb"
    path.write_text(
        nb(("markdown", "# B"), ("code", "z = 'student'"), ("code", "b = 1"))
    )
    files = v2()
    files["notebooks/b.ipynb"] = nb(
        ("markdown", "# B"), ("code", "z = 'teacher'"), ("code", "b = 10")
    )
    publish(site, files)

    course_obj.update(folder, on_conflict="merge", sync=False)
    if "z = 'teacher'" not in cells(path):
        assert (folder / "notebooks" / "b.new.ipynb").is_file()
        assert any("could not be placed" in line for line in lines)


def test_merge_is_only_offered_for_notebooks_with_a_base(tmp_path):
    action = classify("README.md", "b", "l", "n")
    assert Chooser("merge").choose(action, tmp_path, tmp_path) == "new"


def test_a_capital_answer_applies_to_the_rest(tmp_path, site):
    asked = []

    def ask(prompt):
        asked.append(prompt)
        return "K"

    shown = []
    course_obj = Course(show=shown.append, ask=ask)
    folder = setup_folder(tmp_path, site, course_obj)
    student_edits(folder)
    (folder / "notebooks" / "b.ipynb").write_text(nb(("code", "mine")))
    publish(site, v2())

    course_obj.update(folder, sync=False)
    assert len(asked) == 1  # asked for the first conflict only
    assert "mine" in cells(folder / "notebooks" / "b.ipynb")


def test_without_a_terminal_the_default_is_new(tmp_path, site, course):
    course_obj, _ = course  # no ask
    folder = setup_folder(tmp_path, site, course_obj)
    student_edits(folder)
    publish(site, v2())
    course_obj.update(folder, sync=False)
    assert (folder / "notebooks" / "a.new.ipynb").is_file()


def test_the_folder_default_is_used(tmp_path, site, course):
    course_obj, _ = course
    folder = setup_folder(tmp_path, site, course_obj, on_conflict="replace")
    student_edits(folder)
    publish(site, v2())
    course_obj.update(folder, sync=False)
    assert list((folder / "notebooks").glob("a.mine-*.ipynb"))


# ---- the environment ---------------------------------------------------------


def test_dependency_changes(tmp_path):
    base, local = tmp_path / "base.toml", tmp_path / "local.toml"
    base.write_text(pyproject("six", "numpy"))
    local.write_text(pyproject("six", "requests>=2", "torch"))
    changes = course_env.student_changes(base, local)
    assert changes.added == ["requests>=2", "torch"]
    assert changes.removed == ["numpy"]


def test_the_students_packages_are_added_again(tmp_path, site, course, monkeypatch):
    calls = []

    def fake_uv(args, cwd):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(course_env, "_uv", fake_uv)
    monkeypatch.setattr(course_env.shutil, "which", lambda name: "/usr/bin/" + name)
    course_obj, _ = course
    folder = setup_folder(tmp_path, site, course_obj)
    (folder / "pyproject.toml").write_text(pyproject("six", "requests"))
    (folder / "uv.lock").write_text("version = 1\n# mine\n")
    publish(site, v2())

    course_obj.update(folder, sync=False)
    assert (folder / "uv.lock").read_text() == v2()["uv.lock"]  # the course's lock
    assert course_env.dependencies(folder / "pyproject.toml") == ["six", "attrs"]
    assert calls == [["add", "--no-sync", "requests"]]
    assert course_env.dependencies(folder / "pyproject.mine.toml") == [
        "six",
        "requests",
    ]


def test_untouched_environment_is_simply_replaced(tmp_path, site, course, monkeypatch):
    monkeypatch.setattr(course_env, "_uv", lambda *a: pytest.fail("uv called"))
    course_obj, _ = course
    folder = setup_folder(tmp_path, site, course_obj)
    publish(site, v2())
    course_obj.update(folder, sync=False)
    assert (folder / "pyproject.toml").read_text() == v2()["pyproject.toml"]
    assert not (folder / "pyproject.mine.toml").exists()


def shared_env(path: Path) -> Path:
    """A stand-in for a teaching room's environment, built from V1's lock."""
    (path / "bin").mkdir(parents=True)
    (path / "bin" / "python").write_text("")
    (path / "uv.lock").write_text(V1["uv.lock"])
    return path


def test_shared_mode_leaves_the_environment_out(tmp_path, site, course):
    course_obj, lines = course
    shared = shared_env(tmp_path / "shared")
    folder = setup_folder(tmp_path, site, course_obj, shared_env=shared)
    assert not (folder / "pyproject.toml").exists()
    assert Folder(folder).load().mode == "shared"

    publish(site, v2())  # a new lock the shared env was not built from
    course_obj.update(folder, sync=False)
    assert not (folder / "uv.lock").exists()
    assert any("shared environment was built for another" in line for line in lines)


def test_init_uses_the_room_environment_when_it_is_there(tmp_path, site, course):
    course_obj, lines = course
    shared = shared_env(tmp_path / "room")
    url = publish(site, V1, shared_env=str(shared))
    folder = tmp_path / "work"
    assert course_obj.init(url, folder, sync=False) == 0
    config = Folder(folder).load()
    assert (config.mode, config.shared_env) == ("shared", str(shared))
    assert not (folder / "uv.lock").exists()
    assert any("shared environment of this room" in line for line in lines)


@pytest.mark.parametrize("why", ["absent", "own-env"])
def test_init_makes_its_own_environment_otherwise(tmp_path, site, course, why):
    course_obj, _ = course
    shared = tmp_path / "room"
    if why == "own-env":
        shared_env(shared)
    url = publish(site, V1, shared_env=str(shared))
    folder = tmp_path / "work"
    assert course_obj.init(url, folder, sync=False, own_env=why == "own-env") == 0
    assert Folder(folder).load().mode == "own"
    assert (folder / "uv.lock").exists()


def test_update_follows_a_moved_room_environment(tmp_path, site, course):
    course_obj, lines = course
    url = publish(site, V1, shared_env=str(shared_env(tmp_path / "room")))
    folder = tmp_path / "work"
    course_obj.init(url, folder, sync=False)
    moved = shared_env(tmp_path / "room2")
    publish(site, V1, shared_env=str(moved))
    course_obj.update(folder, sync=False)
    assert Folder(folder).load().shared_env == str(moved)


@pytest.mark.parametrize("mode", ["shared", "own"])
def test_lab_runs_jupyter_from_the_folder_environment(tmp_path, site, mode):
    calls = []
    course_obj = Course(
        show=lambda _: None, run=lambda cmd, cwd: calls.append((cmd, cwd)) or 0
    )
    shared = shared_env(tmp_path / "room")
    url = publish(site, V1, shared_env=str(shared))
    folder = tmp_path / "work"
    course_obj.init(url, folder, sync=False, own_env=mode == "own")
    assert course_obj.lab(folder) == 0
    ((command, cwd),) = calls
    assert cwd == folder
    if mode == "shared":
        assert command == [
            str(shared / "bin" / "python"),
            "-m",
            "jupyter",
            "lab",
            "notebooks",
        ]
    else:
        assert command == ["uv", "run", "jupyter", "lab", "notebooks"]


def test_lab_says_when_the_room_environment_is_missing(tmp_path, site, course):
    course_obj, _ = course
    shared = shared_env(tmp_path / "room")
    folder = setup_folder(tmp_path, site, course_obj, shared_env=shared)
    shutil.rmtree(shared)
    with pytest.raises(CourseError, match="not on this machine"):
        course_obj.lab(folder)


# ---- the room's cache, with an environment of one's own --------------------


def test_own_env_records_the_room_cache_when_it_is_here(tmp_path, site, course):
    course_obj, lines = course
    cache = tmp_path / "cache"
    cache.mkdir()
    url = publish(site, V1, shared_cache=str(cache))
    folder = tmp_path / "work"
    assert course_obj.init(url, folder, sync=False) == 0
    config = Folder(folder).load()
    assert (config.mode, config.shared_cache) == ("own", str(cache))
    assert any("room's cache" in line for line in lines)


@pytest.mark.parametrize("why", ["absent", "refused", "shared-env"])
def test_no_room_cache_recorded(tmp_path, site, course, why):
    course_obj, _ = course
    cache = tmp_path / "cache"
    if why != "absent":
        cache.mkdir()
    shared = str(shared_env(tmp_path / "room")) if why == "shared-env" else ""
    url = publish(site, V1, shared_env=shared, shared_cache=str(cache))
    folder = tmp_path / "work"
    course_obj.init(url, folder, sync=False, shared_cache=why != "refused")
    assert Folder(folder).load().shared_cache == ""


def test_update_follows_a_moved_room_cache(tmp_path, site, course):
    course_obj, _ = course
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache2").mkdir()
    url = publish(site, V1, shared_cache=str(tmp_path / "cache"))
    folder = tmp_path / "work"
    course_obj.init(url, folder, sync=False)
    publish(site, V1, shared_cache=str(tmp_path / "cache2"))
    course_obj.update(folder, sync=False)
    assert Folder(folder).load().shared_cache == str(tmp_path / "cache2")


def test_lab_points_an_own_environment_at_the_room_cache(tmp_path, site, monkeypatch):
    calls = []
    course_obj = Course(
        show=lambda _: None,
        run=lambda cmd, cwd, env=None: calls.append((cmd, env)) or 0,
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    url = publish(site, V1, shared_cache=str(cache))
    folder = tmp_path / "work"
    course_obj.init(url, folder, sync=False)
    (folder / "resources.lock").write_text("{}")
    monkeypatch.delenv("CS_LAB_CACHE_PATH", raising=False)
    monkeypatch.delenv("CS_LAB_CACHE_LOCK", raising=False)
    assert course_obj.lab(folder) == 0
    ((command, env),) = calls
    assert command[:3] == ["uv", "run", "jupyter"]
    assert env["CS_LAB_CACHE_PATH"] == str(cache)
    assert env["IR_DATASETS_HOME"] == f"{cache}/ir_datasets"
    assert env["CS_LAB_CACHE_LOCK"] == str((folder / "resources.lock").resolve())


# ---- several rooms ----------------------------------------------------------


def _room(name, env=None, cache=None):
    room = {"name": name}
    if env:
        room["shared_env"] = str(env)
    if cache:
        room["shared_cache"] = str(cache)
    return room


def test_init_takes_the_first_room_that_is_here(tmp_path, site, course):
    course_obj, lines = course
    here = shared_env(tmp_path / "lab")
    rooms = [_room("ppti", tmp_path / "ppti"), _room("lab", here)]
    url = publish(site, V1, rooms=rooms)
    folder = tmp_path / "work"
    assert course_obj.init(url, folder, sync=False) == 0
    config = Folder(folder).load()
    assert (config.mode, config.room, config.shared_env) == ("shared", "lab", str(here))
    assert any("the lab room" in line for line in lines)


def test_own_env_takes_the_first_room_cache_that_is_here(tmp_path, site, course):
    course_obj, _ = course
    (tmp_path / "lab-cache").mkdir()
    rooms = [
        _room("ppti", tmp_path / "ppti", tmp_path / "ppti-cache"),
        _room("lab", cache=tmp_path / "lab-cache"),
    ]
    url = publish(site, V1, rooms=rooms)
    folder = tmp_path / "work"
    course_obj.init(url, folder, sync=False)
    config = Folder(folder).load()
    assert (config.mode, config.room) == ("own", "lab")
    assert config.shared_cache == str(tmp_path / "lab-cache")


def test_update_stays_in_the_folder_room(tmp_path, site, course):
    course_obj, _ = course
    lab = shared_env(tmp_path / "lab")
    url = publish(site, V1, rooms=[_room("ppti", tmp_path / "ppti"), _room("lab", lab)])
    folder = tmp_path / "work"
    course_obj.init(url, folder, sync=False)
    # the first room is now here too, and the lab's environment moved
    shared_env(tmp_path / "ppti")
    moved = shared_env(tmp_path / "lab2")
    rooms = [_room("ppti", tmp_path / "ppti"), _room("lab", moved)]
    publish(site, V1, rooms=rooms)
    course_obj.update(folder, sync=False)
    config = Folder(folder).load()
    assert (config.room, config.shared_env) == ("lab", str(moved))


def test_rooms_from_a_release_before_them():
    from cs_lab.course.release import _rooms

    assert _rooms({}) == []
    (room,) = _rooms({"shared_env": "/e", "shared_cache": "/c"})
    assert (room.name, room.shared_env, room.shared_cache) == ("", "/e", "/c")


# ---- safety ----------------------------------------------------------------


def test_a_tampered_bundle_is_refused(tmp_path, site):
    url = publish(site, V1)
    release = Release.fetch(url)
    with zipfile.ZipFile(site / "tp.zip", "a") as archive:
        archive.writestr("extra.txt", "x")
    with pytest.raises(ReleaseError):
        release.download(tmp_path / "out")


def test_another_course_at_the_same_url_is_refused(tmp_path, site, course):
    course_obj, _ = course
    folder = setup_folder(tmp_path, site, course_obj)
    publish(site, v2(), name="another")
    with pytest.raises(Exception, match="another"):
        course_obj.update(folder, sync=False)


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not available")
def test_a_git_folder_gets_a_commit_before_and_after(tmp_path, site, course):
    course_obj, _ = course
    folder = setup_folder(tmp_path, site, course_obj)
    git = ["git", "-C", str(folder)]
    subprocess.run([*git, "init", "-q"], check=True)
    student_edits(folder)
    publish(site, v2())
    course_obj.update(folder, on_conflict="new", sync=False)
    log = subprocess.run(
        [*git, "log", "--format=%s"], capture_output=True, text=True
    ).stdout
    assert log.splitlines() == [
        "cs-lab: course release " + Folder(folder).load().release,
        "cs-lab: my work, before the course update",
    ]


@pytest.mark.parametrize("form", ["dir", "json"])
def test_release_url_local_path(tmp_path, form):
    from cs_lab.course.release import release_url

    folder = tmp_path / "a@b"
    url = release_url(str(folder if form == "dir" else folder / "release.json"))
    assert url == (folder / "release.json").as_uri()


def test_a_release_format_newer_than_this_cs_lab_is_refused(tmp_path, site, course):
    course_obj, _ = course
    url = publish(site, V1)
    release = json.loads((site / "release.json").read_text())
    release["format"] = 99
    (site / "release.json").write_text(json.dumps(release))
    with pytest.raises(ReleaseError, match="cs-lab@latest"):
        course_obj.init(url, tmp_path / "work", sync=False)
