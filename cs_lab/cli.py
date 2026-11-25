"""Command line: inspect the cache, list, download and audit declared resources.

Resources are given with ``--from module:attribute`` (or ``--from path.py``,
which reads ``RESOURCES`` from a file that need not be importable): the module
is imported and the attribute read from it. It must be a resource mapping
(``{section: [Resource, ...]}``) or a zero-argument callable returning one::

    cs-lab cache info
    cs-lab cache list --from llm_course.resources:RESOURCES
    cs-lab cache list --from src/llm_course/resources.py
    cs-lab cache download --from llm_course.resources:get_resources --section practical2
    cs-lab cache download --from llm_course.resources:RESOURCES --key gpt2

``scan`` and ``check`` work the other way round, on the *sources* that load the
resources: they read the loader calls out of the code, so that a declaration
can be written from them (``scan --emit``) and kept honest (``check``)::

    cs-lab cache scan  sources/
    cs-lab cache scan  sources/ --emit 02-generation
    cs-lab cache check sources/ --declaration mycourse/resources.py

Revisions (see :mod:`cs_lab.revisions`): ``lock`` pins each resource at a
commit hash, ``download --lock`` fetches those, ``--register`` records the
lock in the cache; ``usage`` then tells which lock uses what, and ``gc``
removes what none uses. The cache is ``--cache DIR``, or CS_LAB_CACHE_PATH::

    cs-lab cache lock --from src/mycourse/resources.py -o resources.lock
    cs-lab cache download --from src/mycourse/resources.py \\
        --lock resources.lock --register mycourse
    cs-lab cache usage --cache /shared/cache
    cs-lab cache gc --cache /shared/cache          # what would go; --yes to do it
    cs-lab cache migrate --cache /shared/cache --adopt
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from . import config
from .resources import Resources, download_resources, format_resources
from .scan import DEFAULT_GUARDS, compare, emit_section, parse_declaration, scan_paths


def _import_path(path: Path) -> object:
    """Import a ``.py`` file directly, without it being on ``sys.path``."""
    if not path.is_file():
        raise SystemExit(f"no such file: {path}")
    name = f"_cs_lab_resources_{abs(hash(str(path.resolve())))}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # a bad declaration file, not our problem to fix
        del sys.modules[name]
        raise SystemExit(f"{path}: {exc}") from exc
    return module


def _split_spec(spec: str) -> Tuple[str, str]:
    """``module:attr``, ``file.py`` or ``file.py:attr`` -> (target, attribute)."""
    head, sep, tail = spec.rpartition(":")
    if sep and head.endswith(".py"):
        return head, tail  # file.py:attr
    if spec.endswith(".py"):
        return spec, "RESOURCES"  # file.py, attribute implied
    module_name, sep, attr_path = spec.partition(":")
    if not sep or not attr_path:
        raise SystemExit(
            f"invalid resource spec {spec!r}: expected module:attribute, "
            "path/to/resources.py, or path/to/resources.py:attribute"
        )
    return module_name, attr_path


def load_resources(spec: str) -> Resources:
    """Resolve ``module:attribute`` (or a ``.py`` path) to a resource mapping."""
    target, attr_path = _split_spec(spec)
    if target.endswith(".py"):
        obj: object = _import_path(Path(target))
    else:
        obj = importlib.import_module(target)
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    if callable(obj):
        obj = obj()
    if not isinstance(obj, dict):
        raise SystemExit(f"{spec} did not yield a {{section: [resources]}} mapping")
    return obj


def _load_all(specs: Sequence[str]) -> List[Tuple[str, Resources]]:
    return [(spec, load_resources(spec)) for spec in specs]


def _use_cache(args: argparse.Namespace) -> None:
    """``--cache DIR`` stands for CS_LAB_CACHE_PATH, for this process."""
    if getattr(args, "cache", None):
        os.environ[config.ENV_PATH] = str(args.cache)


def _root(args: argparse.Namespace) -> Path:
    root = config.get_cache_path()
    if root is None:
        raise SystemExit(f"no cache: give --cache DIR, or set {config.ENV_PATH}")
    if not root.is_dir():
        raise SystemExit(f"no such directory: {root}")
    return root


def cmd_info(args: argparse.Namespace) -> int:
    print(config.describe())
    return 0


def cmd_env(args: argparse.Namespace) -> int:
    from .envinfo import ProbeError, for_folder

    try:
        described, note = for_folder(args.directory, args.python)
    except ProbeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if note:
        print(f"({note})")
    print(described)
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    for spec, resources in _load_all(args.resources):
        print(f"{spec}:")
        if not resources:
            print("  (no resources)")
            continue
        print("\n".join(format_resources(resources, indent="  ")))
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    root = config.get_cache_path()
    if root is None:
        logging.warning(
            "%s is not set: resources will go to the default HuggingFace cache "
            "(~/.cache/huggingface). Set it to populate a shared cache.",
            config.ENV_PATH,
        )
    else:
        print(f"Cache location: {root}")

    if args.lock:
        config.set_lock(args.lock)
        print(f"Revisions from {args.lock}")
    elif args.register:
        raise SystemExit("--register needs --lock")

    failed: List[Tuple[str, object, Exception]] = []
    with config.hub_online():
        for spec, resources in _load_all(args.resources):
            if args.section and args.section not in resources:
                logging.warning("%s: no section %r", spec, args.section)
                continue
            try:
                download_resources(
                    resources,
                    section=args.section,
                    key=args.key,
                    include_optional=args.optional,
                    keep_going=args.keep_going,
                    failures=failed,
                )
            except Exception:
                if not args.keep_going:
                    raise
                failed.append((spec, None, sys.exc_info()[1]))
                logging.exception("%s: download failed", spec)

    if args.register and root is not None:
        from .revisions import register

        print(f"Lock registered as {register(root, args.register, args.lock)}")

    if failed:
        print(f"\n{len(failed)} resource(s) could not be downloaded:")
        for section_name, resource, exc in failed:
            name = getattr(resource, "key", section_name)
            print(f"  - {section_name}/{name}: {exc}")
        return 1
    return 0


def cmd_lock(args: argparse.Namespace) -> int:
    from .revisions import lock

    output = Path(args.output)
    previous = config.read_lock(output) if output.exists() else {}
    resources: Resources = {}
    for _, group in _load_all(args.resources):
        for section, items in group.items():
            resources.setdefault(section, []).extend(items)
    with config.hub_online():
        locked = lock(resources, previous, args.upgrade)
    config.write_lock(output, locked)
    changed = [
        key
        for key in locked
        if previous.get(key, {}).get("revision") != locked[key]["revision"]
    ]
    gone = sorted(set(previous) - set(locked))
    print(f"{output}: {len(locked)} entries, {len(changed)} new or changed", end="")
    print(f", {len(gone)} removed" if gone else "")
    for key in changed:
        print(f"  {key} @ {locked[key]['revision'][:12]}")
    return 0


def _short(revision: str) -> str:
    return revision[:12] if config.is_sha(revision) else revision


def _size(path: Path) -> str:
    total = 0
    for dirpath, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(dirpath) / name).stat().st_size
            except OSError:
                pass
    for unit in ("B", "K", "M", "G"):
        if total < 1024:
            return f"{total:.0f}{unit}"
        total /= 1024
    return f"{total:.1f}T"


def cmd_usage(args: argparse.Namespace) -> int:
    from .revisions import registered, usage

    root = _root(args)
    locks = registered(root)
    print(f"Cache {root}")
    print(f"Registered locks ({root / config.LOCKS_DIR}):")
    for name in locks or ["(none)"]:
        print(f"  {name}")
    uses, missing = usage(root)
    print("Revisions:")
    for use in uses:
        if args.unused and use.used:
            continue
        holders = use.locks + [f"ref:{ref}" for ref in use.refs]
        size = f"  {_size(use.path)}" if args.size else ""
        print(
            f"  {use.entry.key} @ {_short(use.revision)}{size}  "
            f"{', '.join(holders) if holders else 'UNUSED'}"
        )
    if missing:
        print("Locked but not in the cache:")
        for name, key, sha in missing:
            print(f"  {key} @ {sha[:12]}  ({name})")
    return 0


def cmd_gc(args: argparse.Namespace) -> int:
    from .revisions import garbage, registered, remove

    root = _root(args)
    if not registered(root) and not args.force:
        raise SystemExit(
            f"no lock registered in {root / config.LOCKS_DIR}: every locked "
            "revision would count as unused (--force to go on anyway)"
        )
    unused = garbage(root, legacy=args.legacy)
    if not unused:
        print("Nothing to remove")
        return 0
    for use in unused:
        size = f"  {_size(use.path)}" if args.size else ""
        verb = "removing" if args.yes else "would remove"
        print(f"{verb} {use.entry.key} @ {_short(use.revision)}{size}")
        if args.yes:
            remove(use)
    if not args.yes:
        print("(dry run: --yes to remove)")
    return 0


def cmd_migrate(args: argparse.Namespace) -> int:
    from .revisions import adopt, migrate, registered

    root = _root(args)
    for key in migrate(root):
        print(f"{key}: moved to {config.LEGACY}/")
    if args.adopt or args.lock:
        locked: dict = {}
        for path in args.lock or []:
            locked.update(config.read_lock(path))
        if not args.lock:
            for entries in registered(root).values():
                locked.update(entries)
        with config.hub_online():
            for key, outcome in adopt(root, locked):
                print(f"{key}: {outcome}")
    return 0


def cmd_register(args: argparse.Namespace) -> int:
    from .revisions import register

    print(register(_root(args), args.name, args.lock))
    return 0


def cmd_unregister(args: argparse.Namespace) -> int:
    path = _root(args) / config.LOCKS_DIR / f"{args.name}.lock"
    if not path.exists():
        raise SystemExit(f"no lock {args.name!r} ({path})")
    path.unlink()
    print(f"removed {path}")
    return 0


def _scan(args: argparse.Namespace):
    search_paths = list(args.search_path or [])
    for path in args.sources:
        root = path if path.is_dir() else path.parent
        for candidate in (root, root.parent / "src", root.parent):
            if candidate.is_dir() and candidate not in search_paths:
                search_paths.append(candidate)
    return scan_paths(args.sources, guards=args.guard, search_paths=search_paths)


def cmd_scan(args: argparse.Namespace) -> int:
    scanned = _scan(args)
    if args.emit is not None:
        sections = [args.emit] if args.emit else sorted(scanned)
        unknown = [s for s in sections if s not in scanned]
        if unknown:
            raise SystemExit(f"no such section: {', '.join(unknown)}")
        print("RESOURCES = {")
        for section in sections:
            print(emit_section(section, scanned[section]))
        print("}")
        return 0

    for section in sorted(scanned):
        print(f"{section}:")
        for resource in sorted(scanned[section], key=lambda r: (r.kind, r.key)):
            note = ""
            if resource.bypass:
                module = "datasets" if resource.kind == "dataset" else "transformers"
                note = f"  (bypasses cs_lab: import it from cs_lab.{module})"
            print(f"  - {resource}{note}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    scanned = _scan(args)
    try:
        declared = parse_declaration(args.declaration)
    except (OSError, SyntaxError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    errors, warnings = compare(scanned, declared)
    for warning in warnings:
        print(f"warning: {warning}")
    for error in errors:
        print(f"error: {error}")

    if errors:
        print(
            f"\n{args.declaration} is out of sync ({len(errors)} difference(s)); "
            "`cs-lab cache scan --emit <section>` prints a skeleton to start from."
        )
        return 1
    total = sum(1 for items in scanned.values() for r in items if not r.bypass)
    print(f"{args.declaration} matches the sources ({total} cached loads).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cs-lab",
        description=(
            "What a teaching notebook needs from the machine it runs on. The "
            "compute-profile ladder and hardware detection are library-only "
            "(`from cs_lab import Profile, hardware`); what has a command "
            "line is the shared resource cache, below."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    groups = parser.add_subparsers(dest="group", required=True)

    p_env = groups.add_parser(
        "env",
        help="what this Python is (the room's shared environment, yours…), "
        "the course folder, and where models and datasets come from",
    )
    p_env.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=None,
        help="where to look for the course folder (default: here)",
    )
    p_env.add_argument(
        "--python",
        type=Path,
        help="describe this interpreter instead of the course folder's",
    )
    p_env.set_defaults(func=cmd_env)

    # Everything here is about the resource cache, and says so: the package
    # covers more than that now (a profile ladder, hardware detection), and a
    # later group must not have to compete with `list` or `check` for a name.
    cache = groups.add_parser(
        "cache",
        help="the shared resource cache: inspect it, declare and download what "
        "a course needs",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = cache.add_subparsers(dest="command", required=True)

    def add_cache(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--cache",
            type=Path,
            metavar="DIR",
            help=f"the cache (default: ${config.ENV_PATH})",
        )

    p_info = sub.add_parser("info", help="show the cache configuration")
    add_cache(p_info)
    p_info.set_defaults(func=cmd_info)

    def add_from(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--from",
            dest="resources",
            metavar="MODULE:ATTR|FILE.py",
            action="append",
            required=True,
            help="resource mapping or callable returning one; a .py path is "
            "imported directly, its RESOURCES read unless :ATTR says otherwise "
            "(repeatable)",
        )

    p_list = sub.add_parser("list", help="list declared resources")
    add_from(p_list)
    p_list.set_defaults(func=cmd_list)

    p_dl = sub.add_parser("download", help="download declared resources")
    add_from(p_dl)
    p_dl.add_argument("--section", help="only this section (lecture, practical...)")
    p_dl.add_argument("--key", help="only the resource with this key")
    p_dl.add_argument(
        "--optional", action="store_true", help="include optional resources"
    )
    p_dl.add_argument(
        "--keep-going",
        action="store_true",
        help="continue with the next --from group when a download fails",
    )
    p_dl.add_argument(
        "--lock", type=Path, metavar="FILE", help="download the revisions it pins"
    )
    p_dl.add_argument(
        "--register",
        metavar="NAME",
        help="record the lock in the cache as NAME (for usage / gc)",
    )
    add_cache(p_dl)
    p_dl.set_defaults(func=cmd_download)

    p_lock = sub.add_parser(
        "lock", help="pin each declared resource at a commit hash (needs the Hub)"
    )
    add_from(p_lock)
    p_lock.add_argument(
        "-o",
        "--output",
        default="resources.lock",
        metavar="FILE",
        help="the lock, updated in place (default: %(default)s)",
    )
    p_lock.add_argument(
        "--upgrade",
        nargs="*",
        metavar="KEY",
        help="re-resolve these entries (e.g. models/gpt2), or all when none given",
    )
    p_lock.set_defaults(func=cmd_lock)

    p_usage = sub.add_parser(
        "usage", help="each revision in the cache, and the locks using it"
    )
    add_cache(p_usage)
    p_usage.add_argument("--unused", action="store_true", help="only unused ones")
    p_usage.add_argument("--size", action="store_true", help="with sizes (slow)")
    p_usage.set_defaults(func=cmd_usage)

    p_gc = sub.add_parser(
        "gc", help="remove the revisions no registered lock (or ref) uses"
    )
    add_cache(p_gc)
    p_gc.add_argument("--yes", action="store_true", help="remove (default: list)")
    p_gc.add_argument(
        "--legacy",
        action="store_true",
        help="also content from before revisions were kept, when unused",
    )
    p_gc.add_argument("--size", action="store_true", help="with sizes (slow)")
    p_gc.add_argument(
        "--force", action="store_true", help="even with no lock registered"
    )
    p_gc.set_defaults(func=cmd_gc)

    p_mig = sub.add_parser(
        "migrate", help="move a cache from before revisions to the current layout"
    )
    add_cache(p_mig)
    p_mig.add_argument(
        "--adopt",
        action="store_true",
        help="rename legacy content to the hash the registered locks pin, when "
        "the Hub's main was at it on download (needs the Hub)",
    )
    p_mig.add_argument(
        "--lock",
        type=Path,
        action="append",
        metavar="FILE",
        help="adopt with this lock instead of the registered ones (repeatable)",
    )
    p_mig.set_defaults(func=cmd_migrate)

    p_reg = sub.add_parser("register", help="record a lock in the cache")
    add_cache(p_reg)
    p_reg.add_argument("name")
    p_reg.add_argument("lock", type=Path)
    p_reg.set_defaults(func=cmd_register)

    p_unreg = sub.add_parser(
        "unregister", help="forget a lock: what only it used becomes garbage"
    )
    add_cache(p_unreg)
    p_unreg.add_argument("name")
    p_unreg.set_defaults(func=cmd_unregister)

    def add_scan(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "sources",
            nargs="+",
            type=Path,
            metavar="PATH",
            help="source files, or directories whose *.py are scanned",
        )
        p.add_argument(
            "--guard",
            action="append",
            default=list(DEFAULT_GUARDS),
            metavar="NAME",
            help="`if NAME:` marks an optional stand-in (default: %(default)s)",
        )
        p.add_argument(
            "--search-path",
            action="append",
            type=Path,
            metavar="DIR",
            help="root under which imported modules are found and scanned too",
        )

    p_scan = sub.add_parser("scan", help="list the resources the sources load")
    add_scan(p_scan)
    p_scan.add_argument(
        "--emit",
        nargs="?",
        const="",
        metavar="SECTION",
        help="print a declaration skeleton (all sections, or just SECTION)",
    )
    p_scan.set_defaults(func=cmd_scan)

    p_check = sub.add_parser(
        "check", help="check a declaration against the sources (exit 1 on drift)"
    )
    add_scan(p_check)
    p_check.add_argument(
        "--declaration",
        required=True,
        type=Path,
        metavar="FILE",
        help="Python file holding the RESOURCES mapping",
    )
    p_check.set_defaults(func=cmd_check)

    add_course_parser(groups)
    return parser


def _course(args: argparse.Namespace):
    from .course.commands import Course, interactive_ask

    return Course(ask=None if getattr(args, "no_input", False) else interactive_ask())


def cmd_course_init(args: argparse.Namespace) -> int:
    return _course(args).init(
        args.url,
        args.directory,
        shared_env=args.shared_env,
        on_conflict=args.on_conflict or "",
        sync=not args.no_sync,
        own_env=args.own_env,
    )


def cmd_course_update(args: argparse.Namespace) -> int:
    return _course(args).update(
        args.directory,
        on_conflict=args.on_conflict or "",
        dry_run=args.dry_run,
        reset_env=args.reset_env,
        sync=not args.no_sync,
    )


def cmd_course_lab(args: argparse.Namespace) -> int:
    return _course(args).lab(args.directory, args.lab_args)


def cmd_course_status(args: argparse.Namespace) -> int:
    return _course(args).status(args.directory, offline=args.offline)


def add_course_parser(groups) -> None:
    from .course.folder import CHOICES

    course = groups.add_parser(
        "course",
        help="a course's practicals in a folder of yours, kept up to date",
        description=(
            "Set a folder up from a course's release (the release.json next to "
            "its zip), and update it to later releases without losing your "
            "work: files you did not touch are replaced, and for a file both "
            "you and the course changed, you choose."
        ),
    )
    sub = course.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser, conflict_help: str) -> None:
        p.add_argument(
            "--on-conflict",
            choices=CHOICES,
            help=conflict_help,
        )
        p.add_argument(
            "--no-sync", action="store_true", help="do not run `uv sync` afterwards"
        )
        p.add_argument(
            "--no-input",
            action="store_true",
            help="never ask: use --on-conflict, the folder's default, or `new`",
        )

    p_init = sub.add_parser("init", help="set a folder up from a release")
    p_init.add_argument(
        "url", help="the course's release (release.json, or its directory)"
    )
    p_init.add_argument(
        "directory", nargs="?", type=Path, help="default: the course's name"
    )
    p_init.add_argument(
        "--shared-env",
        type=Path,
        metavar="PATH",
        help="use this shared environment instead of one of your own (default: "
        "the teaching room's, when the release names one and it is here)",
    )
    p_init.add_argument(
        "--own-env",
        action="store_true",
        help="an environment of your own, even where a shared one is available",
    )
    add_common(
        p_init,
        "the default for later updates, when a file changed on both sides "
        "(stored in .course/config.toml)",
    )
    p_init.set_defaults(func=cmd_course_init)

    p_update = sub.add_parser("update", help="update a folder to the latest release")
    p_update.add_argument("directory", nargs="?", type=Path, default=Path("."))
    p_update.add_argument(
        "--dry-run", action="store_true", help="say what would change, change nothing"
    )
    p_update.add_argument(
        "--reset-env",
        action="store_true",
        help="take the course's environment as is, dropping your own packages",
    )
    add_common(
        p_update,
        "what to do with every file both you and the course changed: merge "
        "(notebooks), new (theirs next to yours), replace (yours kept aside), keep",
    )
    p_update.set_defaults(func=cmd_course_update)

    p_lab = sub.add_parser(
        "lab", help="JupyterLab, from the folder's environment (shared or its own)"
    )
    p_lab.add_argument("directory", nargs="?", type=Path, default=Path("."))
    p_lab.add_argument(
        "lab_args",
        nargs=argparse.REMAINDER,
        help="passed to `jupyter lab` (default: the notebooks/ directory)",
    )
    p_lab.set_defaults(func=cmd_course_lab)

    p_status = sub.add_parser("status", help="the folder's release, and your changes")
    p_status.add_argument("directory", nargs="?", type=Path, default=Path("."))
    p_status.add_argument(
        "--offline", action="store_true", help="do not look for a newer release"
    )
    p_status.set_defaults(func=cmd_course_status)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    # Only our own logger talks at INFO: setting the root logger there as well
    # turned every HTTP request huggingface_hub makes into a line of output.
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    logging.getLogger("cs_lab").setLevel(
        logging.DEBUG if args.verbose else logging.INFO
    )
    _use_cache(args)
    try:
        return args.func(args)
    except _course_errors() as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _course_errors() -> Tuple[type, ...]:
    from .course import CourseError, FolderError, ReleaseError

    return (CourseError, FolderError, ReleaseError)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
