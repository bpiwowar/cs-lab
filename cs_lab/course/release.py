"""A course release: ``release.json`` and the zip it describes.

The course's build writes, next to the student zip, a ``release.json`` listing
each file of the zip with its hash (jupytext-notebook-helper's
``release.py``). Fetching one is plain HTTP(S) — or ``file://``, for tests and
for a course folder on a shared disk.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional

from .. import __version__

#: The release.json layouts this version reads. A course's build raises the
#: format when a release relies on something older versions would get wrong
#: (2: `rooms`), so that they refuse it rather than misread it.
SUPPORTED_FORMAT = 2


class ReleaseError(Exception):
    """A release that cannot be fetched or used."""


@dataclass
class Room:
    """A teaching room (or any group of machines sharing a disk) where the
    course installed an environment and/or a model cache, as absolute paths:
    a folder uses them on a machine where they exist."""

    name: str = ""
    shared_env: Optional[str] = None
    shared_cache: Optional[str] = None


def _rooms(data: dict) -> List[Room]:
    """``rooms`` — or, from a release before it, the one room its top-level
    ``shared_env`` / ``shared_cache`` describe."""
    rooms = [
        Room(
            str(room.get("name", "")),
            room.get("shared_env") or None,
            room.get("shared_cache") or None,
        )
        for room in data.get("rooms") or []
    ]
    if not rooms and (data.get("shared_env") or data.get("shared_cache")):
        rooms.append(
            Room("", data.get("shared_env") or None, data.get("shared_cache") or None)
        )
    return rooms


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def release_url(url: str) -> str:
    """The URL of release.json, given it or the directory that holds it —
    as a URL or a local path."""
    if "://" not in url:  # a local path
        url = Path(url).resolve().as_uri()
    if url.endswith(".json"):
        return url
    return url.rstrip("/") + "/release.json"


def _read(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            return response.read()
    except (OSError, ValueError) as exc:  # URLError is an OSError
        raise ReleaseError(f"cannot fetch {url}: {exc}") from exc


@dataclass
class Release:
    url: str
    name: str
    id: str
    created: str
    bundle_path: str
    bundle_sha256: str
    files: Dict[str, str]
    env: List[str] = field(default_factory=list)
    lock: Optional[str] = None
    source: Optional[str] = None
    #: The teaching rooms with a shared environment and/or model cache for
    #: this release, in order of preference: ``init`` uses the first whose
    #: paths exist on the machine.
    rooms: List[Room] = field(default_factory=list)

    @classmethod
    def fetch(cls, url: str) -> "Release":
        url = release_url(url)
        try:
            data = json.loads(_read(url))
        except ValueError as exc:
            raise ReleaseError(f"{url} is not a release description") from exc
        return cls.from_json(url, data)

    @classmethod
    def from_json(cls, url: str, data: dict) -> "Release":
        if data.get("format", 0) > SUPPORTED_FORMAT:
            raise ReleaseError(
                f"{url} uses release format {data['format']}, newer than this "
                f"cs-lab ({__version__}) reads: run it as `uvx cs-lab@latest ...`"
                " (or upgrade cs-lab)"
            )
        try:
            return cls(
                url=url,
                name=data["name"],
                id=data["id"],
                created=data.get("created", ""),
                bundle_path=data["bundle"]["path"],
                bundle_sha256=data["bundle"]["sha256"],
                files=dict(data["files"]),
                env=list(data.get("env", [])),
                lock=data.get("lock"),
                source=data.get("source"),
                rooms=_rooms(data),
            )
        except KeyError as exc:
            raise ReleaseError(f"{url}: missing {exc} in the release") from exc

    @property
    def bundle_url(self) -> str:
        return urllib.parse.urljoin(self.url, self.bundle_path)

    def content_files(self) -> List[str]:
        """The files that are not the environment's."""
        return [path for path in self.files if path not in self.env]

    def download(self, target: Path) -> Path:
        """Fetch the zip, check it, and extract it into ``target`` (created).

        Every file is checked against its hash, and no path may leave
        ``target``: a release is data from a server, not something trusted.
        """
        data = _read(self.bundle_url)
        if hashlib.sha256(data).hexdigest() != self.bundle_sha256.removeprefix(
            "sha256:"
        ):
            raise ReleaseError(
                f"{self.bundle_url} does not match its release description "
                "(was it updated in the middle of the download? try again)"
            )
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile() as stream:
            stream.write(data)
            stream.seek(0)
            with zipfile.ZipFile(stream) as archive:
                for info in archive.infolist():
                    if info.is_dir():
                        continue
                    self._extract(archive, info, target)
        for path, digest in self.files.items():
            if sha256_file(target / path) != digest:
                raise ReleaseError(f"{path}: content does not match the release")
        return target

    def _extract(self, archive: zipfile.ZipFile, info, target: Path) -> None:
        relative = PurePosixPath(info.filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ReleaseError(f"refusing a path outside the folder: {relative}")
        if info.filename not in self.files:
            raise ReleaseError(f"{info.filename} is not in the release description")
        destination = target.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source, open(destination, "wb") as out:
            shutil.copyfileobj(source, out)
