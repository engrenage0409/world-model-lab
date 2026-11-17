import json
from pathlib import Path

import pytest

from scripts import setup_baselines


def test_manifest_has_three_pinned_upstream_sources():
    data = json.loads(setup_baselines.MANIFEST.read_text(encoding="utf-8"))
    entries = data["sources"]
    assert {item["version"] for item in entries} == {"v1", "v2", "v3"}
    assert len({item["folder"] for item in entries}) == 3
    for item in entries:
        assert item["url"].startswith("https://github.com/")
        assert len(item["commit"]) == 40
        assert all(ch in "0123456789abcdef" for ch in item["commit"])


def test_setup_refuses_to_touch_existing_checkout(tmp_path):
    target = tmp_path / "external" / "dreamer_v3"
    target.mkdir(parents=True)
    marker = target / "keep.txt"
    marker.write_text("important", encoding="utf-8")
    entry = {"folder": "dreamer_v3", "url": "https://github.com/example/repo.git",
             "commit": "0"*40, "name": "test"}
    with pytest.raises(FileExistsError):
        setup_baselines.install_source(entry, root=tmp_path)
    assert marker.read_text(encoding="utf-8") == "important"
