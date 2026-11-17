"""Clone and pin external Dreamer baselines without altering the user's environment.

The upstream projects are kept as separate Git checkouts under external/.
Dependencies are NOT installed automatically (several use legacy Gym 0.19).
"""
import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "baselines" / "sources.json"


def git(*args, cwd=None):
    subprocess.run(["git", *map(str, args)], cwd=cwd, check=True)


def install_source(entry, root=ROOT):
    destination = root / "external" / entry["folder"]
    if destination.exists():
        raise FileExistsError(
            f"{destination} already exists. Refusing to modify it; "
            "inspect the checkout or choose a clean workspace.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    git("clone", "--no-checkout", entry["url"], destination)
    try:
        git("checkout", "--detach", entry["commit"], cwd=destination)
    except Exception:
        # Avoid leaving a misleading partially prepared checkout.
        print(f"Checkout failed for {entry['name']}. Inspect {destination} manually.")
        raise
    print(f"Ready: {entry['name']} at {destination} ({entry['commit'][:12]})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["v1", "v2", "v3", "all"], required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with MANIFEST.open(encoding="utf-8") as handle:
        entries = json.load(handle)["sources"]
    selected = entries if args.version == "all" else [
        e for e in entries if e["version"] == args.version]
    for entry in selected:
        print(f"{entry['version']}: {entry['url']} @ {entry['commit']}")
        if not args.dry_run:
            install_source(entry)


if __name__ == "__main__":
    main()
