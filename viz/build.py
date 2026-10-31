"""Bake training history, replays and calibration data into the standalone visualiser page.

    python viz/build.py   ->  viz/buckshot_lab.html

Inputs: train/cem_history.json (train/cem.py), viz/replays.json (train/replay.py),
viz/calibration.json (train/test_belief.py viz/calibration.json).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    history = load(os.path.join(REPO, "train", "cem_history.json"))
    replays = load(os.path.join(HERE, "replays.json"))
    calib = load(os.path.join(HERE, "calibration.json"))
    with open(os.path.join(HERE, "template.html"), encoding="utf-8") as f:
        page = f.read()
    for token, data in (("/*HISTORY*/null", history), ("/*REPLAYS*/null", replays), ("/*CALIB*/null", calib)):
        assert token in page, token
        page = page.replace(token, json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    out = os.path.join(HERE, "buckshot_lab.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote {out} ({len(page) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
