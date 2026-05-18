import json
from pathlib import Path

with open(Path(__file__).resolve().parent / "name_path.json") as f:
    name_path_dict = json.load(f)
