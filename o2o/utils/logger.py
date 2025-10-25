import csv
import os
from typing import Dict, Any


class CSVLogger:
    def __init__(self, path: str, fieldnames: list[str]):
        dirn = os.path.dirname(path)
        if dirn:
            os.makedirs(dirn, exist_ok=True)
        self.path = path
        self.fieldnames = fieldnames
        self._init_file()

    def _init_file(self):
        write_header = not os.path.exists(self.path)
        self.f = open(self.path, "a", newline="", encoding="utf-8")
        self.w = csv.DictWriter(self.f, fieldnames=self.fieldnames)
        if write_header:
            self.w.writeheader()
            self.f.flush()

    def log(self, row: Dict[str, Any]):
        self.w.writerow(row)
        self.f.flush()

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass
