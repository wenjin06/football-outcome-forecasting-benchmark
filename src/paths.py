"""
Repository path resolution.

Every script derives its input/output directories from this module, so the
repository can be cloned and run from any location without editing paths.
Raw match data (football-data.co.uk CSVs) is read from the directory given
by the FOOTBALL_DATA_DIR environment variable, falling back to data/raw/
inside the repository.
"""
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, "results")
PROCESSED = os.path.join(BASE, "data", "processed")
RAW = os.path.join(BASE, "data", "raw")
RAW_UNDERSTAT = os.path.join(BASE, "data", "raw_understat")
FIGURES = os.path.join(BASE, "paper", "figures")
TABLES = os.path.join(BASE, "paper", "tables")
PAPER = os.path.join(BASE, "paper")


def raw_data_dir():
    """Directory containing football-data.co.uk CSVs (leagues E0/D1/F1/I1/SP1).

    Override with the FOOTBALL_DATA_DIR environment variable; defaults to
    data/raw/ inside the repository.
    """
    return os.environ.get("FOOTBALL_DATA_DIR", RAW)
