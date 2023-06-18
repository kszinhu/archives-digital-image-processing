from face_recognition import DB_WRITE_ERROR, SOMETHING_BROKE_ERROR, SUCCESS
from .config import CONFIG_FILE_PATH

from json import loads, dumps
from typing import Dict
from pathlib import Path

import configparser

DEFAULT_DB_FILE_PATH = Path.home().joinpath(f".{Path.home().stem}_face_recognition.json")


def get_database_path(config_file: Path) -> Path:
    config_parser = configparser.ConfigParser()
    config_parser.read(config_file)
    return Path(config_parser["General"]["database"])


def init_database(db_path: Path) -> int:
    """Create face-recognition database."""
    try:
        db_path.write_text("{}")
    except OSError:
        return DB_WRITE_ERROR

    return SUCCESS


def write_to_database(data: Dict[str, str]) -> int:
    """Write data to database."""
    db_path = get_database_path(config_file=CONFIG_FILE_PATH)
    try:
        db = loads(db_path.read_text())

        for key, value in data.items():
            db[key] = value

        db_path.write_text(dumps(db))

    except OSError:
        return DB_WRITE_ERROR
    except Exception:
        return SOMETHING_BROKE_ERROR

    return SUCCESS
