from face_recognition import DB_WRITE_ERROR, SOMETHING_BROKE_ERROR, SUCCESS
from .config import CONFIG_FILE_PATH

from json import loads, dumps
from typer import confirm
from typing import Dict, Any
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
        if db_path.exists():
            if not confirm(f'Database path "{db_path}" already exists. Do you want to overwrite it?', default=False):
                return SUCCESS
        db_path.write_text("{}")
    except OSError:
        return DB_WRITE_ERROR

    return SUCCESS


def read_from_database(key: str) -> Any:
    """Read data from database."""
    db_path = get_database_path(config_file=CONFIG_FILE_PATH)
    try:
        db = loads(db_path.read_text())
    except OSError:
        raise SystemError("DATABASE READ ERROR: Could not read from database")
    except Exception:
        raise SystemError("SOMETHING BROKE ERROR: Something broke")

    return db[key]


def write_to_database(data: Dict[str, str] | Dict[str, Dict[str, str]]) -> int:
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
