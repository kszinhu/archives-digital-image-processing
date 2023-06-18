from face_recognition import (
    DIR_ERROR,
    DB_WRITE_ERROR,
    FILE_ERROR,
    SUCCESS,
    __app_name__,
)

from pathlib import Path
from typer import get_app_dir, confirm

import configparser


CONFIG_DIR_PATH = Path(get_app_dir(__app_name__))
CONFIG_FILE_PATH = CONFIG_DIR_PATH / "config.ini"


def _init_config_file() -> int:
    try:
        CONFIG_DIR_PATH.mkdir(exist_ok=True)
    except OSError:
        return DIR_ERROR
    try:
        if CONFIG_FILE_PATH.exists():
            if not confirm("Config file already exists. Do you want to overwrite it?", default=False):
                return SUCCESS
        CONFIG_FILE_PATH.touch(exist_ok=True)
    except OSError:
        return FILE_ERROR
    return SUCCESS


def _create_database(db_path: str) -> int:
    config_parser = configparser.ConfigParser()
    config_parser["General"] = {"database": db_path}
    try:
        if Path(db_path).exists():
            if not confirm(f'Database path "{db_path}" already exists. Do you want to overwrite it?', default=False):
                return SUCCESS
        with CONFIG_FILE_PATH.open("w") as file:
            config_parser.write(file)
    except OSError:
        return DB_WRITE_ERROR
    return SUCCESS


def init_app(config_path: str) -> int:
    """Initialize the application."""
    config_code = _init_config_file()
    if config_code != SUCCESS:
        return config_code

    database_code = _create_database(config_path)
    if database_code != SUCCESS:
        return database_code

    return SUCCESS


def get_config() -> configparser.ConfigParser:
    config_parser = configparser.ConfigParser()
    config_parser.read(CONFIG_FILE_PATH)
    return config_parser
