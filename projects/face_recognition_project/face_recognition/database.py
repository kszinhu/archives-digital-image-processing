import configparser
from pathlib import Path
from face_recognition import DB_WRITE_ERROR, SUCCESS

DEFAULT_DB_FILE_PATH = Path.home().joinpath(f'.{Path.home().stem}_face_recognition.json')

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
