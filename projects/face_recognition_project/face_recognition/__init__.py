from typer import colors

__app_name__ = "face-recognition"
__version__ = "0.0.1"

(
    SUCCESS,
    DIR_ERROR,
    FILE_ERROR,
    DB_READ_ERROR,
    DB_WRITE_ERROR,
    SOMETHING_BROKE_ERROR,
) = range(6)

ERRORS = {
    DIR_ERROR: "Directory does not exist",
    FILE_ERROR: "File does not exist",
    DB_READ_ERROR: "Error reading from database",
    DB_WRITE_ERROR: "Error writing to database",
    SOMETHING_BROKE_ERROR: "Something broke",
}

MESSAGES_TYPES = {
    "SUCCESS": {
        "message": "SUCCESS",
        "color": colors.GREEN,
    },
    "ERROR": {
        "message": "ERROR",
        "color": colors.RED,
    },
    "INFO": {
        "message": "INFO",
        "color": colors.BLUE,
    },
}
