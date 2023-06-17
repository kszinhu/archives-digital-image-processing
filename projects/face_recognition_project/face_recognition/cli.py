from torch.cuda import is_available

from face_recognition.util import secho 
from typing import Optional
from face_recognition import __app_name__, __version__, ERRORS, config, database
import typer

app = typer.Typer()

#*


#* Init

@app.command()
def init(db_path: str = typer.Option(database.DEFAULT_DB_FILE_PATH, "--db-path", "-d", help="Path to database file.")) -> None:
    """Initialize the application."""
    app_init_error = config.init_app(db_path)
    
    if app_init_error != config.SUCCESS:
        secho(f'Creating config file failed with: {ERRORS[app_init_error]}', message_type='ERROR')
        raise typer.Exit(code=app_init_error)
    
    db_init_error = database.init_database(database.get_database_path(config.CONFIG_FILE_PATH))
    
    if db_init_error:
        secho(f'Creating database failed with: {ERRORS[db_init_error]}', message_type='ERROR')
        raise typer.Exit(code=db_init_error)
    else:
        secho(f'Created database at: \n{db_path}\n', message_type='SUCCESS', bold=True)

    is_cuda_available = is_available()
    if is_cuda_available:
        secho('Cuda is available', message_type='INFO', bold=True)

#* Version

def _version_callback(value: bool):
    if value:
        typer.echo(f'{__app_name__} installed version {__version__}')
        raise typer.Exit()
    
@app.callback()
def main(version: Optional[bool] = typer.Option(None, "--version", "-v", callback=_version_callback, help="Show version and exit.", is_eager=True)) -> None:
    return