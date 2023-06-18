from face_recognition import __app_name__, __version__, ERRORS, config, database
from .utils import secho, parser_additional_params
from .dataset.handler import load_dataset
from .database import write_to_database

from pathlib import Path
from typing import Optional, List, Dict
from torch.cuda import is_available

import typer

import pdb

app = typer.Typer()


# * Set dataset path
# python -m face_recognition dataset -d ATT_FACES -p /home/username/datasets/att_faces min-samples=15,train-size=0.8
@app.command()
def dataset(
    type: str = typer.Option(..., "--type", "-t", prompt="Please enter dataset type", help="Set dataset type."),
    path: Path = typer.Option(..., "--path", "-p", prompt="Please enter dataset path", help="Set dataset path."),
    dataset_params: Optional[Dict[str, str]] = typer.Argument(
        None, help="Dataset parameters.", parser=parser_additional_params
    ),
) -> None:
    """Set dataset path."""
    pdb.set_trace()
    try:
        loaded_dataset = (
            load_dataset(type, path, kwargs=dataset_params) if dataset_params is not None else load_dataset(type, path)
        )
    except Exception as error:
        secho(f"Loading dataset failed with: {error}", message_type="ERROR", err=True)
        raise typer.Exit(code=1)

    secho(f"Loaded dataset: {loaded_dataset.name}", message_type="SUCCESS", bold=True)

    try:
        if loaded_dataset.name is not None and path is not None:
            params = dataset_params if dataset_params is not None else {}
            write_to_database(
                {
                    "dataset": loaded_dataset.name,
                    "dataset_path": str(path.absolute()),
                    "dataset_params": params,
                }  # type: ignore
            )
    except Exception as error:
        secho(f"Writing to config file failed with: {error}", message_type="ERROR", err=True)
        raise typer.Exit(code=1)


# * Init


@app.command()
def init(
    db_path: str = typer.Option(
        database.DEFAULT_DB_FILE_PATH,
        "--db-path",
        "-d",
        help="Path to config database file.",
    )
) -> None:
    """Initialize the application."""
    app_init_error = config.init_app(db_path)

    if app_init_error != config.SUCCESS:
        secho(f"Creating config file failed with: {ERRORS[app_init_error]}", message_type="ERROR")
        raise typer.Exit(code=app_init_error)

    db_init_error = database.init_database(database.get_database_path(config.CONFIG_FILE_PATH))

    if db_init_error:
        secho(f"\nCreating database failed with: {ERRORS[db_init_error]}", message_type="ERROR")
        raise typer.Exit(code=db_init_error)
    else:
        secho(f"Database at: \n{db_path}", message_type="SUCCESS", bold=True, nl=True)

    is_cuda_available = is_available()
    if is_cuda_available:
        secho("Cuda is available", message_type="INFO")


# * Version


def _version_callback(value: bool):
    if value:
        typer.echo(f"{__app_name__} installed version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", callback=_version_callback, help="Show version and exit.", is_eager=True
    )
) -> None:
    return