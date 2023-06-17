from typer import echo, style, colors
from face_recognition import MESSAGES_TYPES

def secho(message: str, message_type: str = MESSAGES_TYPES['INFO']['message'], nl: bool = True, err: bool = False, bold: bool = False, *args, **kwargs) -> None:
    """Prints a message to the console."""
    if message_type not in MESSAGES_TYPES:
        raise ValueError(f"Invalid message type: {message_type}")
    
    message_start = style(f'{message_type}', fg=colors.WHITE, bg=MESSAGES_TYPES[message_type]['color'])
    provided_message = style(message, fg=colors.WHITE, bold=bold)

    echo(f'{message_start} - {provided_message}', nl=nl, err=err, *args, **kwargs)