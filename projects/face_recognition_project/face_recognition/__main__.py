from face_recognition import __app_name__, cli

# python -m face_recognition -v
def main():
    cli.app(prog_name=__app_name__)

if __name__ == "__main__":
    main()