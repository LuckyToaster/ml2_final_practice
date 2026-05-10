from contextlib import contextmanager
from halo import Halo

@contextmanager
def spinner_task(text, spinner='dots'):
    spinner = Halo(text=text, spinner=spinner).start()
    try: yield spinner
    finally: spinner.succeed(text)

