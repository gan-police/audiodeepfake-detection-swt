import os
from typing import Optional


class Logger:
    def __init__(self, path: Optional[str] = None):
        self.log_file_path = os.path.abspath(path) if path is not None else ""
        self.debug = __debug__ or path is None

    def log(self, message: str) -> None:
        if self.debug:
            print(message)
        else:
            with open(self.log_file_path, "a") as file:
                file.write(message + "\n")
