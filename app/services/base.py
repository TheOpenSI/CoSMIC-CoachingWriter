import os


class ServiceBase:
    def __init__(self, log_file: str | None = None):
        self.log_file = log_file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.root = f"{current_dir}/../.."

    def set_log_file(self, log_file: str):
        self.log_file = log_file
