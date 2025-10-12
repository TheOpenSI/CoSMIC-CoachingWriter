"""
base.py
-------

Defines the **ServiceBase** class, a lightweight foundation for all core
services in the CoSMIC Coaching Writer stack.

### Responsibilities
- Provide a consistent root path reference for all derived services.
- Optionally store a per-service log file path.
- Standardize initialization patterns for dependency injection.

### Notes
This class intentionally avoids heavy dependencies or initialization
side effects to ensure lightweight service startup.
"""

import os


class ServiceBase:
    """
    Abstract base class for all system services.

    Attributes:
        log_file (str | None): Optional log file path.
        root (str): Project root directory (two levels up from this file).
    """

    def __init__(self, log_file: str | None = None):
        self.log_file = log_file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.root = f"{current_dir}/../.."

    def set_log_file(self, log_file: str):
        """Dynamically set or override the active log file."""
        self.log_file = log_file
