"""
base.py
-------

This module defines the **ServiceBase** class, a minimal base for all
services in the CoSMIC Coaching Writer. It provides a shared root path
and optional logging configuration.
"""

import os


class ServiceBase:
    """
    Base class for all services.

    Attributes:
        log_file (str | None): Optional log file path.
        root (str): Project root directory (two levels up from this file).
    """

    def __init__(self, log_file: str | None = None):
        self.log_file = log_file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.root = f"{current_dir}/../.."

    def set_log_file(self, log_file: str):
        """
        Update the log file path.

        Args:
            log_file (str): Path to log file.
        """
        self.log_file = log_file