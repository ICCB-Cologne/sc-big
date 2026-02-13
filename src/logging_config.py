"""
Logging Configuration
=====================

Centralized logging configuration for the Bayesian SNV caller.

License
-------
This file is part of ``sc-big``.

Copyright (C) 2026 Daniel Schütte, daniel.schuette@iccb-cologne.org

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
more details. You should have received a copy of the GNU General Public
License along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
import logging
import sys


def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Configure logging for the application.

    Parameters
    ----------
    verbose : bool
        If True, set log level to DEBUG. Otherwise, use INFO.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Suppress matplotlib logging
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.pyplot").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    return logging.getLogger("bayesian_snv_caller")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Parameters
    ----------
    name : str
        Name for the logger (typically __name__).

    Returns
    -------
    logging.Logger
        Logger instance.
    """
    return logging.getLogger(f"bayesian_snv_caller.{name}")
