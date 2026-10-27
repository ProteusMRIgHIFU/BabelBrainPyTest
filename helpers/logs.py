"""
Logging helpers for the tests. Messages are shown live (log_cli in pytest.ini) and saved in
PyTest_Reports/pytest.log.
"""
import logging
import time
from contextlib import contextmanager

import numpy as np


@contextmanager
def log_duration(label):
    """Log `label` when the block starts and how long it took when it ends."""
    logging.info(f"{label}...")
    start = time.perf_counter()
    yield
    logging.info(f"{label} took {time.perf_counter() - start:.2f} s")


def describe_array(name, data):
    """Log the shape, type and value range of an array (or the size of a point list)."""
    data = np.asarray(data)
    if data.size == 0:
        logging.info(f"{name}: empty {data.dtype} array of shape {data.shape}")
    elif data.dtype == bool:
        logging.info(f"{name}: {data.shape} bool, {int(data.sum())} voxel(s) set ({100 * data.mean():.2f}%)")
    else:
        logging.info(f"{name}: {data.shape} {data.dtype}, range [{np.nanmin(data):.4g}, {np.nanmax(data):.4g}]")
