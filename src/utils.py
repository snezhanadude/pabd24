"""Demo functions for three type of heavy prediction tasks"""

import time
import numpy as np


def predict_io_bounded(area):
    """Emulate io delay"""
    time.sleep(1)
    avg_price = 200_000                 # RUB / m2
    return int(area * avg_price)


def predict_cpu_bounded(area, n=5_000_000):
    """Emulate single thread computation"""
    avg_price = sum([x for x in range(n)]) / n
    return int(area * avg_price)


def predict_cpu_multithread(area, n=5_000_000):
    """Emulate multi thread computation"""
    avg_price = np.mean(np.arange(n))
    return int(area * avg_price)