import time
import matplotlib.pyplot as plt
import itertools
import numpy as np

def timetracker(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f'Time taken: {round(time.time() - start, 2)}s')
        return result
    return wrapper