"""
This module handles the user's inputs.
"""
import numpy as np

def raise_value_error(name, value, min = 0, max = np.infty):
    if value <= min:
        raise ValueError(f'{name} should be greater than or equal to {min}. The given value was {value}')
    if value >= max:
        raise ValueError(f'{name} should be lower than or equal to {max}. The given value was {value}')
