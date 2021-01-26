from typing import Union, List, Tuple, Sequence

import numpy as np

Number = Union[float, int]
Vector = Union[List[float], Tuple[float, float, float]]
NumericArray = Union[Sequence[Number], np.ndarray]
Color = Union[str, Sequence[Number]]
