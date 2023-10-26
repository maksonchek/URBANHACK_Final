from typing import Dict
from collections import Counter
from abc import ABC, abstractmethod
import cv2
import numpy as np


class FacadeModel(ABC):
    def __init__(self):
        self._points = []

    @abstractmethod
    def build(self, *args, **kwargs) -> None:
        pass

    def todict(self) -> Dict:
        self._validate()
        d = dict()
        for x, y, n, k in self._points:
            d[str((int(x), int(y)))] = [int(n), int(k)]
        return d

    def _validate(self):
        for point in self._points:
            x, y, n, k = point
            assert all(v >= 0 for v in point), "All values must be non-negative"
            assert n >= 1, "Number of the floor must be greater than zero"
            assert k >= 1, "Number of the vertical must be greater than zero"
            assert isinstance(n, int), "Number of the floor must be integer"
            assert isinstance(k, int), "Number of the vertical must be integer"
        counter = Counter((n, k) for _, _, n, k in self._points)
        assert all(occur == 1 for occur in counter.values()), "(n, k) pairs must be unique"


class DummyFacadeModel(FacadeModel):
    def __init__(self):
        super().__init__()

    def find_xynv(self, lines):
        val = lines
        for x in val:
            x.append((x[0] + x[2]) / 2)
            x.append((x[1] + x[3]) / 2)

        sorted_val = sorted(val, key=lambda x: x[2], reverse = True)

        sorted_val = sorted(val, key=lambda x: x[3], reverse = True)

        sorted_val[0].append(1)

        floor = 1
        start_index = 0
        count = 1

        for index in range(1, len(sorted_val)):
            if (sorted_val[index][3] > sorted_val[index - 1][5]): 
                sorted_val[index].append(floor)
            else:
                floor += 1
                start_index += 1
                sorted_val[index].append(floor)


        sorted_val = sorted(sorted_val, key=lambda x: x[0])

        sorted_val[0].append(1)
        row = 1
        start_index = 0
        count = 1
        for index in range(1, len(sorted_val)):
        
            if (sorted_val[index][0] < sorted_val[index - 1][4]): 
                sorted_val[index].append(row)
                count += 1
            else:
                row += 1
                start_index += 1
                sorted_val[index].append(row)
        sorted_val = np.array(sorted_val)
        return sorted_val[:,4:]
        
    def build(self, cells: np.ndarray, inverse_matrix = None) -> None:
        lines = [list(i) for i in cells]
        points = self.find_xynv(lines)
        if inverse_matrix is not None:
            for i, point in enumerate(points[:,:3]):
                inverse_point = cv2.perspectiveTransform(point, inverse_matrix)
                points[i] = inverse_point
        points = points.astype(int)
        unique_indices = np.array(list({tuple(x): i for i, x in enumerate(points[:, [2, 3]])}.values()))
        if len(unique_indices):
            self._points = points[unique_indices].tolist()
