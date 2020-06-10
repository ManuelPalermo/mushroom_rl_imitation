import numpy as np


class KERNEL:
    def __call__(self, data):
        raise NotImplementedError


class RBF(KERNEL):
    def __init__(self, compare_values, range_denominator=10):
        self._compare_values = compare_values
        self._range_denominator = range_denominator

        self._max_values = np.max(compare_values, axis=0)
        self._min_values = np.min(compare_values, axis=0)

        self._sigma = 2 * np.power((self._max_values - self._min_values) / self._range_denominator, 2)

    def __call__(self, data):
        results = []
        for sample in data:
            results.append(self._transform_data(np.array([sample])))
        return np.vstack(results)

    def _transform_data(self, data):
        distance_standardized = np.power(data - self._compare_values, 2) / self._sigma
        exp_distance = np.exp(-1 * np.sum(distance_standardized, axis=1))
        transformed_vector = np.append(exp_distance, 1)
        return transformed_vector
