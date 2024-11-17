import numpy as np


class EWM_Topsis_Vikor:
    def __init__(self, matrix):
        '''
        Initialize the class with a decision matrix.

        Args:
            matrix: Input decision matrix
        '''
        self.matrix = matrix
        # Calculate the weights using entropy method
        self.weights = self.entropy_weight(matrix)

    @staticmethod
    def entropy_weight(matrix):
        '''
        Calculate the weights using the entropy method.

        Args:
            matrix: Input matrix

        Returns:
            weights: The calculated weights
        '''
        # Normalize the matrix by column sum
        matrix_normalized = matrix / np.sum(matrix, axis=0)

        # Calculate the entropy of each criterion
        entropy = - np.sum(matrix_normalized * np.log2(matrix_normalized), axis=0)

        # Calculate the weights based on entropy
        weights = (1 - entropy) / np.sum(1 - entropy)

        return weights

    @staticmethod
    def normalize_matrix(matrix):
        '''
        Normalize the decision matrix using the Euclidean norm.

        Args:
            matrix: Input decision matrix

        Returns:
            matrix_normalized: The normalized decision matrix
        '''
        return matrix / np.sqrt(np.sum(matrix ** 2, axis=0))

    def topsis_vikor(self, k):
        '''
        Apply TOPSIS and VIKOR methods to calculate the Q values.

        Args:
            k: Weight coefficient (trade-off between TOPSIS and VIKOR)

        Returns:
            Q: The calculated Q value
        '''
        # Normalize the decision matrix
        matrix_normalized = self.normalize_matrix(self.matrix)

        # Weight the normalized matrix with the entropy weights
        matrix_weighted = matrix_normalized * self.weights

        # Determine the positive and negative ideal solutions
        positive_ideal = np.max(matrix_weighted, axis=1)
        negative_ideal = np.min(matrix_weighted, axis=1)

        # Calculate distances from positive and negative ideal solutions (TOPSIS)
        d_plus = np.sqrt(np.sum((matrix_weighted - positive_ideal[:, np.newaxis]) ** 2, axis=1))
        d_minus = np.sqrt(np.sum((matrix_weighted - negative_ideal[:, np.newaxis]) ** 2, axis=1))

        # Calculate VIKOR distances
        s_plus = np.sum((-matrix_weighted + positive_ideal[:, np.newaxis]) /
                        (positive_ideal[:, np.newaxis] - negative_ideal[:, np.newaxis]), axis=1)
        s_minus = np.sum((matrix_weighted - negative_ideal[:, np.newaxis]) /
                         (positive_ideal[:, np.newaxis] - negative_ideal[:, np.newaxis]), axis=1)

        r_plus = np.max((-matrix_weighted + positive_ideal[:, np.newaxis]) /
                        (positive_ideal[:, np.newaxis] - negative_ideal[:, np.newaxis]))
        r_minus = np.max((matrix_weighted - negative_ideal[:, np.newaxis]) /
                         (positive_ideal[:, np.newaxis] - negative_ideal[:, np.newaxis]))

        # Calculate SDR for TOPSIS and VIKOR
        SDR_plus = 1 / ((1 / s_plus) + (1 / d_plus) + (1 / r_plus))
        SDR_minus = 1 / ((1 / s_minus) + (1 / d_minus) + (1 / r_minus))

        # Calculate the final Q value based on k (trade-off parameter)
        Q = k * SDR_plus + (1 - k) * SDR_minus

        return Q

    def find_inflection_points(self):
        '''
        Find the inflection points in the Q values by varying k.

        Returns:
            inflection_ks: The k values where the inflection points occur
        '''
        results = []
        # Loop through a range of k values to calculate Q for each
        for k in np.linspace(0, 1, num=10000001):
            q_values = self.topsis_vikor(k)
            # Sort the Q values and store the results
            sorted_q = sorted(enumerate(q_values), key=lambda x: x[1])
            results.append(sorted_q)

        inflection_ks = []
        # Detect the inflection points where the order of Q values changes
        for i in range(0, len(results) - 1):
            if (results[i][0][0] != results[i + 1][0][0] or
                    results[i][1][0] != results[i + 1][1][0] or
                    results[i][2][0] != results[i + 1][2][0]):
                inflection_ks.append(i / (len(results) - 1))

        return inflection_ks