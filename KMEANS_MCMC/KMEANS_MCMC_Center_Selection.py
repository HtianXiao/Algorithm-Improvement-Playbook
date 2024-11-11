import numpy as np

class ClusterImprovement:
    def __init__(self, X, K):
        self.X = X
        self.K = K
        self.centers = None

    def generate_initial_centers(self):
        n_samples, n_features = self.X.shape
        # Randomly choose the first cluster center
        self.centers = [self.X[np.random.choice(n_samples)]]
        # Choose the next cluster center
        for i in range(self.K - 1):
            # Change the proposal distribution
            distances = np.sqrt(((self.X - self.centers[-1])**2).sum(axis=1))
            d = distances**2
            prob = (d / d.sum()) + 1/(2*len(self.X))
            prob /= prob.sum()
            # Use MCMC sampling
            for j in range(100):
                idx = np.random.choice(n_samples, size=1, p=prob)[0]
                proposed_center = self.X[idx]
                new_distances = np.sqrt(((self.X - proposed_center)**2).sum(axis=1))
                nd = new_distances**2
                # Regularization Term
                new_prob = (nd / nd.sum()) + 1/(2*len(self.X))
                new_prob /= new_prob.sum()
                acceptance_prob = min(1, new_prob[idx] / prob[idx])
                if np.random.rand() < acceptance_prob:
                    self.centers.append(proposed_center)
                    break
        return np.array(self.centers)

# Example usage:
# Assuming X is a numpy array of your data points and K is the number of clusters you want
# X = np.array([...])  # Your data points
# K = 3  # Number of clusters
# ci = ClusterImprovement(X, K)
# initial_centers = ci.generate_initial_centers()