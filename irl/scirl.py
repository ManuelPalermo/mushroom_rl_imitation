import numpy as np

def LSTDmu(psi, psi_prime, phi, gamma):
    """
    Least-Squares Temporal Differences \mu
    This function is used to calculate to estimate the expert feature
    expectation in the SCIRL algorithm both described at :
    \"Edouard Klein, Matthieu Geist, Bilal PIOT, and Olivier Pietquin.
    Inverse Reinforcement Learning through Structured Classification. In
    Advances in Neural Information Processing Systems (NIPS 2012), Lake
    Tahoe (NV, USA), December 2012.\"

    Args:
         psi (np.ndarray): feature matrix whose rows are the feature
            vectors with respect to present state;
         psi_prime (np.ndarray): feature matrix whose rows are the feature
            vectors with respect to next state;
         phi (np.ndarray): 2D feature matrix whose rows are the
            reward’s feature vectors.

    Returns:
        Weights by which the dot product with the psi matrix gives the
        feature expectations.

    """
    A = np.dot(psi.T, psi - gamma * psi_prime)
    b = np.dot(psi.T, phi)

    return np.dot(np.linalg.inv(A + 0.0001*np.identity(A.shape[0])), b)
