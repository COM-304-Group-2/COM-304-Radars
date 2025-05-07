import numpy as np

class KalmanFilter:
    def __init__(self, dt, process_noise, measurement_noise, state_dim=4, measurement_dim=2):
        self.dt = dt
        self.A = np.eye(state_dim)
        for i in range(state_dim // 2):
            self.A[i, i + 2] = dt
        self.H = np.zeros((measurement_dim, state_dim))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.Q = np.eye(state_dim) * process_noise
        self.R = np.eye(measurement_dim) * measurement_noise
        self.P = np.eye(state_dim) * 10  # Initial covariance
        self.x = np.zeros((state_dim, 1))  # Initial state
        
    def predict(self):
        # Prediction step
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(self.A, np.dot(self.P, self.A.T)) + self.Q
        return self.x

    def update(self, z):
        # Measurement update step
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(self.H, self.P))
        return self.x
