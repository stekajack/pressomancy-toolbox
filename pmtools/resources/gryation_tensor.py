import numpy as np

class GyrationTensor:

    def __init__(self, part_positions: np.ndarray):
        if part_positions.ndim != 2 or part_positions.shape[1] != 3:
            raise ValueError("GyrationTensor expects an (N, 3) array of 3D positions.")
        self._tensor = self._compute(part_positions)
        self._eigenvalues = np.empty(3, dtype=float)
        self._eigenvectors = np.empty((3, 3), dtype=float)
        self._comp_done = False
    
    def _compute(self, positions: np.ndarray):
        """
        Calculate the gyration tensor from the provided positions.
        The tensor is computed as the outer product of the position vectors.
        """
        r_cm = np.mean(positions, axis=0)
        r_sub = positions - r_cm

        # Compute the gyration tensor elements
        xx = np.mean(r_sub[:, 0]**2)
        yy = np.mean(r_sub[:, 1]**2)
        zz = np.mean(r_sub[:, 2]**2)
        xy = np.mean(r_sub[:, 0]*r_sub[:, 1])
        xz = np.mean(r_sub[:, 0]*r_sub[:, 2])
        yz = np.mean(r_sub[:, 1]*r_sub[:, 2])

        # Construct the gyration tensor
        gyration_tensor = np.array(
            [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
        return gyration_tensor
    
    def _sanity_check(self):
        if self._comp_done is False:
            self._compute_eig()
            self._comp_done = True
        
    @property
    def array(self):
        return self._tensor

    @property
    def eigenvalues(self):
        self._sanity_check()
        return self._eigenvalues

    @property
    def eigenvectors(self):
        self._sanity_check()        
        return self._eigenvectors

    def _compute_eig(self):
        _, self._eigenvectors = np.linalg.eig(self._tensor)
        P_inv = np.linalg.inv(self._eigenvectors) # type: ignore
        X = np.dot(P_inv, self._tensor)
        B = np.dot(X, self._eigenvectors)
        self._eigenvalues = np.diag(B)
        order=np.argsort(self._eigenvalues)
        self._eigenvalues= self._eigenvalues[order]
        self._eigenvectors = self._eigenvectors[:, order]
    
    def get_R2(self):
        self._sanity_check()        
        return np.sum(self._eigenvalues)

    def get_k2(self):
        x,y,z=self._eigenvalues
        return 3/2*(x**2+y**2+z**2)/np.sum(self._eigenvalues)**2-1/2

    def __repr__(self):
        return f"GyrationTensor(\n{self._tensor})"