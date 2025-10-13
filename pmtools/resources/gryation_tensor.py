import numpy as np

class GyrationTensor:
    """
    Compute and expose properties of the gyration tensor for a set of points.

    Given 3D particle positions, this class computes the (3×3) gyration tensor,
    as well as its eigenvalues and eigenvectors, and provides convenience
    accessors for derived quantities such as :math:`R_g^2` and the relative
    shape anisotropy :math:`\\kappa^2`.

    Notes
    -----
    The gyration tensor is defined for positions :math:`\\{\\mathbf{r}_i\\}` as

    .. math::

        \\mathbf{S} = \\langle (\\mathbf{r}_i - \\mathbf{r}_{\\rm cm})
        (\\mathbf{r}_i - \\mathbf{r}_{\\rm cm})^\\top \\rangle ,

    where :math:`\\mathbf{r}_{\\rm cm}` is the center of mass and
    :math:`\\langle\\cdot\\rangle` denotes the average over particles.
    """

    def __init__(self, part_positions: np.ndarray) -> None:
        """
        Initialize the gyration tensor from particle positions.

        Parameters
        ----------
        part_positions : numpy.ndarray
            Array of shape ``(N, 3)`` containing 3D particle coordinates.

        Raises
        ------
        ValueError
            If ``part_positions`` is not a 2D array with 3 columns.
        """
        if part_positions.ndim != 2 or part_positions.shape[1] != 3:
            raise ValueError("GyrationTensor expects an (N, 3) array of 3D positions.")
        self._tensor = self._compute(part_positions)
        self._eigenvalues = np.empty(3, dtype=float)
        self._eigenvectors = np.empty((3, 3), dtype=float)
        self._comp_done = False
    
    def _compute(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute the gyration tensor.

        The tensor is computed as the average outer product of particle
        displacements relative to the center of mass.

        Parameters
        ----------
        positions : numpy.ndarray
            Array of shape ``(N, 3)`` with particle coordinates.

        Returns
        -------
        numpy.ndarray
            The gyration tensor of shape ``(3, 3)``.
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
    
    def _sanity_check(self) -> None:
        """
        Ensure eigen-decomposition has been performed.

        Notes
        -----
        Lazily computes eigenvalues and eigenvectors on first access.
        """
        if self._comp_done is False:
            self._compute_eig()
            self._comp_done = True
        
    @property
    def array(self) -> np.ndarray:
        """
        The raw gyration tensor.

        Returns
        -------
        numpy.ndarray
            Tensor of shape ``(3, 3)``.
        """
        return self._tensor

    @property
    def eigenvalues(self) -> np.ndarray:
        """
        Eigenvalues of the gyration tensor (ascending order).

        Returns
        -------
        numpy.ndarray
            Array of shape ``(3,)`` with eigenvalues :math:`\\lambda_1 \\le \\lambda_2 \\le \\lambda_3`.
        """
        self._sanity_check()
        return self._eigenvalues

    @property
    def eigenvectors(self) -> np.ndarray:
        """
        Eigenvectors of the gyration tensor.

        Returns
        -------
        numpy.ndarray
            Matrix of shape ``(3, 3)`` whose columns are normalized eigenvectors
            corresponding to ``eigenvalues``.
        """
        self._sanity_check()        
        return self._eigenvectors

    def _compute_eig(self) -> None:
        """
        Compute and sort the eigen-decomposition of the gyration tensor.

        Notes
        -----
        Uses ``numpy.linalg.eig`` and sorts the eigenpairs by ascending
        eigenvalue; both ``self._eigenvalues`` and ``self._eigenvectors`` are
        updated in-place.
        """
        _, self._eigenvectors = np.linalg.eig(self._tensor)
        P_inv = np.linalg.inv(self._eigenvectors) # type: ignore
        X = np.dot(P_inv, self._tensor)
        B = np.dot(X, self._eigenvectors)
        self._eigenvalues = np.diag(B)
        order=np.argsort(self._eigenvalues)
        self._eigenvalues= self._eigenvalues[order]
        self._eigenvectors = self._eigenvectors[:, order]
    
    def get_R2(self) -> float:
        """
        Radius of gyration squared, :math:`R_g^2`.

        Returns
        -------
        float
            Sum of the eigenvalues of the gyration tensor.
        """
        self._sanity_check()        
        return np.sum(self._eigenvalues)

    def get_k2(self) -> float:
        """
        Relative shape anisotropy, :math:`\\kappa^2`.

        Defined here as

        .. math::

            \\kappa^2 = \\frac{3}{2} \\frac{\\lambda_1^2 + \\lambda_2^2 + \\lambda_3^2}
            {(\\lambda_1 + \\lambda_2 + \\lambda_3)^2} - \\frac{1}{2}

        where :math:`\\lambda_i` are the eigenvalues of the gyration tensor.

        Returns
        -------
        float
            Dimensionless measure of anisotropy in ``[0, 1]``.
        """
        self._sanity_check()        
        x,y,z=self._eigenvalues
        return 3/2*(x**2+y**2+z**2)/np.sum(self._eigenvalues)**2-1/2

    def __repr__(self) -> str:
        """
        Official string representation.

        Returns
        -------
        str
            Readable representation including the tensor array.
        """
        return f"GyrationTensor(\n{self._tensor})"