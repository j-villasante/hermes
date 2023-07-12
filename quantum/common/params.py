from qiskit.quantum_info import DensityMatrix
from numpy import linalg as LA

def polarization(density_matrix: DensityMatrix) -> int:
    s1 = 2 * density_matrix.data[0, 0].real - 1
    s2 = 2 * density_matrix.data[0, 1].real
    s3 = - 2 * density_matrix.data[0, 1].imag
    return LA.norm([s1, s2, s3])
