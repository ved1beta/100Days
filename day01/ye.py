import numpy as np
import time

# Parameters
N = 1024  # Number of elements

# Initialize host arrays
h_A = np.ones(N, dtype=np.float32)  # [1.0, 1.0, ...]
h_B = np.arange(N, dtype=np.float32)  # [0.0, 1.0, ..., 1023.0]
h_C = np.empty_like(h_A)  # Result array

# Time the vector addition
start = time.time()
h_C = h_A + h_B  # Vectorized addition
computation_time = (time.time() - start) * 1000  # Convert to milliseconds

# Print results (last 10 elements)
print(f"Computation Time: {computation_time:.3f} ms")
print("\nLast 10 elements of h_C:")
print(h_C[-10:])