import numpy as np
import os

output = []
for file in os.listdir("output"):
    filename = os.path.join("output", file)
    print(filename)
    with open(filename, "r") as f:
        dimensions = list(map(int, [f.readline(), f.readline(), f.readline()]))
        M = np.zeros(dimensions + [3])
        n = f.readline()
        while n != "":
            m = n.split()
            M[int(m[0])][int(m[1])][int(m[2])] = [float(m[3]), float(m[4]), float(m[5])]
            n = f.readline()
    output.append(M)

output = np.array(output)

np.savez_compressed("output.npz", output=output)