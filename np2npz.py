#
# For D dimensional numpy arrays to store in npy file
#

import numpy as np
# the numpy arrays
x = np.arange(10)
print(x)
y = np.sin(x)
z =[x, y]
# save to file
np.savez("features.npz", x=x, y=y, features=z)
loaded_outfile = np.load("features.npz")
print(loaded_outfile.files) # contains dict with array names
print(loaded_outfile['x'])
print(loaded_outfile['y'])
print(loaded_outfile['z'])
