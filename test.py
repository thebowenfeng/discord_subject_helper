import numpy as np

two_dim_list = np.empty((0, 1))

print("Shape b4 append" + str(two_dim_list.shape))

two_dim_list = np.append(two_dim_list, [[5]], axis=0)

print("New array " + str(two_dim_list))
print("Shape after append " + str(two_dim_list.shape))