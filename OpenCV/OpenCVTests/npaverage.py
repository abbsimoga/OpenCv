import numpy as np

def stabilizeCoordinates(newCords, oldCords):
    array = np.array([newCords, oldCords])
    return np.average(array, axis=0).astype(int)

print(stabilizeCoordinates([(400, 1047), (866, 736), (1094, 736), (1522, 1047)],[(477, 1043), (815, 753), (1097, 746), (1546, 1042)]))