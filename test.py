import numpy as np

if __name__ == '__main__':
    a = np.arange(9).reshape(3,3)
    b = np.array([2,3,4])
    print(np.where(a>b)[0].size)