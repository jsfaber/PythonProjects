# Working through DeeplearningBook - Trying everything in book

import numpy as np

def Chapter2():
    # 1
    x = np.array ((1,2,3))
    X = np.matrix([[1,2],[3,4]])
    tensor = np.array([
        [[1,2],[3,4]],
        [[5,6],[7,8]]])
    XT = np.transpose(X)
    # 2
    X_sq = X*X

    # 3
    I = np.eye(3)
    b = np.matrix((-1,1)).transpose()
    X_inv = np.linalg.inv(X)
    y = X_inv*b
    y_ = np.linalg.solve(X,b)

    # 5
    x_norm = np.linalg.norm(x)

    # 6
    X_diag = np.diag(X,0)

    # 7
    X = np.matrix([[1,2,3],[4,5,6],[7,8,9]])
    [V,D] = np.linalg.eig(X)
    print(V)
    print(D)

def main():
    Chapter2()

# Start main
if __name__ == "__main__":
    main()

