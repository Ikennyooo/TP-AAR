import numpy as np
from utils import normalize_matrix
from scipy.sparse import csr_matrix
from utils import dimension_reduction


# 小规模属性图的低通滤波器（X：原始特征矩阵；A：邻接矩阵；k1：滤波器的阶数；p：滤波器参数；）
def LowPassFilter(X, A, k1, p=0.5):
    I = np.eye(A.shape[0])
    S = A + I
    S = normalize_matrix(S)
    L = I - S
    FM = I - p * L
    FMk = np.linalg.matrix_power(FM, k1)
    H_low = FMk.dot(X)
    return H_low

# 大规模属性图的低通滤波器（带有稀疏矩阵）
def LowPassFilter_sparse(X, A, Dr, k1, p=0.5):
    N = X.shape[0]
    row_I = np.arange(N)
    col_I = np.arange(N)
    data_I = np.ones(N)
    I = csr_matrix((data_I, (row_I, col_I)), shape=(N, N))
    D = np.array(Dr)
    D = D + 1
    D = np.power(D, -0.5)
    D = csr_matrix((D, (row_I, col_I)), shape=(N, N))
    S = A + I
    S = csr_matrix(S)
    S = D * S * D

    # I - p * L_S = I - p * (I - S) = (1-p)I + pS
    F_M = (1 - p) * I + p * S
    H_low = F_M * X
    f_order = k1 - 1
    while f_order > 0:
        H_low = F_M * H_low
        f_order -= 1
    return H_low


def multi_view_processing(X, A, Dr, k=2, dims=100):
    try:

        lenX = len(X)
        lenA = len(A)
        H = []

        if lenX > lenA:
            for x in X:
                if x.shape[1] >= dims:
                    x = dimension_reduction(x, dims)
                Htmp = LowPassFilter_sparse(x, A[0], Dr[0], k1=k)
                H.append(Htmp)
        else:
            for i in range(lenA - lenX):
                X.append(X[0])
            for (x, a, d) in zip(X, A, Dr):
                if x.shape[1] >= dims:
                    x = dimension_reduction(x, dims)

                Htmp = LowPassFilter_sparse(x, a, d, k1=k)
                H.append(Htmp)

        assert len(H) == max(lenX, lenA)
        return H

    except Exception as e:
        print("Error: {}".format(e))
        return
