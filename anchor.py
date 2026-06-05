import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

from sklearn.cross_decomposition import CCA
from sklearn.neighbors import NearestNeighbors


import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances_argmin_min

# 通过k-means选择锚点
def sampling_kmeans(X, anchor_num=50):
    inds = []
    for x in X:
        KM = KMeans(n_clusters=anchor_num, random_state=1234).fit(x)
        anchor = KM.cluster_centers_
        inds.append(anchor)

    return inds


# 通过k-means++选择锚点
def sampling_minikmeans(X, anchor_num=50):
    inds = []
    for x in X:
        KM = MiniBatchKMeans(init="k-means++", n_clusters=anchor_num, random_state=1234, batch_size=1000).fit(x)
        anchor = KM.cluster_centers_
        inds.append(anchor)

    return inds


# 通过随机采样选择锚点
def sampling_random(X, anchor_num=50):
    inds = []
    for x in X:
        N = x.shape[0]
        ind = np.random.choice([i for i in range(N)], anchor_num)
        inds.append(x[ind])

    return inds


# ================================================================================================================

def sampling_gmm(X, anchor_num=50):
    """使用高斯混合模型(GMM)选择锚点，适合非球形分布数据"""
    inds = []
    for x in X:
        gmm = GaussianMixture(
            n_components=anchor_num,
            covariance_type='diag',  # 适合高维数据
            init_params='kmeans',    # 用kmeans初始化加速收敛
            max_iter=200,
            random_state=1234
        ).fit(x)
        # 选择每个成分中最具代表性的点(均值点)
        inds.append(gmm.means_)
    return inds
#
# def sampling_kmeans(X, anchor_num=50):
#     """优化后的k-means++锚点选择算法"""
#     inds = []
#     for x in X:
#         KM = KMeans(
#             n_clusters=anchor_num,
#             init='k-means++',
#             n_init=10,              # 增加初始化次数提高质量
#             max_iter=300,
#             tol=1e-5,               # 更严格的收敛标准
#             algorithm='elkan',      # 更高效的算法
#             random_state=1234
#         ).fit(x)
#         # 选择距离中心最近的实际数据点作为锚点(而非直接使用中心点)
#         closest, _ = pairwise_distances_argmin_min(KM.cluster_centers_, x)
#         inds.append(x[closest])
#     return inds
#
# def sampling_minikmeans(X, anchor_num=50):
#     """优化后的mini-batch k-means++锚点选择算法"""
#     inds = []
#     for x in X:
#         batch_size = min(2048, max(512, len(x)//10))  # 自适应批次大小
#         KM = MiniBatchKMeans(
#             n_clusters=anchor_num,
#             init='k-means++',
#             n_init=5,
#             max_iter=150,
#             batch_size=batch_size,
#             compute_labels=False,
#             random_state=1234,
#             init_size=3*anchor_num,  # 更大的初始化样本
#             reassignment_ratio=0.005 # 更少的重新分配
#         ).fit(x)
#         # 同样选择实际数据点而非中心点
#         closest, _ = pairwise_distances_argmin_min(KM.cluster_centers_, x)
#         inds.append(x[closest])
#     return inds