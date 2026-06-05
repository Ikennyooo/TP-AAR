import numpy as np
from data_loader import datasets
from graph_filtering import multi_view_processing
from anchor import sampling_kmeans, sampling_minikmeans, sampling_gmm
from clustering import Efficient_multi_view_clustering_with_weights
from metrics import evaluate_clustering
import pandas as pd

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import seaborn as sns


class TPAAR():
    def __init__(self, dataname="Citeseer"):
        self.datasets = datasets

        if dataname == "ACM":
            self.alphas = [0.001, 1, 10, 100, 1000]
            self.betas = [0.001, 1, 10, 100, 1000]
        elif dataname == "DBLP":
            self.alphas = [0.001, 0.002, 1, 10, 100]
            self.betas = [0.001, 0.002, 1, 10, 100]
        elif dataname == "AMAP":
            self.alphas = [0.001, 1, 10, 100, 1000]
            self.betas = [0.001, 1, 10, 100, 1000]
        elif dataname == "AMAC":
            self.alphas = [0.001, 1, 10, 100, 1000]
            self.betas = [0.001, 1, 10, 100, 1000, 10000]
        elif dataname == "YELP":
            self.alphas = [0.001, 1, 10, 100, 1000]
            self.betas = [0.001, 1, 10, 100, 1000, 10000]
        else:
            self.alphas = [0.001, 1, 10, 100, 1000]
            self.betas = [0.001, 1, 10, 100, 1000, 10000]

        self.data_name = dataname
        self.data_type = 'graph data'

        if dataname == "ACM":
            self.max_dims = 400
        elif dataname == "DBLP":
            self.max_dims = 30
        elif dataname == "AMAP":
            self.max_dims = 100
        elif dataname == "AMAC":
            self.max_dims = 100
        elif dataname == "YELP":
            # self.max_dims = 100   # 0.7957
            # self.max_dims = 200   # 0.7957
            # self.max_dims = 300   # 0.7957
            # self.max_dims = 400   # 0.7957
            self.max_dims = 75
        else:
            self.max_dims = 100

        self.k = 2

        self.algorithms = [Efficient_multi_view_clustering_with_weights]

        self.best_paras = {
            "alpha": 0,
            "beta": 0,
            "m": 0,
        }
        self.best_re = {
            "ACC": 0,
            "NMI": 0,
            "ARI": 0,
            "F1": 0,
            "PUR": 0,
            "Time": 0
        }

    def loadData(self):
        if self.data_name in ["ACM", "DBLP", "IMDB", "YELP"]:
            X, As, Drs, gnd = self.datasets["multi-relational"](self.data_name)
            self.data_type = 'multi-relational ' + self.data_type
        elif self.data_name in ["AMAP", "AMAC"]:
            X, As, Drs, gnd = self.datasets["multi-attribute"](self.data_name)
        else:
            print("No such dataset!!!!!")
            return

        self.gnd = gnd
        self.C = len(np.unique(self.gnd))

        if self.data_name == "ACM":
            self.ms = [self.C, 10, 30, 50, 70, 100]
        elif self.data_name == "DBLP":
            self.ms = [self.C, 10, 30, 50]
        elif self.data_name == "AMAP":
            self.ms = [self.C, 10, 30, 50, 70, 100, 200, 300]
        elif self.data_name == "AMAC":
            self.ms = [self.C, 10, 30, 50, 70, 100, 200, 300]
        elif self.data_name == "YELP":
            self.ms = [self.C, 10, 30, 50, 70, 100, 200, 300]
        else:
            self.ms = [self.C, 10, 30, 50, 70, 100, 200, 300]
        self.N = X[0].shape[0]
        return X, As, Drs, gnd

    def graphFiltering(self, k=2):
        X, As, Drs, _ = self.loadData()
        H = multi_view_processing(X=X, A=As, Dr=Drs, k=k, dims=self.max_dims)
        self.V = len(H)
        return H

    def initailize_B_multiview(self, m, H):
        B = []
        for v in range(self.V):
            if self.N > 30000:
                B_tmp = sampling_minikmeans(H, m)[0]
                B.append(B_tmp)
            else:
                B_tmp = sampling_kmeans(H, m)[0]
                B.append(B_tmp)
        return B

    def showData(self):
        print("------------------------------------------------------------------------------")
        print("{} is {} with {} view(s) and {} nodes".format(self.data_name, self.data_type, self.V, self.N))
        print("------------------------------------------------------------------------------")

    def train(self):
        global best_S, best_gnd, Time, best_embeddings, best_true_labels, best_predict_labels, affinity_matrix, Z
        H = self.graphFiltering()
        self.showData()
        ti = 0
        total_time = 0
        for m in self.ms:
            if m < self.C:
                continue
            B = self.initailize_B_multiview(m=m, H=H)
            for alpha in self.alphas:
                for beta in self.betas:
                    time_begin = time.time()
                    if dataname == "ACM":
                        Z, _, _ = self.algorithms[0](H, B, dataname, alpha=alpha, beta=beta, gamma=0.25, lambda_w=0.025)
                    elif dataname == "DBLP":
                        Z, _, _ = self.algorithms[0](H, B, dataname, alpha=alpha, beta=beta, gamma=0.3, lambda_w=0.3)
                    elif dataname == "AMAP":
                        Z, _, _ = self.algorithms[0](H, B, dataname, alpha=alpha, beta=beta, gamma=0.3, lambda_w=0.03)
                    elif dataname == "AMAC":
                        Z, _, _ = self.algorithms[0](H, B, dataname, alpha=alpha, beta=beta, gamma=0.3, lambda_w=0.03)
                    elif dataname == "YELP":
                        Z, _, _ = self.algorithms[0](H, B, dataname, alpha=alpha, beta=beta, gamma=0.3, lambda_w=0.03)
                    time_end = time.time()
                    Time = np.fabs(time_end - time_begin)
                    ti += 1
                    total_time += Time
                    acc, nmi, ari, f1, pur = evaluate_clustering(Z, self.gnd)
                    print(
                        "m: {0: <3} alpha: {1: <5} beta: {2: <5} ACC: {3:.4f} NMI: {4:.4f} ARI: {5:.4f} F1: {6:.4f} "
                        "PUR: {7:.4f} Time: {8:.4f}".format(
                            m, alpha, beta, acc, nmi, ari, f1, pur, total_time / ti  # 时间！
                        )
                    )
                    if acc > self.best_re["ACC"]:
                        self.best_re["ACC"] = acc
                        self.best_re["NMI"] = nmi
                        self.best_re["ARI"] = ari
                        self.best_re["F1"] = f1
                        self.best_re["PUR"] = pur
                        self.best_paras["alpha"] = alpha
                        self.best_paras["beta"] = beta
                        self.best_paras["m"] = m

        print(
            "maybe best: \nm: {0: <3} alpha: {1: <5} beta: {2: <5} ACC: {3:.4f} NMI: {4:.4f} ARI: {5:.4f} F1: {6:.4f} "
            "PUR: {7:.4f} Time: {8:.2f} ".format(
                self.best_paras["m"], self.best_paras["alpha"], self.best_paras["beta"], self.best_re["ACC"],
                self.best_re["NMI"], self.best_re["ARI"], self.best_re["F1"], self.best_re["PUR"], Time
            )
        )


if __name__ == "__main__":
    dataname = "ACM"
    res = TPAAR(dataname=dataname)
    res.train()
