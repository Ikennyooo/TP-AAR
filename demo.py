from data_loader import datasets
from clustering import Efficient_multi_view_clustering_with_weights
from metrics import evaluate_clustering

import pickle


class TPAAR:
    def __init__(self, dataname="Citeseer"):

        self.gnd = None
        self.datasets = datasets

        self.data_name = dataname
        self.data_type = 'graph data'

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
        if self.data_name in ["ACM", "DBLP", "YELP"]:
            X, As, Drs, gnd = self.datasets["multi-relational"](self.data_name)
            self.data_type = 'multi-relational ' + self.data_type
        elif self.data_name in ["AMAP", "AMAC"]:
            X, As, Drs, gnd = self.datasets["multi-attribute"](self.data_name)
        else:
            print("No such dataset!!!!!")
            return
        self.gnd = gnd
        return X, As, Drs, gnd

    def showData(self):
        print("------------------------------------------------------------------------------")
        print("{} is {} ".format(self.data_name, self.data_type))
        print("------------------------------------------------------------------------------")

    def train(self):
        self.loadData()
        self.showData()
        with open('ACM_Z.pkl', 'rb') as f:
            Z = pickle.load(f)
        acc, nmi, ari, f1, pur = evaluate_clustering(Z, self.gnd)
        print(
            "ACC: {:.4f} NMI: {:.4f} ARI: {:.4f} F1: {:.4f} "
            "PUR: {:.4f}".format(
                acc, nmi, ari, f1, pur
            )
        )


if __name__ == "__main__":
    dataname = "ACM"
    res = TPAAR(dataname=dataname)
    res.train()
