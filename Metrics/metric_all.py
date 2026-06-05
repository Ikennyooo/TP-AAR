from sklearn import metrics
import munkres
import numpy as np
from sklearn.metrics import accuracy_score

def purity_score(y_true, y_pred):


    """Purity score
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return accuracy_score(y_true, y_voted_labels)

class clustering_metrics( ) :
    def __init__(self, true_label, predict_label) :
        self.true_label = true_label
        self.pred_label = predict_label

    def clusteringAcc(self) :
        # best mapping between true_label and predict label
        l1 = list(set(self.true_label))
        numclass1 = len(l1)
        # print('gnd:',numclass1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        # print('prd:',numclass2)
        if numclass1 != numclass2 :
            print('Class Not equal, Error!!!!')
            return 0, 0, 0, 0, 0, 0, 0, 0

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1) :
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2) :
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)

        # match two clustering results by Munkres algorithm
        m = munkres.Munkres( )
        cost = cost.__neg__( ).tolist( )

        indexes = m.compute(cost)

        # get the match results
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1) :
            # correponding label in l2:
            c2 = l2[indexes[i][1]]

            # ai is the index with label==c2 in the pred_label list
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c

        acc = metrics.accuracy_score(self.true_label, new_predict)
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
        recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
        PUR = purity_score(self.true_label, new_predict)
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, PUR

    def evaluationClusterModelFromLabel(self) :
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, pur = self.clusteringAcc( )

        # print('ACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, NMI=%f, ADJ_RAND_SCORE=%f' % (acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, nmi, adjscore))
        #
        # fh = open('recoder.txt', 'a')
        #
        # fh.write('GAAMA=%f,k=%f,a=%f,ACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, NMI=%f, ADJ_RAND_SCORE=%f' % (a,b,c,acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, nmi, adjscore) )
        # fh.write('\r\n')
        # fh.flush()
        # fh.close()

        return acc, nmi, adjscore, f1_macro, pur
