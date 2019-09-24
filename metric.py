from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class metric:
    def accuracy(self, true_y, pred_y):
        return accuracy_score(y_true=true_y, y_pred=pred_y)

    def f1(self, true_y, pred_y):
        return f1_score(y_true= true_y, y_pred=pred_y, average='micro')

    def precision(self, true_y, pred_y):
        return precision_score(y_true=true_y, y_pred=pred_y, average='micro')

    def recall(self, true_y, pred_y):
        return recall_score(y_true=true_y, y_pred=pred_y, average='micro')

