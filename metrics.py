from sklearn.metrics import accuracy_score


def get_accuracy(y_true, y_pred):
    return accuracy_score(y_pred, y_true)
