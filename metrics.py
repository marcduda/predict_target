from sklearn.metrics import accuracy_score


def get_accuracy(y_true, y_pred):
    """
    :param y_true: true class of target
    :param y_pred: predicted class of target
    :return: accuracy score (between 0 and 1)
    """
    return accuracy_score(y_pred, y_true)
