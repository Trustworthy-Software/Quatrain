def evaluation_metrics(self, y_trues, y_pred_probs):
    fpr, tpr, thresholds = roc_curve(y_true=y_trues, y_score=y_pred_probs, pos_label=1)
    auc_ = auc(fpr, tpr)

    y_preds = [1 if p >= 0.5 else 0 for p in y_pred_probs]

    acc = accuracy_score(y_true=y_trues, y_pred=y_preds)
    prc = precision_score(y_true=y_trues, y_pred=y_preds)
    rc = recall_score(y_true=y_trues, y_pred=y_preds)
    f1 = 2 * prc * rc / (prc + rc)

    print('\n***------------***')
    print('Evaluating AUC, F1, +Recall, -Recall')
    print('Test data size: {}, Incorrect: {}, Correct: {}'.format(len(y_trues), y_trues.count(0), y_trues.count(1)))
    print('Accuracy: %f -- Precision: %f -- +Recall: %f -- F1: %f ' % (acc, prc, rc, f1))
    tn, fp, fn, tp = confusion_matrix(y_trues, y_preds).ravel()
    recall_p = tp / (tp + fn)
    recall_n = tn / (tn + fp)
    print('AUC: {:.3f}, +Recall: {:.3f}, -Recall: {:.3f}'.format(auc_, recall_p, recall_n))
    # return , auc_

    # print('AP: {}'.format(average_precision_score(y_trues, y_pred_probs)))
    return recall_p, recall_n, acc, prc, rc, f1, auc_
