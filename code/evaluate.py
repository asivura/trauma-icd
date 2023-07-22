import numpy as np
from sklearn import metrics

# Inspired by https://github.com/biplob1ly/TransICD/blob/f5274925bee3a53c6476c8d4cbf8859716db8e74/code/trainer.py#L96
def precision_at_k(true_labels, pred_probs):
    # num true labels in top k predictions / k
    ks = [1, 5, 8, 10, 15, 20]
    sorted_pred = np.argsort(pred_probs)[:, ::-1]
    output = []
    p5_scores = None
    for k in ks:
        topk = sorted_pred[:, :k]

        # get precision at k for each example
        vals = []
        for i, tk in enumerate(topk):
            if len(tk) > 0:
                num_true_in_top_k = true_labels[i, tk].sum()
                denom = len(tk)
                vals.append(num_true_in_top_k / float(denom))

        output.append(np.mean(vals))
        if k == 5:
            p5_scores = np.array(vals)
    return output, p5_scores

def recall_at_k(true_labels, pred_probs):
    # num true labels in top k predictions / k
    ks = [1, 5, 8, 10, 15, 20]
    sorted_pred = np.argsort(pred_probs)[:, ::-1]
    output = []
    r5_scores = None
    for k in ks:
        topk = sorted_pred[:, :k]
        # get precision at k for each example
        vals = []
        for i, tk in enumerate(topk):
            if len(tk) > 0:
                num_true_in_top_k = true_labels[i, tk].sum()
                denom = true_labels[i,:].sum()
                if denom != 0:
                  vals.append(num_true_in_top_k / float(denom))
                else:
                  continue

        output.append(np.mean(vals))
        if k == 5:
            r5_scores = np.array(vals)
    return output, r5_scores


def compute_metrics(targets, probs, threshold=None, prefix='', eval_args={}):
  probs = np.array(probs)
  targets = np.array(targets)
  
  if threshold is None:
    #choose threshold to maximize f1 macro
    cur_max = 0
    for t in np.linspace(0, 1, 100):
      t = round(t, 2)
      preds = (probs > t).astype(float)
      score = metrics.f1_score(targets, preds, average='weighted', zero_division=0)
      if score >= cur_max:
        threshold = t
        cur_max = score

  preds = (probs > threshold).astype(float)
  accuracy = metrics.accuracy_score(targets, preds)
  f1_score_micro = metrics.f1_score(targets, preds, average='micro', zero_division=0)
  f1_score_macro = metrics.f1_score(targets, preds, average='macro', zero_division=0)
  f1_score_samples = metrics.f1_score(targets, preds, average='samples', zero_division=0)
  f1_score_weighted = metrics.f1_score(targets, preds, average='weighted', zero_division=0)

  #to calculate auc scores we need to exclude labels which are not present in the dataset
  mask = targets.sum(axis=0) != 0
  auc_score_micro = metrics.roc_auc_score(targets[:, mask], probs[:, mask], average='micro')
  auc_score_macro = metrics.roc_auc_score(targets[:, mask], probs[:, mask], average='macro')
  auc_score_samples = metrics.roc_auc_score(targets[:, mask], probs[:, mask], average='samples')
  auc_score_weighted = metrics.roc_auc_score(targets[:, mask], probs[:, mask], average='weighted')

  precision_at_ks, p5_scores = precision_at_k(targets, probs)
  recall_at_ks, r5_scores = recall_at_k(targets, probs)

  if prefix != '':
      prefix += '_'

  res = {
      f'{prefix}threshold': threshold,
      f'{prefix}accuracy': accuracy,
      f'{prefix}f1_score_micro': f1_score_micro,
      f'{prefix}f1_score_macro': f1_score_macro,
      f'{prefix}f1_score_samples': f1_score_samples,
      f'{prefix}f1_score_weighted': f1_score_weighted,
      f'{prefix}auc_score_micro': auc_score_micro,
      f'{prefix}auc_score_macro': auc_score_macro,
      f'{prefix}auc_score_samples': auc_score_samples,
      f'{prefix}auc_score_weighted': auc_score_weighted,
      f'{prefix}targets_shape': targets.shape,
      f'{prefix}probs_shape': probs.shape,
      f'{prefix}_model_train_args': str(eval_args)
  }
  ks = [1, 5, 8, 10, 15, 20]
  
  for ks_name, ks_scores in zip(["precision", "recall"], [precision_at_ks, recall_at_ks]):
    for k, k_score in zip(ks, ks_scores):
      res[f'{prefix}{ks_name}@{k}'] = k_score
  
  return res