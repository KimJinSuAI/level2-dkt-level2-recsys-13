def average_ensemble(all_preds):
    return [ sum(preds) / len(all_preds) for preds in zip(*all_preds) ]

def weighted_ensemble(all_preds, weights):
    return [ sum([pred*weight for pred, weight in zip(preds, weights)]) / sum(weights) for preds in zip(*all_preds) ]