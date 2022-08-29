import numpy as np
import torch


def get_num_classes(annotation_path):
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
    labels = []
    for path in lines:
        path_split = path.split(";")
        labels.append(int(path_split[0]))
    num_classes = np.max(labels) + 1
    return num_classes

def triplet_loss(alpha = 0.2):
    def _triplet_loss(y_pred, batch_size):
        anchor, positive, negative = y_pred[:int(batch_size)], y_pred[int(batch_size):2*int(batch_size)], y_pred[2*int(batch_size):]

        pos_dist = torch.sqrt(torch.sum(torch.pow(anchor - positive, 2), axis=-1))
        neg_dist = torch.sqrt(torch.sum(torch.pow(anchor - negative, 2), axis=-1))

        keep_all = (neg_dist - pos_dist < alpha).cpu().cuda().flatten()
        hard_triplest = np.where(keep_all==1)

        pos_dist = pos_dist[hard_triplest]
        neg_dist = neg_dist[hard_triplest]

        basic_loss = pos_dist - neg_dist + alpha
        loss = torch.sum(basic_loss) / torch.max(torch.tensor(1), torch.tensor(len((hard_triplest[0]))))
        return loss
    return _triplet_loss