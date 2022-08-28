from torch import nn


class Facenet(nn.Module):
    def __init__(self, backbone="mobilenet", mode="train", pretrain=False):
        super(Facenet, self).__init__()
