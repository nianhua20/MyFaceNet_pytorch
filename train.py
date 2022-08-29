import torch

from nets.facenet import facenet
from utils.facenet_training import get_num_classes, triplet_loss

if __name__ == "__main__":
    Cuda = True

    annotation_path = "cls_train.txt"

    input_shape = [160, 160, 3]

    model_path = ""
    pretrained = False

    # ------------------------------------------------------------------#
    #   是否开启LFW评估
    # ------------------------------------------------------------------#
    lfw_eval_flag = True
    # ------------------------------------------------------------------#
    #   LFW评估数据集的文件路径和对应的txt文件
    # ------------------------------------------------------------------#
    lfw_dir_path = "lfw"
    lfw_pairs_path = "model_data/lfw_pair.txt"

    batch_size = 60
    epochs = 100
    Init_Epoch = 0

    # ------------------------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    # ------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    # ------------------------------------------------------------------#
    Init_lr = 1e-3
    Min_lr = Init_lr * 0.01
    # ------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=1e-3
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #                   adam会导致weight_decay错误，使用adam时建议设置为0。
    # ------------------------------------------------------------------#
    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 0
    # ------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有step、cos
    # ------------------------------------------------------------------#
    lr_decay_type = "cos"
    # ------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值，默认每个世代都保存
    # ------------------------------------------------------------------#
    save_period = 1
    # ------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    # ------------------------------------------------------------------#
    save_dir = 'logs'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes = get_num_classes()

    model = facenet(num_classes=num_classes)

    loss = triplet_loss()

    model_train = model.train().cuda()

    if batch_size % 3 != 0:
        raise ValueError("Batch Size must be the mutiple of 3.")

    nbs = 64
    lr_limit_max = 1e-1
    lr_limit_min = 5e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)




