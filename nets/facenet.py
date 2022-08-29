from torch import nn

def conv_bn(input_channels, output_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, bias=False),
        nn.BatchNorm2d(output_channels),
        nn.ReLU()
    )

def conv_dw(input_channels, output_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=stride, bias=False),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(),

        nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(output_channels),
        nn.ReLU()
    )

class facenet(nn.Module):
    def _init_(self, dropout_score=0.5, mode="train", num_classes=None, embedding_size=128):
        super(facenet, self)._init_()
        self.stage1 = nn.Sequential(
            # h,w,3 -> h/2,w/2,32
            conv_bn(3, 32, stride=2),
            # h/2,w/2,32 -> h/2,w/2,64
            conv_dw(32, 64, stride=1),

            # h/2,w/2,64 -> h/4,w/4,128
            conv_dw(64, 128, stride=2),
            # h/4,w/4,128 -> h/4,w/4,128
            conv_dw(128, 128, stride=1),

            # h/4,w/4,128 -> h/8,w/8,256
            conv_dw(128, 256, stride=2),
            # h/8,w/8,256 -> h/8,w/8,256
            conv_dw(256, 256, stride=1),

            # h/8,w/8,256 -> h/16,w/16,512
            conv_dw(256, 512, stride=2)
        )
        self.stage2 = nn.Sequential(
            # h/16,w/16,512 -> h/16,w/16,512
            conv_dw(512, 512, stride=1),
            conv_dw(512, 512, stride=1),
            conv_dw(512, 512, stride=1),
            conv_dw(512, 512, stride=1),
            conv_dw(512, 512, stride=1)
        )
        self.stage3 = nn.Sequential(
            # h/16,w/16,512 -> h/32,w/32,1024
            conv_dw(512, 1024, stride=2),
            # h/32,w/32,1024 -> h/32,w/32,1024
            conv_dw(1024, 1024, stride=1),
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_score)
        self.bottleneck = nn.Linear(1024, embedding_size, bias=False)
        self.last_bn = nn.BatchNorm1d(embedding_size, eps=0.001, momentum=0.1, affine=True)
        if mode == "train":
            self.classifier = nn.Linear(embedding_size, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weights, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mode="predict"):
        if mode == "predict":
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.avg(x)
            x = self.view(x.size(0), -1)
            x = self.dropout(x)
            x = self.bottleneck(x)
            x = self.last_bn(x)
            x = nn.functional.normalize(x, p=2, dim=1)
            return x

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.bottleneck(x)
        before_normalize = self.last_bn(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        # 是对还未l2标准化的128d特征向量进行分类
        cls = self.classifier(before_normalize)
        return x, cls





