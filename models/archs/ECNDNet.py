import paddle
import paddle.nn as nn



class  ECNDNet(nn.Layer):
    def __init__(self, channels=1, num_of_layers=15):
        super(ECNDNet, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        groups =1 
        layers = []
        kernel_size1 = 1
        self.conv1_1 = nn.Sequential(nn.Conv2D(in_channels=channels,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias_attr=False),nn.BatchNorm2D(features),nn.ReLU())
        self.conv1_2 = nn.Sequential(nn.Conv2D(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias_attr=False,dilation=2),nn.BatchNorm2D(features),nn.ReLU())
        self.conv1_3 = nn.Sequential(nn.Conv2D(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias_attr=False),nn.BatchNorm2D(features),nn.ReLU())
        self.conv1_4 = nn.Sequential(nn.Conv2D(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias_attr=False),nn.BatchNorm2D(features),nn.ReLU())
        self.conv1_5 = nn.Sequential(nn.Conv2D(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias_attr=False,dilation=2),nn.BatchNorm2D(features),nn.ReLU())
        self.conv1_6 = nn.Sequential(nn.Conv2D(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias_attr=False),nn.BatchNorm2D(features),nn.ReLU())
        self.conv1_7 = nn.Sequential(nn.Conv2D(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias_attr=False),nn.BatchNorm2D(features),nn.ReLU())
        self.conv1_8 = nn.Sequential(nn.Conv2D(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias_attr=False),nn.BatchNorm2D(features),nn.ReLU())
        self.conv1_9 = nn.Sequential(nn.Conv2D(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias_attr=False,dilation=2),nn.BatchNorm2D(features),nn.ReLU())
        self.conv1_10 = nn.Sequential(nn.Conv2D(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias_attr=False),nn.BatchNorm2D(features),nn.ReLU())
        self.conv1_11 = nn.Sequential(nn.Conv2D(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias_attr=False),nn.BatchNorm2D(features),nn.ReLU())
        self.conv1_12 = nn.Sequential(nn.Conv2D(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias_attr=False,dilation=2),nn.BatchNorm2D(features),nn.ReLU())
        self.conv1_13 = nn.Sequential(nn.Conv2D(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias_attr=False),nn.BatchNorm2D(features),nn.ReLU())
        self.conv1_14 = nn.Sequential(nn.Conv2D(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias_attr=False),nn.BatchNorm2D(features),nn.ReLU())
        self.conv1_15 = nn.Sequential(nn.Conv2D(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias_attr=False),nn.BatchNorm2D(features),nn.ReLU())
        self.conv1_16 = nn.Sequential(nn.Conv2D(in_channels=features,out_channels=1,kernel_size=kernel_size,padding=1,groups=groups,bias_attr=False),nn.ReLU())
        self.conv3 = nn.Conv2D(in_channels=1,out_channels=1,kernel_size=kernel_size,stride=1,padding=1,groups=1,bias_attr=True)
        
        for name, sublayer in self.named_sublayers():
            if isinstance(sublayer, nn.Conv2D):
                # 初始化 Conv2D 层的权重
                fan_out = sublayer._kernel_size[0] * sublayer._kernel_size[1] * sublayer._out_channels
                sublayer.weight.set_value(paddle.normal(mean=0.0, std=(2.0 / fan_out) ** 0.5, shape=sublayer.weight.shape))
                
            if isinstance(sublayer, nn.BatchNorm2D):
                # 初始化 BatchNorm2D 层的权重
                sublayer.weight.set_value(paddle.normal(mean=0.0, std=(2.0 / fan_out) ** 0.5, shape=sublayer.weight.shape))
                
                # 对权重进行裁剪
                clip_b = 0.025
                w = sublayer.weight.shape[0]
                weights = sublayer.weight
                clipped_weights = paddle.clip(weights, min=-clip_b, max=clip_b)
                weights_positive = paddle.where(clipped_weights >= 0, paddle.maximum(clipped_weights, paddle.to_tensor(clip_b)), clipped_weights)
                final_weights = paddle.where(weights_positive < 0, paddle.minimum(weights_positive, paddle.to_tensor(-clip_b)), weights_positive)
                sublayer.weight.set_value(final_weights)

                # 设置 running_var
                sublayer._variance.set_value(paddle.full_like(sublayer._variance, 0.01))
        
    def forward(self, x):
        input = x 
        x1 = self.conv1_1(x)
        x1 = self.conv1_2(x1)
        x1 = self.conv1_3(x1)
        x1 = self.conv1_4(x1)
        x1 = self.conv1_5(x1)
        x1 = self.conv1_6(x1)
        x1 = self.conv1_7(x1)   
        x1t = self.conv1_8(x1)
        x1 = self.conv1_9(x1t)
        x1 = self.conv1_10(x1)
        x1 = self.conv1_11(x1)
        x1 = self.conv1_12(x1)
        x1 = self.conv1_13(x1)
        x1 = self.conv1_14(x1)
        x1 = self.conv1_15(x1)
        x1 = self.conv1_16(x1)
        out = self.conv3(x1)
        out1 = x - out
        return out1
