import torch
import torch.nn as nn

class Kirito(nn.Module):
    def __init__(self, input_size=224, channa_num=3, neuron_num=128):
        super(Kirito, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(channa_num, 32, kernel_size=9, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer4 = nn.Sequential(  # 第四层卷积层
            nn.Conv2d(128, 256, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer5 = nn.Sequential(  # 第五层卷积层
            nn.Conv2d(256, 512, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # 展平层
        self.flatten = nn.Flatten()
        
        # 计算全连接层的输入大小
        self.fc_input_size = self._get_fc_input_size(input_size, channa_num)
        self.fc1 = nn.Linear(self.fc_input_size, neuron_num)
        self.fc2 = nn.Linear(neuron_num, 2)  # 输出改为2，表示方位角和极角

    def _get_fc_input_size(self, input_size, channa_num):
        with torch.no_grad():
            x = torch.zeros(1, channa_num, input_size, input_size)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)  # 加入第四层卷积的向前传递
            x = self.layer5(x)  # 加入第五层卷积的向前传递
            return self.flatten(x).shape[1]  # 计算展平后的全连接层输入大小

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)  # 加入第四层卷积的向前传递
        out = self.layer5(out)  # 加入第五层卷积的向前传递
        out = self.flatten(out)  # 展平层
        out = self.fc1(out)
        out = self.fc2(out)
        return out