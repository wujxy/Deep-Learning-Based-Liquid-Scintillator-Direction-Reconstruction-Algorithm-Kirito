import numpy as np
from matplotlib import pyplot as plt
import pickle
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

from Kirito import Kirito
from Dataset import CherenkovDataset, inverse_mercator_projection, pixel_to_latlon



pmt_radius = 19433.975


ver3 =1
name = 'e_20mev_2w_64_ver3'
# name = 'test'
image_size = 64
model_name = 'CNN'

num_epochs = 200
channa_num = 5

#CNN
neuron_num = 128


# 找到正确的设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#data loading
with open('test_dataset_{}.pkl'.format(name), 'rb') as f:
    dataset = pickle.load(f)
    
# 分割训练集和测试集
test_size = 0.05
train_size = int((1 - test_size) * len(dataset))
test_size = len(dataset) - train_size

torch.manual_seed(42)
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

print(f"Train dataset length: {len(train_dataset)}")
print(f"Test dataset length: {len(test_dataset)}")

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#模型调用
input_size = image_size
model = Kirito(input_size=input_size, channa_num=channa_num,neuron_num=neuron_num).to(device)
print('model:', model_name)
criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_list = []
# 训练模型
for epoch in range(num_epochs):
    model.train()
    for i, (images, (labels, vertex_positions, momentum)) in enumerate(train_loader):
        images = torch.tensor(images, dtype=torch.float32).to(device)
        labels = torch.tensor(labels, dtype=torch.float32).to(device)
        vertex_positions = torch.tensor(vertex_positions, dtype=torch.float32).to(device)
        momentum = torch.tensor(momentum, dtype=torch.float32).to(device)

        # 检查张量形状
        assert images.shape[1] == channa_num, f"Expected 5 channels, but got {images.shape[1]}"

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels[:, :2])  

        # 反向传播和优化
        if model_name == 'ViT':
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    loss_list.append(loss.item())

# 评估模型并计算夹角
model.eval()
predictions = []
actuals = []
vertex_positions_list = []
momentum_list = []
with torch.no_grad():
    for images, (labels, vertex_positions, momentum) in test_loader:
        images = images.float().to(device)
        labels = labels.float().to(device)
        vertex_positions = vertex_positions.to(device)
        momentum = momentum.to(device)

        # 检查张量形状
        assert images.shape[1] == channa_num, f"Expected 5 channels, but got {images.shape[1]}"

        outputs = model(images)
        predictions.append(outputs.cpu().numpy())  # 转移到CPU以便后续处理
        actuals.append(labels.cpu().numpy())
        vertex_positions_list.append(vertex_positions.cpu().numpy())
        momentum_list.append(momentum.cpu().numpy())

#储存模型
torch.save(model.state_dict(), '{}_model.pth'.format(model_name))

# 将损失值保存到文件
with open('{}_losses.txt'.format(model_name), 'w') as f:
    for loss in loss_list:
        f.write(f"{loss:.4f}\n")

print("Loss values have been saved to 'epoch_losses.txt'")

# 转换为numpy数组
predictions = np.concatenate(predictions)
vertex_positions = np.concatenate(vertex_positions_list)

#输出为像素坐标，需要逆解经纬度
predictions = np.array([pixel_to_latlon(x, y, image_size) for x, y in predictions])


# 逆墨卡托投影将二维坐标转换为三维坐标
predicted_3d = inverse_mercator_projection(predictions,pmt_radius)

# 计算每个事件的方向向量
predicted_directions = predicted_3d - vertex_positions
actual_directions = np.concatenate(momentum_list)

pred_vectors = predicted_directions
act_vectors = actual_directions

# 计算夹角
dot_products = np.sum(pred_vectors * act_vectors, axis=1)
norms_pred = np.linalg.norm(pred_vectors, axis=1)
norms_act = np.linalg.norm(act_vectors, axis=1)
cosine_angles = dot_products / (norms_pred * norms_act)
cosine_angles = np.clip(cosine_angles, -1.0, 1.0)  # 避免浮点数误差


# 画出夹角分布图
plt.figure(figsize=(12, 6))
plt.hist(cosine_angles, bins=50, alpha=0.7, color='b')
plt.title('cos Distribution between Predicted and Actual Directions by {}'.format(model_name))
plt.xlabel('cos (degrees)')
plt.ylabel('Frequency')
plt.savefig('costheta_distribution_{}_{}.jpg'.format(name,model_name))
plt.show()

# 统计模型参数
summary(model, input_size=(channa_num, image_size, image_size))