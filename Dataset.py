import pandas as pd
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader

ver3 = 1
pmt_radius = 19433.975
image_size = 64
name = 'muon_500mev_1k_64_ver3_02'
is_noise = 0  # 是否加入顶点噪声
# name = 'test'


df = pd.read_feather(f'MuonData_v1.2.feather')

print('Data loaded')
print('Event_num:', len(df['ev_id'].unique()))

#time cut
df = df[(df['hit_ev_direct_cosTheta']<1)&(df['hit_ev_direct_cosTheta']>0.64)]

# 计算球面与直线的交点
def find_sphere_intersection(sphere_center, sphere_radius, point, direction):
    x0, y0, z0 = sphere_center
    r = sphere_radius
    x1, y1, z1 = point
    dx, dy, dz = direction
    a = dx**2 + dy**2 + dz**2
    b = 2 * (dx * (x1 - x0) + dy * (y1 - y0) + dz * (z1 - z0))
    c = (x1 - x0)**2 + (y1 - y0)**2 + (z1 - z0)**2 - r**2
    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return None 
    t1 = (-b + np.sqrt(discriminant)) / (2 * a)
    t2 = (-b - np.sqrt(discriminant)) / (2 * a)
    t = max(t1, t2)
    if t < 0:
        return None  
    intersection = (x1 + t * dx, y1 + t * dy, z1 + t * dz)
    return intersection

# 墨卡托投影方法
def mercator_projection(positions):
    x = np.arctan2(positions[:, 1], positions[:, 0])
    y = np.arcsinh(positions[:, 2] / np.linalg.norm(positions[:, :2], axis=1))
    return np.vstack((x, y)).T

#逆投影
def inverse_mercator_projection(positions_2d, radius):
    lon = positions_2d[:, 0]
    lat = np.arcsin(np.tanh(positions_2d[:, 1]))
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)
    return np.vstack((x, y, z)).T

#逆解经纬度
def pixel_to_latlon(cherenkov_center_x, cherenkov_center_y, image_size):
    lon = cherenkov_center_x / image_size * 2 * np.pi - np.pi
    lat = cherenkov_center_y / image_size * np.pi - np.pi / 2
    return lon, lat

# 自定义Dataset
class CherenkovDataset(Dataset):
    def __init__(self, df):
        self.image_size = image_size
        self.df = df
        self.pmt_radius = 19433.975  # 直接设置球面半径
        self.x_data, self.y_data, self.vertex_positions, self.momentum = self.prepare_data()

    def prepare_data(self):
        event_ids = self.df['ev_id'].unique()
        x_data_list = []
        y_data_list = []
        vertex_positions_list = []
        momentum_list = []

        for event_id in event_ids:
            event_data = self.df[self.df['ev_id'] == event_id]
            pmt_positions = event_data[['pmt_x', 'pmt_y', 'pmt_z']].values
            vertex_position = event_data[['ev_x', 'ev_y', 'ev_z']].values[0]
            momentum = event_data[['ev_px', 'ev_py', 'ev_pz']].values[0]
            pmt_time = event_data['hit_time'].values
            hit_time_notof = event_data['hit_time_notof'].values
            order = event_data['hit_time'].values
            
            #为真实顶点加上1/vertex_position)

            # 墨卡托投影
            pmt_positions_2d = mercator_projection(pmt_positions)
            distances = np.linalg.norm(pmt_positions - vertex_position, axis=1)

            x_data = np.zeros((5, self.image_size, self.image_size))
            for i, pos in enumerate(pmt_positions_2d):
                x = int((pos[0] + np.pi) / (2 * np.pi) * self.image_size)
                y = int((pos[1] + np.pi / 2) / np.pi * self.image_size)
                # 确保坐标在有效范围内
                if 0 <= x < self.image_size and 0 <= y < self.image_size:
                    x_data[0, x, y] = 1
                    x_data[1, x, y] = distances[i]
                    x_data[2, x, y] = pmt_time[i]
                    x_data[3, x, y] = hit_time_notof[i]
                    x_data[4, x, y] = order[i]
                    # x_data[3, x, y] += 1 # 记录击中次数

            # Cherenkov环的中心坐标作为标签
            cherenkov_center = find_sphere_intersection((0, 0, 0), self.pmt_radius, vertex_position.tolist(), momentum.tolist())
            if cherenkov_center is not None:
                cherenkov_center_2d = mercator_projection(np.array([cherenkov_center]))[0]
                cherenkov_center_x = int((cherenkov_center_2d[0] + np.pi) / (2 * np.pi) * self.image_size) #将切伦科夫环中心转化为像素坐标
                cherenkov_center_y = int((cherenkov_center_2d[1] + np.pi / 2) / np.pi * self.image_size)
                cherenkov_center_2d = np.array([cherenkov_center_x, cherenkov_center_y])    
                x_data_list.append(x_data)
                y_data_list.append(cherenkov_center_2d)
                vertex_positions_list.append(vertex_position)
                momentum_list.append(momentum)
        return np.array(x_data_list), np.array(y_data_list), np.array(vertex_positions_list), np.array(momentum_list)
    
    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], (self.y_data[idx], self.vertex_positions[idx], self.momentum[idx])
print('Dataset creating')
# 创建Dataset和DataLoader
dataset = CherenkovDataset(df)
print('Dataset created')

print('x_data:', dataset.x_data.shape)
print('y_data:', dataset.y_data.shape)
print('vertex_positions:', dataset.vertex_positions.shape)


with open('test_dataset_{}.pkl'.format(name), 'wb') as f:
    pickle.dump(dataset, f)
print('Dataset saved')