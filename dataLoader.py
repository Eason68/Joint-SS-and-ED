from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import os
from torch.utils.data import Dataset
from tqdm import tqdm
from dataPrepare import dataPrepare


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, coord, feat, label):
        for t in self.transforms:
            coord, feat, label = t(coord, feat, label)
        return coord, feat, label


class ToTensor(object):
    def __call__(self, coord, feat, label):
        coord = torch.from_numpy(coord)
        if not isinstance(coord, torch.FloatTensor):
            coord = coord.float()
        feat = torch.from_numpy(feat)
        if not isinstance(feat, torch.FloatTensor):
            feat = feat.float()
        label = torch.from_numpy(label)
        if not isinstance(label, torch.LongTensor):
            label = label.long()
        return coord, feat, label


class RandomRotate(object):
    def __init__(self, angle=[0, 0, 1]):
        self.angle = angle

    def __call__(self, coord, feat, label):
        angle_x = np.random.uniform(-self.angle[0], self.angle[0]) * np.pi
        angle_y = np.random.uniform(-self.angle[1], self.angle[1]) * np.pi
        angle_z = np.random.uniform(-self.angle[2], self.angle[2]) * np.pi
        cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
        cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
        cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
        R_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
        R_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
        R_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
        R = np.dot(R_z, np.dot(R_y, R_x))
        coord = np.dot(coord, np.transpose(R))
        return coord, feat, label


class RandomScale(object):
    def __init__(self, scale=[0.9, 1.1], anisotropic=False):
        self.scale = scale
        self.anisotropic = anisotropic

    def __call__(self, coord, feat, label):
        scale = np.random.uniform(self.scale[0], self.scale[1], 3 if self.anisotropic else 1)
        coord *= scale
        return coord, feat, label


class RandomShift(object):
    def __init__(self, shift=[0.2, 0.2, 0]):
        self.shift = shift

    def __call__(self, coord, feat, label):
        shift_x = np.random.uniform(-self.shift[0], self.shift[0])
        shift_y = np.random.uniform(-self.shift[1], self.shift[1])
        shift_z = np.random.uniform(-self.shift[2], self.shift[2])
        coord += [shift_x, shift_y, shift_z]
        return coord, feat, label


class RandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, coord, feat, label):
        if np.random.rand() < self.p:
            coord[:, 0] = -coord[:, 0]
        if np.random.rand() < self.p:
            coord[:, 1] = -coord[:, 1]
        return coord, feat, label


class RandomJitter(object):
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, coord, feat, label):
        assert (self.clip > 0)
        jitter = np.clip(self.sigma * np.random.randn(coord.shape[0], 3), -1 * self.clip, self.clip)
        coord += jitter
        return coord, feat, label


class ChromaticAutoContrast(object):
    def __init__(self, p=0.2, blend_factor=None):
        self.p = p
        self.blend_factor = blend_factor

    def __call__(self, coord, feat, label):
        if np.random.rand() < self.p:
            lo = np.min(feat, 0, keepdims=True)
            hi = np.max(feat, 0, keepdims=True)
            scale = 255 / (hi - lo)
            contrast_feat = (feat[:, :3] - lo) * scale
            blend_factor = np.random.rand() if self.blend_factor is None else self.blend_factor
            feat[:, :3] = (1 - blend_factor) * feat[:, :3] + blend_factor * contrast_feat
        return coord, feat, label


class ChromaticTranslation(object):
    def __init__(self, p=0.95, ratio=0.05):
        self.p = p
        self.ratio = ratio

    def __call__(self, coord, feat, label):
        if np.random.rand() < self.p:
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.ratio
            feat[:, :3] = np.clip(tr + feat[:, :3], 0, 255)
        return coord, feat, label


class ChromaticJitter(object):
    def __init__(self, p=0.95, std=0.005):
        self.p = p
        self.std = std

    def __call__(self, coord, feat, label):
        if np.random.rand() < self.p:
            noise = np.random.randn(feat.shape[0], 3)
            noise *= self.std * 255
            feat[:, :3] = np.clip(noise + feat[:, :3], 0, 255)
        return coord, feat, label


class HueSaturationTranslation(object):
    @staticmethod
    def rgb_to_hsv(rgb):
        # Translated from source of colorsys.rgb_to_hsv
        # r,g,b should be a numpy arrays with values between 0 and 255
        # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
        rgb = rgb.astype('float')
        hsv = np.zeros_like(rgb)
        # in case an RGBA array was passed, just copy the A channel
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        hsv[..., 0] = np.select([r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv

    @staticmethod
    def hsv_to_rgb(hsv):
        # Translated from source of colorsys.hsv_to_rgb
        # h,s should be a numpy arrays with values between 0.0 and 1.0
        # v should be a numpy array with values between 0.0 and 255.0
        # hsv_to_rgb returns an array of uints between 0 and 255.
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype('uint8')
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype('uint8')

    def __init__(self, hue_max=0.5, saturation_max=0.2):
        self.hue_max = hue_max
        self.saturation_max = saturation_max

    def __call__(self, coord, feat, label):
        # Assume feat[:, :3] is rgb
        hsv = HueSaturationTranslation.rgb_to_hsv(feat[:, :3])
        hue_val = (np.random.rand() - 0.5) * 2 * self.hue_max
        sat_ratio = 1 + (np.random.rand() - 0.5) * 2 * self.saturation_max
        hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
        hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)
        feat[:, :3] = np.clip(HueSaturationTranslation.hsv_to_rgb(hsv), 0, 255)
        return coord, feat, label


class RandomDropColor(object):
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, coord, feat, label):
        if np.random.rand() < self.p:
            feat[:, :3] = 0
            # feat[:, :3] = 127.5
        return coord, feat, label



class S3DISDataset(Dataset):
    def __init__(self, split="train", data_folder="s3dis_data", num_points=4096, test_area=5, block_size=1.0, sample_rate=1.0, transform=False):
        """
        DataLoader
        :param split: train or test
        :param data_folder: 数据集路径
        :param num_points: 每次采样的点数
        :param test_area: 在第几个Area测试
        :param block_size: 采样点的半径
        :param sample_rate: 采样率
        :param transform: 是否使用transform
        """
        super(S3DISDataset, self).__init__()
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 本文件所在文件夹
        DATA_PATH = os.path.join(BASE_DIR, "data", data_folder)
        if not os.path.exists(DATA_PATH):
            dataPrepare()
        self.num_points = num_points
        self.block_size = block_size
        self.transform = transform

        # 根据split生成训练集房间列表或测试级房间列表
        rooms = sorted(os.listdir(DATA_PATH))
        if split == 'train':
            rooms_split = [room for room in rooms if not 'Area_{}'.format(test_area) in room]
        else:
            rooms_split = [room for room in rooms if 'Area_{}'.format(test_area) in room]

        # 每个房间点集列表、点标签列表，每个房间点数列表
        self.room_points, self.room_labels = [], []
        num_points_all = []

        # 遍历每个房间
        print("Loading %s data:" % split)
        for room_name in tqdm(rooms_split, total=len(rooms_split)):

            # 根据路径读取数据
            room_path = os.path.join(DATA_PATH, room_name)
            if not os.path.exists(room_path):
                print(room_path + " not found!")
                continue
            room_data = np.load(room_path)  # xyzrgb l, N x 7
            points = room_data[:, :6]  # xyz rgb, N x 6
            labels = room_data[:, 6] # l, N x 1

            # 将读取到的数据存入列表中
            self.room_points.append(points)
            self.room_labels.append(labels)
            num_points_all.append(labels.size)

        # 每个房间被采样的概率, (1 x len(room_split), )
        sample_prob = num_points_all / np.sum(num_points_all)

        # 采样次数，即将所有点采一遍需要几次
        num_iter = int(np.sum(num_points_all) * sample_rate / num_points)

        # 按比例列出房间需要被采样的次数, for example: [0,0,1,1,1,2], 0房间2次，1房间3次，2房间1次
        room_idxs = []
        for index in range(len(rooms_split)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.room_idxs = np.array(room_idxs) # (1 x num_iter, )

        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def __getitem__(self, index):
        """
        重写由下标访问数据方法
        :param index: 房间下标, 0 ~ num_iter
        :return: selected_points采样点, num_points x 6
                 selected_labels采样点标签, num_points x 1
        """
        room_idx = self.room_idxs[index]
        points = self.room_points[room_idx]  # N x 6
        labels = self.room_labels[room_idx]  # N x 1
        N_points = points.shape[0]

        # 找一个在block_size范围内邻居数量多于1024的中心
        while (True):
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, self.block_size / 2.0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, self.block_size / 2.0]
            point_idxs = np.where((points[:, :3] >= block_min) & (points[:, :3] <= block_max))[0]
            if point_idxs.size > 1024:
                break

        # 对block_size范围内的点取num_points个点
        selected_point_idxs = np.random.choice(point_idxs, self.num_points, replace=(point_idxs.size < self.num_points))

        # 坐标归一化
        points = points[selected_point_idxs, :]  # num_points x 6
        labels = labels[selected_point_idxs]  # num_points x 1
        points[:, 0:3] -= center
        points[:, 3:6] /= 255.0


        # 数据增强
        if self.transform:
            transform = Compose([RandomScale([0.9, 1.1]), ChromaticAutoContrast(), ChromaticTranslation(),
                                 ChromaticJitter(), HueSaturationTranslation()])
            points[:, :3], points[:, 3:6], labels = transform(points[:, :3], points[:, 3:6], labels)

        # 转化为tensor
        points = torch.from_numpy(points).float()
        labels = torch.from_numpy(labels).long()

        return points, labels

    def __len__(self):
        """
        :return: 采样次数, num_iter
        """
        return len(self.room_idxs)


if __name__ == '__main__':
    data_folder = 's3dis_data'
    num_point, test_area, block_size, sample_rate = 4096, 5, 1.0, 1

    point_data = S3DISDataset(split='train', data_folder=data_folder, num_points=num_point, test_area=test_area,
                              block_size=block_size, sample_rate=sample_rate, transform=False)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(1)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(1)[1].shape)

    import torch, time, random

    manual_seed = 42
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)


    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)


    train_loader = torch.utils.data.DataLoader(point_data, batch_size=3, shuffle=True, num_workers=0, pin_memory=True,
                                               worker_init_fn=worker_init_fn)
    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i + 1, len(train_loader), time.time() - end))
            end = time.time()