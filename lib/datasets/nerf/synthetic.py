import torch.utils.data as data
import numpy as np
import os
from lib.utils import data_utils
from lib.config import cfg
from torchvision import transforms as T
import imageio.v2 as imageio
import json
import cv2


class Dataset(data.Dataset):
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        data_root, split, scene = kwargs['data_root'], kwargs['split'], cfg.scene
        view = kwargs['view']
        self.input_ratio = kwargs['input_ratio']
        self.data_root = os.path.join(data_root, scene)
        self.split = split
        self.batch_size = cfg.task_arg.N_rays

        # read image
        image_paths = []
        poses = []
        imgs = []
        json_info = json.load(open(os.path.join(self.data_root, 'transforms_{}.json'.format(self.split))))
        for frame in json_info['frames']:
            fname = os.path.join(self.data_root, frame['file_path'][2:] + '.png')
            image_paths.append(os.path.join(self.data_root, frame['file_path'][2:] + '.png'))
            poses.append(np.array(frame['transfom_matrix']))
            imgs.append(imageio.imread(fname))
            imgs = (np.array(imgs) / 255.).astype(np.float32) # 像素值标准化

        # img = imageio.imread(image_paths[view])/255.
        if cfg.task_arg.white_bkgd:
            imgs = imgs[..., :3] * imgs[..., -1:] + (1 - imgs[..., -1:]) # 处理白色背景
        else:
            imgs = imgs[..., :3]

        if self.input_ratio != 1.:
            imgs = cv2.resize(imgs, None, fx=self.input_ratio, fy=self.input_ratio, interpolation=cv2.INTER_AREA)
        # set image
        self.imgs = np.array(imgs).astype(np.float32)
        self.poses = np.array(poses).astype(np.float32) # camera poses 外参矩阵 (n,4,4)
        H, W = imgs[0].shape[:2]
        camera_angle_x = float(json_info['camera_angle_x'])
        focal = .5 * W / np.tan(.5 * camera_angle_x)  # 计算焦距
        self.hwf = [int(H), int(W), focal]
        K = np.array([
            [focal,0,0.5*int(W)],
            [0,focal,0.5*int(H)],
            [0,0,1]
        ]) # 内参矩阵
        self.near = 2.
        self.far = 6.

        # set X Y
        X, Y = np.meshgrid(np.arange(W), np.arange(H)) # 每个坐标点都会作为一个输入
        dirs = np.stack([(X-K[0][2])/K[0][0], -(Y-K[1][2])/K[1][1], -np.ones_like(X)], -1)
        rays = []
        for p in poses[:,:3,:4]:
            rays_d = np.array([p.dot(dir) for dir in dirs])
            rays_o = np.broadcast_to(p[:3,-1], np.shape((rays_d))) # 平移矩阵
            rays.append(np.stack([rays_o,rays_d], 0))

        rays = np.array(rays) # (400, 2, 800, 800, 3)
        rays_rgb = np.concatenate([rays, imgs[:None]],1) # N,ro+rd+rgb,H,W,3
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4]) # N,H,W,3,3
        rays_rgb = rays_rgb.reshape([-1,3,3]) # N*H*W,3,3
        rays_rgb = rays_rgb.astype(np.float32)
        self.rays_rgb = rays_rgb


    def __getitem__(self, index):
        if self.split == 'train':
            ids = np.random.choice(self.rays_rgb.shape[0], self.batch_size, replace=False)
            rays_rgb = self.rays_rgb[ids]
        else:
            rays_rgb = self.rays_rgb
        ret = {'rays': rays_rgb[:, :2], 'rgb': rays_rgb[:, 2]} # input and output. they will be
        # sent to
        # cuda
        ret.update({'meta': {'H': self.imgs.shape[0], 'W': self.imgs.shape[1],
                             'near': self.near, 'far': self.far}}) # meta means
        # no
        # need
        # to send to
        # cuda
        return ret

    def __len__(self):
        # we only fit 1 images, so we return 1
        return int(self.rays_rgb.shape[0] / self.batch_size)
