import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from lib.networks.encoding import get_encoder
from lib.config import cfg

class Network(nn.Module):
    def __init__(self,):
        super(Network, self).__init__()
        net_cfg = cfg.network
        self.xyz_encoder, self.input_ch = get_encoder(net_cfg.xyz_encoder)
        self.dir_encoder, self.input_view_ch = get_encoder(net_cfg.dir_encoder)
        self.skips = [4]
        D, W = net_cfg.D, net_cfg.W
        self.backbone_layer = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips
             else nn.Linear(W+self.input_ch) for i in range(D-1)])
        self.output_alpha = nn.Sequential(
                nn.Linear(W, 1),
                nn.ReLU()
                )
        self.output_rgb = nn.ModuleList([
            nn.Linear(W, W),
            nn.Linear(self.input_view_ch + W, W // 2),
            nn.Linear(W // 2, 3),
            nn.Sigmoid()]
                )
    def render(self, rays, batch):
        chunk = cfg.task_arg.chunk_size
        N_samples = cfg.task_arg.N_samples
        N_importance =  cfg.task_arg.N_importance
        N_rays = rays.shape[0]
        rays_o = rays[:, 0:3]
        rays_d = rays[:, 3:6]
        views = rays[:, -3:]
        bounds = torch.reshape(rays[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]
        t_vals = torch.linspace(0., 1., steps=N_samples)
        z_vals = near * (1.-t_vals) + far * t_vals
        z_vals = z_vals.expand([N_rays, N_samples]) # 每条光线上的N个采样点
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:, None]
        # N_rays N_samples 3 获得采样点的坐标
        raw = self.getRaw(pts, views, chunk)
        rgb_map, weights = self.raw2rgb(raw, z_vals, rays_d)
        if N_importance > 0:
            weights0 = weights + 1e-5  # prevent nans
            pdf = weights0 / torch.sum(weights, -1, keepdim=True)
            cdf = torch.cumsum(pdf, -1)  # 猜测cum的意思是该元素和所有在该元素之前的元素进行某种运算
            cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))
            # N_rays, N_samples
            u = torch.rand(list(cdf.shape[:-1]) + [N_importance])
            u = u.contiguous()
            inds = torch.searchsorted(cdf, u, right=True)  # 寻找左边的index
            below = torch.max(torch.zeros_like(inds - 1), inds - 1)
            above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
            inds_g = torch.stack([below, above], -1)  #
            matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
            cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
            bins_g = torch.gather(weights0.unsqueeze(1).expand(matched_shape), 2, inds_g)
            denom = (cdf_g[..., 1] - cdf_g[..., 0])
            denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
            t = (u - cdf_g[..., 0]) / denom
            samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
            z_samples = samples.detach()
            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :,None]
            raw0 = self.getRaw(pts, views, chunk)
            rgb_map, weights = self.raw2rgb(raw, z_vals, rays_d)
            # [N_rays, N_samples + N_importance, 3]
        return {'rgb': rgb_map}

    def raw2rgb(self, raw, z_vals, rays_d):
        dist = z_vals[..., 1:] - z_vals[..., :-1]
        dist = torch.cat([dist, torch.Tensor[1e10].expand(dist[..., :1].shape)], -1)
        dist = dist * torch.norm(rays_d[..., None, :], dim=-1)  # N_samples
        rgb = raw[..., :3]  # N_rays,N_samples,3
        alpha = 1 - torch.exp(-raw[..., -1] * dist)
        weights = alpha * torch.cumprod(torch.cat(
            [torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:, :-1]
        # N_rays,N_samples,1 * N_rays,N_samples,3
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # N_rays,3
        return rgb_map, weights
    def getRaw(self, input_pts, input_views, chunk):
        input_views = input_views[:, None].expand(input_pts.shape)  # 把视角和坐标的形状统一，因为多个坐标对应一个视角
        input_pts = torch.reshape(input_pts, [-1, input_pts.shape[-1]])
        input_views = torch.reshape(input_views, [-1, input_views.shape[-1]])
        input_pts = self.xyz_encoder(input_pts)  # N_rays, N_samples, 3
        input_views = self.dir_encoder(input_views)
        output = torch.cat([self.netfwork_fn(input_pts[i:i+chunk], input_views[i:i+chunk])
                         for i in range(0, input_pts.shape[0],chunk)], 0)
        output = torch.reshape(output, list(input_pts.shape[:-1]) + [output.shape[-1]])
        return output # N_rays, N_samples, 4
    def netfwork_fn(self, input_pts, input_views):
        h = input_pts
        for i, l in enumerate(self.backbone_layer):
            h = self.backbone_layer[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        alpha = self.output_alpha(h)
        for i, l in enumerate(self.output_alpha):
            h = self.output_alpha[i](h)
            if i == 1:
                h = torch.cat([h, input_views], -1)
        rgb = h
        outputs = torch.cat([rgb, alpha], -1)
        return outputs
    def batchify(self, rays, batch):
        all_ret = {}
        chunk = cfg.task_arg.chunk_size
        for i in range(0, rays.shape[0], chunk):
            ret = self.render(rays[i:i + chunk], batch)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], dim=0) for k in all_ret}
        return all_ret

    def forward(self, batch):
        # 预处理，数据转成torch，拼成rays
        rays_o = batch['ret'][:, 0] # rgb_o
        rays_o = torch.reshape(rays_o, [-1, rays_o.shape[-1]])
        rays_d = batch['ret'][:, 1] # rgb_d
        rays_d = torch.reshape(rays_d, [-1, rays_d.shape[-1]])
        viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()
        input_views = self.dir_encoder(viewdirs)
        near, far= batch['near'] * torch.ones_like(input_views[..., :1]), batch['far'] * torch.ones_like(input_views[..., :1])
        rays = torch.cat([rays_o, rays_d, near, far, viewdirs], -1)
        # (N_rand,3+3+1+1+3)

        ret = self.batchify(rays, batch)
        # input_pts, input_views = torch.split(x, [self.input_ch, self.input_view_ch], dim=-1)
