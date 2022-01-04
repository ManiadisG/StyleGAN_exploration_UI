import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlretrieve

sys.path.append(os.getcwd())
sys.path.append('./stylegan2')

import stylegan2.legacy as legacy
import stylegan2.dnnlib as dnnlib

url="https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"


class StyleGan():
    def __init__(self, weights_path=None, device=0):
        weights_path = weights_path or "./ffhq.pkl"
        if not os.path.isfile(weights_path):
            print("\nWeights not found in path: " + weights_path+"\nDownloading weights.")
            try:
                urlretrieve(url, weights_path)
            except:
                print("\nDownload failed.")
                exit()
        with dnnlib.util.open_url(weights_path) as fp:
            m_dict = legacy.load_network_pkl(fp)
        self.Generator = m_dict['G'].to(device)
        self.Generator.requires_grad_(False)
        self.Discriminator = m_dict['D'].to(device)
        self.Discriminator.requires_grad_(False)

        self.device = device
        self.label = torch.zeros([1, self.Generator.c_dim]).to(0)
        self.truncation_psi = 0.7
        self.noise_mode = 'const'  # 'const', 'random', 'none'
        self.latent = self.get_latent()

    @torch.no_grad()
    def generate_random_sample(self, plot=False, return_noise=False):
        img, ws = self.get_sample()
        if plot:
            plt.imshow(img[0].cpu().numpy())
        if return_noise:
            return img, ws
        else:
            return img

    def get_latent(self):
        return torch.from_numpy(np.random.randn(1, self.Generator.z_dim)).to(self.device)

    def get_projection(self, z=None):
        if z is None:
            z = self.get_latent()
        return self.Generator.mapping(z, self.label, truncation_psi=self.truncation_psi, truncation_cutoff=None)

    def get_sample(self, z=None, noise_mode=None):
        noise_mode = noise_mode or self.noise_mode
        ws = self.get_projection(z)
        img = self.Generator.synthesis(ws, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return img, ws

    def fix_latent_sample(self, reset_latent=False, noise_mode='const', plot=False, return_noise=False):
        if self.latent is None or reset_latent:
            self.latent = self.get_latent()
        img, ws = self.get_sample(self.latent, noise_mode)
        if plot:
            plt.imshow(img[0].cpu().numpy())
        if return_noise:
            return img, ws
        else:
            return img

    def reset_latent(self):
        self.latent = self.get_latent()
