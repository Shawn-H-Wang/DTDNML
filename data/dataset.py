#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.utils.data as data
import torch
import os
import glob
import scipy.io as io
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans


class Dataset(data.Dataset):
    def __init__(self, args, sp_matrix, isTrain=True):
        super(Dataset, self).__init__()

        self.args = args
        self.sp_matrix = sp_matrix
        self.msi_channels = sp_matrix.shape[1]

        self.isTrain = isTrain

        default_datapath = os.getcwd()
        data_folder = os.path.join(default_datapath, args.data_name)
        if os.path.exists(data_folder):
            for root, dirs, files in os.walk(data_folder):
                if args.mat_name in files:
                    raise Exception("HSI data path does not exist!")
                else:
                    data_path = os.path.join(data_folder, args.mat_name + ".mat")
        else:
            return 0

        self.imgpath_list = sorted(glob.glob(data_path))
        self.img_list = []
        for i in range(len(self.imgpath_list)):
            self.img_list.append(io.loadmat(self.imgpath_list[i])["img"]) # [0:96,0:96,:]
            # self.img_list.append(io.loadmat(self.imgpath_list[i])["HSI"]) # CAVE

        "for single HSI"
        (_, _, self.hsi_channels) = self.img_list[0].shape

        "number of patches"
        patch_size = 64

        "generate simulated data"
        self.img_patch_list = []
        self.img_lr_list = []
        self.img_msi_list = []
        SNR = 25
        for i, img in enumerate(self.img_list):
            (h, w, c) = img.shape
            s = self.args.scale_factor
            "Ensure that the side length can be divisible"
            r_h, r_w = h % s, w % s
            img_patch = img[
                int(r_h / 2) : h - (r_h - int(r_h / 2)),
                int(r_w / 2) : w - (r_w - int(r_w / 2)),
                :,
            ]
            img_patch = (img_patch - np.min(img_patch)) / (np.max(img_patch) - np.min(img_patch))
            self.img_patch_list.append(img_patch)
            "LrHSI"
            img_lr = self.generate_LrHSI(img_patch, s)
            # sigmah = np.sqrt(np.sum(img_lr)**2 / (10 ** (SNR/10)) / (r_h*r_w*c))
            # img_lr_noise = img_lr + sigmah * np.random.normal(size=no.shape(img_lr))
            self.img_lr_list.append(img_lr)
            (self.lrhsi_height, self.lrhsi_width, p) = img_lr.shape
            self.spectral_manifold = self.generate_spectral_manifold(
                np.reshape(img_lr, [self.lrhsi_height * self.lrhsi_width, p]), k=15
            )
            "HrMSI"
            img_msi = self.generate_HrMSI(img_patch, self.sp_matrix)
            (self.msi_height, self.msi_width, p) = img_msi.shape
            img_msi = img_msi.transpose(1,2,0)
            self.spatial_manifold_1 = self.generate_spectral_manifold(
                np.reshape(img_msi, [self.msi_width * p, self.msi_height]), k=25
            )
            img_msi = img_msi.transpose(2,0,1)
            img_msi = img_msi.transpose(0,2,1)
            self.spatial_manifold_2 = self.generate_spectral_manifold(
                np.reshape(img_msi, [self.msi_height * p, self.msi_width]), k=25
            )
            img_msi = img_msi.transpose(0,2,1)
            # io.savemat(r"D:\\Dataset\\Hyperspectral Image Plot\\Pavia_manifold.mat", 
            #            {"spem":self.spectral_manifold,"spam1":self.spatial_manifold_1,"spam2":self.spatial_manifold_2,
            #             "spem_w":self.spectral_weights,"spam1_w":self.spatial_manifold_1_weights,"spam2_w":self.spatial_manifold_2_weights})
            # sigmam = np.sqrt(np.sum(img_msi)**2 / (10 ** (SNR/10)) / (h*w*p))
            # img_msi_noise = img_lr + sigmam * np.random.normal(size=no.shape(img_msi))
            self.img_msi_list.append(img_msi)
            # io.savemat(r"D:\\Dataset\\MIAE\\MIAE\\data\\pavia\\paviac_data_r80.mat", {'MSI':img_msi,'HSI':img_lr, 'REF':img_patch})
            # io.savemat(r"D:\\Dataset\\MIAE\\MIAE\\data\\pavia\\{}_r80.mat".format("img1"), {'MSI':img_msi,'HSI':img_lr, 'REF':img_patch})

    def generate_spatial_manifold(self, clustered_data, msi, sigma_D=900):
        m, n, _ = msi.shape
        # generate the spatial laplacian matrix of MSI
        spatial_weights = np.zeros([m * n, m * n])
        msi_2d = np.reshape(msi, [m*n, _])
        ks = kneighbors_graph(msi_2d, n_neighbors=40, mode="connectivity").toarray()
        for i in range(m):
            for j in range(n):
                pixel_ij = msi[i, j, :]
                idx_manifold_row = i * m + j
                label_ij = clustered_data[i, j]
                idxes = np.where(clustered_data == label_ij)
                for idn in range(idxes[0].shape[0]):
                    ni, nj = idxes[0][idn], idxes[1][idn]
                    idx_manifold_col = ni * m + nj
                    pixel_nij = msi[ni, nj, :]
                    weight = np.math.exp(-np.sum((pixel_ij - pixel_nij) ** 2) / sigma_D)
                    spatial_weights[idx_manifold_row, idx_manifold_col] = weight
        spatial_diag = np.diag(np.sum(spatial_weights, axis=1)) - spatial_weights
        return spatial_diag

    def generate_spectral_manifold(self, lrhsi_2d, k=15):
        """
        Generate manifold from three modes of a tensor
        K is for the number of neighbours
        """
        _, p = lrhsi_2d.shape
        # generate the spectral laplacian matrix of Lr HSI
        spectral_weights = np.zeros([p, p])
        ka = kneighbors_graph(lrhsi_2d.T, n_neighbors=k, mode="connectivity").toarray()
        sigma_S = 1000
        for i in range(p):
            idx = np.where(ka[i] == 1)[0]
            i_image = lrhsi_2d[:, i]
            for id in idx:
            # for j in range(p):
                id_image = lrhsi_2d[:, id]
                # id_image = lrhsi_2d[:, j]
                weight = np.math.exp(-np.sum((i_image - id_image) ** 2) / sigma_S)
                spectral_weights[i, id] = weight
                # spectral_weights[i, j] = weight
        spectral_diag = np.diag(np.sum(spectral_weights, axis=1)) - spectral_weights
        return spectral_diag # , spectral_weights

    def downsamplePSF(self, img, sigma, stride):
        def matlab_style_gauss2D(shape=(3, 3), sigma=1):
            m, n = [(ss - 1.0) / 2.0 for ss in shape]
            y, x = np.ogrid[-m : m + 1, -n : n + 1]
            h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
            h[h < np.finfo(h.dtype).eps * h.max()] = 0
            sumh = h.sum()
            if sumh != 0:
                h /= sumh
            return h

        # generate filter same with fspecial('gaussian') function
        h = matlab_style_gauss2D((stride, stride), sigma)
        if img.ndim == 3:
            img_w, img_h, img_c = img.shape
        elif img.ndim == 2:
            img_c = 1
            img_w, img_h = img.shape
            img = img.reshape((img_w, img_h, 1))
        # from scipy import signal
        from scipy.ndimage.filters import convolve
        out_img = np.zeros((img_w // stride, img_h // stride, img_c))
        for i in range(img_c):
            out = convolve(img[:, :, i], h)
            out_img[:, :, i] = out[::stride, ::stride]
        return out_img

    def generate_LrHSI(self, img, scale_factor):
        img_lr = self.downsamplePSF(img, sigma=self.args.sigma, stride=scale_factor)
        # np.random.seed(10)
        # SNRm = 30
        # w, h, c = np.shape(img_lr)
        # sigmam = np.math.sqrt(np.sum(img_lr) ** 2) / (10 ** (SNRm / 10)) / np.size(img_lr)
        # img_lr = img_lr.reshape(w*h, c)
        # img_lr = img_lr + sigmam * np.random.randn(np.shape(img_lr)[0], np.shape(img_lr)[1])
        # img_lr = img_lr.reshape(h, w, c)
        return img_lr

    def generate_HrMSI(self, img, sp_matrix):
        (h, w, c) = img.shape
        self.msi_channels = sp_matrix.shape[1]
        if sp_matrix.shape[0] == c:
            img_msi = np.dot(img.reshape(w * h, c), sp_matrix).reshape(
                h, w, sp_matrix.shape[1]
            )
        else:
            raise Exception("The shape of sp matrix doesnot match the image")
        # np.random.seed(10)
        # SNRm = 35
        # sigmam = np.math.sqrt(np.sum(img_msi) ** 2) / (10 ** (SNRm / 10)) / np.size(img_msi)
        # img_msi = img_msi.reshape(w*h, self.msi_channels)
        # img_msi = img_msi + sigmam * np.random.randn(np.shape(img_msi)[0], np.shape(img_msi)[1])
        # img_msi = img_msi.reshape(h, w, self.msi_channels)
        return img_msi

    def __getitem__(self, index):
        img_patch = self.img_patch_list[index]
        img_lr = self.img_lr_list[index]
        img_msi = self.img_msi_list[index]
        img_name = os.path.basename(self.imgpath_list[index]).split(".")[0]
        img_tensor_lr = torch.from_numpy(img_lr.transpose(2, 0, 1).copy()).float()
        img_tensor_hr = torch.from_numpy(img_patch.transpose(2, 0, 1).copy()).float()
        img_tensor_rgb = torch.from_numpy(img_msi.transpose(2, 0, 1).copy()).float()
        # spectral manifold
        img_tensor_sm = torch.from_numpy(self.spectral_manifold.copy()).float()
        # spatial manifold
        img_tensor_spm1 = torch.from_numpy(self.spatial_manifold_1.copy()).float()
        img_tensor_spm2 = torch.from_numpy(self.spatial_manifold_2.copy()).float()
        return {
            "lhsi": img_tensor_lr,
            "hmsi": img_tensor_rgb,
            "hhsi": img_tensor_hr,
            "sm": img_tensor_sm,
            "spm1": img_tensor_spm1,
            "spm2": img_tensor_spm2,
            "name": img_name,
        }
        # "spm1": img_tensor_spm1,
        # "spm2": img_tensor_spm2,

    def __len__(self):
        return len(self.imgpath_list)
