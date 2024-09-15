#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn
from torch.autograd import Variable
import itertools
import os
from . import network
from .base_model import BaseModel
import numpy as np

def mode_product(tensor, factor_matrix, mode):
    tensor_dims = tensor.size()
    reshaped_tensor = torch.moveaxis(tensor, mode, -1)

    mode_size = tensor_dims[mode]
    reshaped_tensor = reshaped_tensor.reshape(-1, mode_size)

    result = torch.matmul(reshaped_tensor, factor_matrix)

    result_dims = list(tensor_dims)
    result_dims[mode] = factor_matrix.size(1)
    result = result.view(result_dims)

    return result

def chain_mode_product(tensor, factor_matrices):
    for i in range(1,len(factor_matrices)+1):
        tensor = mode_product(tensor, factor_matrices[i-1], i)
    return tensor

class DTDNML(BaseModel):
    def name(self):
        return "DTDNML"

    @staticmethod
    def modify_commandline_options(parser, isTrain=True):

        parser.set_defaults(no_dropout=True)
        if isTrain:
            parser.add_argument(
                "--lambda_A", type=float, default=1.0, help="weight for PSF-SRF loss"
            )
            parser.add_argument(
                "--lambda_B", type=float, default=1.0, help="weight for spectral manifold"
            )
            parser.add_argument(
                "--lambda_C",
                type=float,
                default=1.0,
                help="weight for spatial manifold",
            )
            parser.add_argument(
                "--lambda_D",
                type=float,
                default=1.0,
                help="weight for LR-MSI constraints",
            )
            parser.add_argument("--num_theta", type=int, default=64)
            parser.add_argument("--avg_crite", type=str, default="No")
            parser.add_argument("--isCalSP", type=str, default="Yes")
        return parser

    def initialize(
        self, opt, hsi_channels, msi_channels, lrhsi_hei, lrhsi_wid, sp_matrix, sp_range
    ):
        BaseModel.initialize(self, opt)
        self.opt = opt
        self.visual_names = ["real_lhsi", "rec_lr_lr"]
        num_s = self.opt.num_theta
        ngf = 64
        
        code_tensor_scale = [500, 500, num_s]
        # code_tensor_scale = [512, 512, num_s]
        code_tensor_scale_lr = [int(lrhsi_hei), int(lrhsi_hei), num_s]
        H, W = int(lrhsi_hei * self.opt.scale_factor), int(
            lrhsi_wid * self.opt.scale_factor
        )

        # net getnerator
        self.hrmsi_feature = network.define_hrmsi_feature(
            in_channel=msi_channels, out_channel=num_s, gpu_ids=self.gpu_ids
        )
        self.lrhsi_feature = network.define_lrhsi_feature(
            in_channel=hsi_channels, out_channel=num_s, gpu_ids=self.gpu_ids
        )
        self.fearture_unet = network.define_feature_unet(
            out_channel=num_s, H=H, Hc=code_tensor_scale[0], W=W, Wc=code_tensor_scale[1], scale=self.opt.scale_factor, gpu_ids=self.gpu_ids
        )
        self.psf_2 = network.define_psf_2(scale=opt.scale_factor, gpu_ids=self.gpu_ids)
        # self.fearture_unet = network.define_feature_concat(in_c=num_s, scale=self.opt.scale_factor, psf=self.psf_2, H=H, Hc=code_tensor_scale[0], W=W, Wc=code_tensor_scale[1], gpu_ids=self.gpu_ids)

        if self.opt.concat == "No":
            self.lr_hsi_dict_wh = network.define_lrdict_wh(
                code_scale=code_tensor_scale_lr,
                hsi_scale_w=int(lrhsi_wid),
                hsi_scale_h=int(lrhsi_hei),
                gpu_ids=self.gpu_ids,
            )
            self.lr_hsi_dict_s = network.define_lrdict_s(
                code_scale=code_tensor_scale_lr,
                hsi_scale_s=hsi_channels,
                gpu_ids=self.gpu_ids,
            )
        else:
            self.lr_hsi_dict_wh = network.define_lrdict_wh(
                code_scale=code_tensor_scale,
                hsi_scale_w=int(lrhsi_wid),
                hsi_scale_h=int(lrhsi_hei),
                gpu_ids=self.gpu_ids,
            )
            self.lr_hsi_dict_s = network.define_lrdict_s(
                code_scale=code_tensor_scale,
                hsi_scale_s=hsi_channels,
                gpu_ids=self.gpu_ids,
            )
            self.lr_hsi_dict_wht = network.define_lrdict_wht(
                code_scale=code_tensor_scale,
                hsi_scale_w=int(lrhsi_wid),
                hsi_scale_h=int(lrhsi_hei),
                gpu_ids=self.gpu_ids,
            )
            self.lr_hsi_dict_st = network.define_lrdict_st(
                code_scale=code_tensor_scale,
                hsi_scale_s=hsi_channels,
                gpu_ids=self.gpu_ids,
            )
        self.hr_msi_dict_wh = network.define_hrdict_wh(
            code_scale=code_tensor_scale,
            msi_scale_w=int(opt.scale_factor * lrhsi_wid),
            msi_scale_h=int(opt.scale_factor * lrhsi_hei),
            gpu_ids=self.gpu_ids,
        )
        self.hr_msi_dict_s = network.define_hrdict_s(
            code_scale=code_tensor_scale, msi_scale_s=msi_channels, gpu_ids=self.gpu_ids
        )
        self.hr_msi_dict_wht = network.define_hrdict_wht(
            code_scale=code_tensor_scale,
            msi_scale_w=int(opt.scale_factor * lrhsi_wid),
            msi_scale_h=int(opt.scale_factor * lrhsi_hei),
            gpu_ids=self.gpu_ids,
        )
        self.hr_msi_dict_st = network.define_hrdict_st(
            code_scale=code_tensor_scale, msi_scale_s=msi_channels, gpu_ids=self.gpu_ids
        )

        
        self.srf = network.define_hr2msi(
            args=self.opt,
            hsi_channels=hsi_channels,
            msi_channels=msi_channels,
            sp_matrix=sp_matrix,
            sp_range=sp_range,
            gpu_ids=self.gpu_ids,
        )

        # LOSS
        if self.opt.avg_crite == "No":
            # self.L1loss = torch.nn.MSELoss(size_average=False).to(self.device)
            self.L1loss = torch.nn.L1Loss(size_average=False).to(self.device)
        else:
            # self.L1loss = torch.nn.MSELoss(size_average=True).to(self.device)
            self.L1loss = torch.nn.L1Loss(size_average=True).to(self.device)

        self.model_names = [
            "hrmsi_feature",
            "lrhsi_feature",
            "fearture_unet",
            "lr_hsi_dict_s",
            "hr_msi_dict_wh",
            "psf_2",
            "srf",
        ]
        self.setup_optimizers()
        self.visual_corresponding_name = {}

    def setup_optimizers(self, lr=None):
        if lr == None:
            lr = self.opt.lr
        else:
            isinstance(lr, float)
            lr = lr
        self.optimizers = []
        self.optimizer_lrhsi_feature = torch.optim.Adam(
            itertools.chain(self.lrhsi_feature.parameters()), lr=lr * 0.5, betas=(0.9, 0.999)
        )
        self.optimizers.append(self.optimizer_lrhsi_feature)
        self.optimizer_hrmsi_feature = torch.optim.Adam(
            itertools.chain(self.hrmsi_feature.parameters()),
            lr=lr * 0.5,
            betas=(0.9, 0.999),
        )
        self.optimizers.append(self.optimizer_hrmsi_feature)
        self.optimizer_fearture_unet = torch.optim.Adam(
            itertools.chain(self.fearture_unet.parameters()),
            lr=lr * 0.8,
            betas=(0.9, 0.999),
            weight_decay=5e-4,
        )
        self.optimizers.append(self.optimizer_fearture_unet)

        self.optimizer_lr_hsi_dict_wh = torch.optim.Adam(
            itertools.chain(self.lr_hsi_dict_wh.parameters()),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=5e-4,
        )
        self.optimizers.append(self.optimizer_lr_hsi_dict_wh)
        self.optimizer_lr_hsi_dict_s = torch.optim.Adam(
            itertools.chain(self.lr_hsi_dict_s.parameters()),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=5e-4,
        )
        self.optimizers.append(self.optimizer_lr_hsi_dict_s)
        self.optimizer_hr_msi_dict_wh = torch.optim.Adam(
            itertools.chain(self.hr_msi_dict_wh.parameters()),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=5e-4,
        )
        self.optimizers.append(self.optimizer_hr_msi_dict_wh)
        self.optimizer_hr_msi_dict_s = torch.optim.Adam(
            itertools.chain(self.hr_msi_dict_s.parameters()),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=5e-4,
        )
        self.optimizers.append(self.optimizer_hr_msi_dict_s)

        self.optimizer_psf_2 = torch.optim.Adam(
            itertools.chain(self.psf_2.parameters()), lr=lr * 0.2, betas=(0.9, 0.999)
        )
        self.optimizers.append(self.optimizer_psf_2)
        if self.opt.isCalSP == "Yes":
            # 0.2
            self.optimizer_srf = torch.optim.Adam(
                itertools.chain(self.srf.parameters()), lr=lr * 0.2, betas=(0.9, 0.999)
            )
            self.optimizers.append(self.optimizer_srf)

    def set_input(self, input, isTrain=True):
        if isTrain:
            self.real_lhsi = Variable(input["lhsi"], requires_grad=True).to(self.device)
            self.real_hmsi = Variable(input["hmsi"], requires_grad=True).to(self.device)
            self.real_hhsi = Variable(input["hhsi"], requires_grad=True).to(self.device)
            self.manifold = Variable(input["sm"], requires_grad=True).to(self.device)
            self.manifold_sh = Variable(input["spm1"], requires_grad=True).to(self.device)
            self.manifold_sw = Variable(input["spm2"], requires_grad=True).to(self.device)
        else:
            with torch.no_grad():
                self.real_lhsi = Variable(input["lhsi"], requires_grad=False).to(
                    self.device
                )
                self.real_hmsi = Variable(input["hmsi"], requires_grad=False).to(
                    self.device
                )
                self.real_hhsi = Variable(input["hhsi"], requires_grad=False).to(
                    self.device
                )
                self.manifold = Variable(input["sm"], requires_grad=False).to(
                    self.device
                )
                # self.manifold_sh = Variable(input["spm1"], requires_grad=False).to(
                #     self.device
                # )
                # self.manifold_sw = Variable(input["spm2"], requires_grad=False).to(
                #     self.device
                # )

        self.image_name = input["name"]
        self.real_input = input

    def my_forward(self, epoch):
        # generate code tensor from LrHSI or HrMSI
        self.code_tensor_lr = self.lrhsi_feature(self.real_lhsi)
        self.code_tensor = self.hrmsi_feature(self.real_hmsi)
        
        self.code_tensor = self.fearture_unet(self.code_tensor, self.code_tensor_lr, epoch)
        # self.code_tensor = self.fearture_unet(self.code_tensor, self.code_tensor_lr)

        # 重建lr-hsi
        if self.opt.concat == "Yes":
            self.rec_lrhsi = self.lr_hsi_dict_s(self.lr_hsi_dict_wh(self.code_tensor))
        else:
            self.rec_lrhsi = self.lr_hsi_dict_s(
                self.lr_hsi_dict_wh(self.code_tensor_lr)
            )
        # 重建hr-msi
        self.rec_hrmsi = self.hr_msi_dict_s(self.hr_msi_dict_wh(self.code_tensor))
        # 重建hr-hsi
        self.rec_hrhsi = self.lr_hsi_dict_s(self.hr_msi_dict_wh(self.code_tensor))

        # 正交约束重构
        # self.rec_lrhsi_orthg = self.lr_hsi_dict_s(self.lr_hsi_dict_wh(self.lr_hsi_dict_st(self.lr_hsi_dict_wht(self.real_lhsi))))
        # self.rec_hrmsi_orthg = self.hr_msi_dict_s(self.hr_msi_dict_wh(self.hr_msi_dict_st(self.hr_msi_dict_wht(self.real_hmsi))))

        # hr-hsi 重构 lrhsi和hrmsi
        # self.rec_hsi_lrhsi = self.lr_hsi_dict_s(self.psf_2(self.hr_msi_dict_wh(self.code_tensor)))
        # self.rec_hsi_hrmsi = self.hr_msi_dict_wh(self.srf(self.lr_hsi_dict_s(self.code_tensor)))
        self.rec_hsi_lrhsi = self.psf_2(self.rec_hrhsi)
        self.rec_hsi_hrmsi = self.srf(self.rec_hrhsi)

        # 构建lr-msi
        self.rec_lrhsi_lrmsi = self.srf(self.real_lhsi)
        self.rec_hrmsi_lrmsi = self.psf_2(self.real_hmsi)

        self.visual_corresponding_name["real_lhsi"] = "rec_lrhsi"
        self.visual_corresponding_name["real_hmsi"] = "rec_hrmsi"
        self.visual_corresponding_name["real_hhsi"] = "rec_hrhsi"

    def my_backward_g_joint(self, epoch):
        # if epoch % 100 == 0:
        #     srf_list = self.srf.module.conv2d_list
        #     srf_weight_list = []
        #     for idx, l in enumerate(srf_list):
        #         srf_weight_list.append(l.weight.detach().squeeze().unsqueeze(-1))
        #     srf_weight = torch.cat(srf_weight_list,1).cpu().numpy()

        #     psf_weight = self.psf_2.module.net.weight.detach().squeeze().cpu().numpy()
        #     import scipy.io as sio
        #     sio.savemat(r"D:\\Dataset\\tensor_factorization\\CrossAttention\\checkpoints\\pavia_scale_8\\srf_psf\\srf_psf_{}.mat".format(epoch), {"srf_rec":srf_weight, "psf_rec":psf_weight})

        # Reconstruct loss
        # lrhsi-1
        self.loss_lr_pixelwise = self.L1loss(self.real_lhsi, self.rec_lrhsi)
        # self.loss_lr_orthg = self.L1loss(self.real_lhsi, self.rec_lrhsi_orthg)
        # hrmsi-1
        self.loss_msi_pixelwise = self.L1loss(self.real_hmsi, self.rec_hrmsi)
        # self.loss_msi_orthg = self.L1loss(self.real_hmsi, self.rec_hrmsi_orthg)
        # Rec loss
        self.loss_rec = self.loss_msi_pixelwise + self.loss_lr_pixelwise
        # self.loss_rec_orthg = self.loss_msi_orthg + self.loss_lr_orthg

        # Constraints loss
        # hrhsi to lrhsi: Learn the PSF
        self.loss_msi_ss_lr = self.L1loss(self.real_lhsi, self.rec_hsi_lrhsi)
        # hrhsi to hrmsi: Learn the SRF
        self.loss_msi_ss_msi = self.L1loss(self.real_hmsi, self.rec_hsi_hrmsi)
        # lrmsi: Lean the PSF and SRF
        self.loss_lrmsi_pixelwise = (
            self.L1loss(self.rec_lrhsi_lrmsi, self.rec_hrmsi_lrmsi) * self.opt.lambda_D
        )
        # Consistance loss
        self.loss_cons = (
            self.loss_msi_ss_lr + self.loss_msi_ss_msi + self.loss_lrmsi_pixelwise
        )
        # self.loss_cons = self.loss_lrmsi_pixelwise

        # if epoch > 3000:
        h = self.lr_hsi_dict_wh.module.conv_h.weight.permute(2, 3, 0, 1).squeeze()
        w = self.lr_hsi_dict_wh.module.conv_w.weight.permute(2, 3, 0, 1).squeeze()
        V = self.lr_hsi_dict_s.module.conv_s.weight.permute(2, 3, 0, 1).squeeze()

        self.rec_lrhsi_orthg = chain_mode_product(chain_mode_product(self.real_lhsi, [V,h,w]), [V.t(),h.t(),w.t()])
        self.loss_lr_orthg = self.L1loss(self.real_lhsi, self.rec_lrhsi_orthg)
        
        # print(V)
        # print(self.rec_hrhsi)
        self.loss_manifold = torch.trace(torch.matmul(torch.matmul(V.T, self.manifold.squeeze().to(device=self.device)), V))

        # Hz=torch.reshape(self.rec_lrhsi.permute(0,2,3,1).squeeze(), [self.rec_lrhsi.size(2)**2, self.rec_lrhsi.size(1)])
        # self.loss_manifold = torch.trace(torch.matmul(torch.matmul(Hz, self.manifold.squeeze().to(device=self.device)), Hz.T))

        H = self.hr_msi_dict_wh.module.conv_h.weight.permute(2, 3, 0, 1).squeeze()
        W = self.hr_msi_dict_wh.module.conv_w.weight.permute(2, 3, 0, 1).squeeze()
        v = self.hr_msi_dict_s.module.conv_s.weight.permute(2, 3, 0, 1).squeeze()

        self.rec_hrmsi_orthg = chain_mode_product(chain_mode_product(self.real_hmsi, [v,H,W]), [v.t(),H.t(),W.t()])
        self.loss_msi_orthg = self.L1loss(self.real_hmsi, self.rec_hrmsi_orthg)
        
        self.loss_rec_orthg = self.loss_msi_orthg + self.loss_lr_orthg

        # R = self.srf.module.srf_matrix
        # B = self.psf_2.module.net.weight.permute(2,3,0,1).squeeze()

        # module_list_parameters = []
        # for name, param in self.srf.module.conv2d_list.named_parameters():
        #     module_list_parameters.append(param.detach().cpu().numpy())
        
        # R = np.array(module_list_parameters)

        # if (epoch % 100) == 0:
        #     # save the feature map
        #     import scipy.io as sio
        #     # "R":R.detach().data.cpu().numpy(),
        #     # "B":B.detach().data.cpu().numpy(),
        #     save_mat = {"W":W.detach().data.cpu().float().numpy(),
        #                 "H":H.detach().data.cpu().float().numpy(),
        #                 "V":V.detach().data.cpu().float().numpy(),
        #                 "w":w.detach().data.cpu().float().numpy(),
        #                 "h":h.detach().data.cpu().float().numpy(),
        #                 "v":v.detach().data.cpu().float().numpy(),}
        #     sio.savemat(r"D:\\Dataset\\tensor_factorization\\CrossAttention\\checkpoints\\pavia_scale_8\\feature_map_20240117\\decoder_weight_{}.mat".format(epoch), 
        #                 save_mat)

        # Mz1 = torch.reshape(self.rec_hrmsi.permute(0,1,3,2).squeeze(), [self.rec_hrmsi.size(2)*self.rec_hrmsi.size(1), self.rec_hrmsi.size(2)])
        # Mz2 = torch.reshape(self.rec_hrmsi.permute(0,1,2,3).squeeze(), [self.rec_hrmsi.size(2)*self.rec_hrmsi.size(1), self.rec_hrmsi.size(2)])
        self.loss_manifold_spm1 = torch.trace(torch.matmul(torch.matmul(H.T, self.manifold_sh.squeeze().to(device=self.device)), H))
        self.loss_manifold_spm2 = torch.trace(torch.matmul(torch.matmul(W.T, self.manifold_sw.squeeze().to(device=self.device)), W))
        # self.loss_manifold_spm1 = torch.trace(torch.matmul(torch.matmul(Mz1, self.manifold_sh.squeeze().to(device=self.device)), Mz1.T))
        # self.loss_manifold_spm2 = torch.trace(torch.matmul(torch.matmul(Mz2, self.manifold_sw.squeeze().to(device=self.device)), Mz2.T))
        

        self.loss_manifold_spa = self.loss_manifold_spm1 + self.loss_manifold_spm2 # self.loss_manifold + self.loss_manifold_spm1 + self.loss_manifold_spm2
        
        self.loss_joint = (
            self.loss_rec
            + self.loss_rec_orthg
            + self.loss_cons * self.opt.lambda_A
            + self.loss_manifold_spa * self.opt.lambda_C
            + self.loss_manifold * self.opt.lambda_B
        ) 
        self.loss_joint.backward(retain_graph=False)

    def optimize_joint_parameters(self, epoch):
        if self.opt.concat == "No":
            self.loss_names = [
                "lr_pixelwise",
                "lr_sparse",
                "lr",
                "msi_pixelwise",
                "msi_sparse",
                "msi",
                "msi_ss_lr",
                "lrmsi_pixelwise",
            ]
        else:
            self.loss_names = [
                "lr_pixelwise",
                "rec",
                "rec_orthg",
                "msi_pixelwise",
                "manifold",
                "manifold_spa",
                "msi_ss_msi",
                "msi_ss_lr",
                "lrmsi_pixelwise",
                "joint",
            ]
        self.visual_names = [
            "real_lhsi",
            "rec_lrhsi",
            "real_hmsi",
            "rec_hrmsi",
            "real_hhsi",
            "rec_hrhsi",
        ]

        self.set_requires_grad(
            [
                self.lrhsi_feature,
                self.hrmsi_feature,
                self.fearture_unet,
                self.lr_hsi_dict_s,
                self.lr_hsi_dict_wh,
                self.hr_msi_dict_wh,
                self.hr_msi_dict_wh,
                self.psf_2,
                self.srf,
            ],
            True,
        )

        self.my_forward(epoch)

        self.optimizer_lrhsi_feature.zero_grad()
        self.optimizer_lr_hsi_dict_wh.zero_grad()
        self.optimizer_lr_hsi_dict_s.zero_grad()

        self.optimizer_hrmsi_feature.zero_grad()
        self.optimizer_hr_msi_dict_wh.zero_grad()
        self.optimizer_hr_msi_dict_s.zero_grad()

        self.optimizer_fearture_unet.zero_grad()

        self.optimizer_psf_2.zero_grad()

        if self.opt.isCalSP == "Yes":
            self.optimizer_srf.zero_grad()

        self.my_backward_g_joint(epoch)

        self.optimizer_lrhsi_feature.step()
        self.optimizer_lr_hsi_dict_wh.step()
        self.optimizer_lr_hsi_dict_s.step()

        self.optimizer_hrmsi_feature.step()
        self.optimizer_hr_msi_dict_wh.step()
        self.optimizer_hr_msi_dict_s.step()

        self.optimizer_fearture_unet.step()

        self.optimizer_psf_2.step()

        if self.opt.isCalSP == "Yes":
            self.optimizer_srf.step()

    def get_visual_corresponding_name(self):
        return self.visual_corresponding_name

    def cal_psnr(self):
        real_hsi = self.real_hhsi.data.cpu().float().numpy()[0]
        rec_hsi = self.rec_hrhsi.data.cpu().float().numpy()[0]
        return self.compute_psnr(real_hsi, rec_hsi)

    def compute_psnr(self, img1, img2):
        assert img1.ndim == 3 and img2.ndim == 3

        img_c, img_w, img_h = img1.shape
        ref = img1.reshape(img_c, -1)
        tar = img2.reshape(img_c, -1)
        msr = np.mean((ref - tar) ** 2, 1)
        max2 = np.max(ref) ** 2
        psnrall = 10 * np.log10(max2 / msr)
        out_mean = np.mean(psnrall)
        return out_mean

    def get_LR(self):
        lr = self.optimizers[0].param_groups[0]["lr"] * 2 * 1000
        return lr
