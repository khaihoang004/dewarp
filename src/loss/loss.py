import torch
from pytorch_msssim import ms_ssim

def flow_loss(pred_bm, gt_bm):
    return torch.mean((pred_bm - gt_bm)**2)

def recon_loss(unwarped, gt):
    return 1 - ms_ssim(unwarped, gt, data_range=1.0)

def combined_loss(pred_bm, gt_bm, unwarped, gt_img, cfg):
    return (cfg.loss.flow_weight * flow_loss(pred_bm, gt_bm) +
            cfg.loss.recon_weight * recon_loss(unwarped, gt_img))