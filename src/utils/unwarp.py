import torch.nn.functional as F

def unwarp(warped, bm):
    """ Unwarp the warped image using the backward mapping.

    Args:
        warped: (B, C, H, W) warped image
        bm: (B, 2, H, W) backward mapping

    Returns:
        unwarped: (B, C, H, W) unwarped image
    """
    _, _, h, w = warped.shape
    bm = F.interpolate(bm, size=(h, w), mode='bilinear', align_corners=True).permute(0, 2, 3, 1)  # Ensure bm is at the same resolution as warped
    unwarped = F.grid_sample(warped, bm, mode='bilinear', align_corners=True)
    return unwarped