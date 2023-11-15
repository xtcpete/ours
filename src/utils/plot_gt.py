import matplotlib
import numpy as np
import torch
import matplotlib.pyplot as plt

def calculate_mkpts(data):
    # obtain image shapes
    hw0_i = data['image0'].shape[2:]
    hw1_i = data['image1'].shape[2:]
    hw0_c = [np.sqrt(data['conf_matrix_gt'].shape[1]), np.sqrt(data['conf_matrix_gt'].shape[1])]
    hw1_c = hw0_c

    b_ids, i_ids, j_ids = data['spv_b_ids'], data['spv_i_ids'], data['spv_j_ids']
    scale = hw0_i[0] / hw0_c[0]
    scale0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
    scale1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale
    mkpts0_c = torch.stack(
              [i_ids % hw0_c[1], i_ids // hw0_c[1]],
              dim=1) * scale0
    mkpts1_c = torch.stack(
              [j_ids % hw1_c[1], j_ids // hw1_c[1]],
              dim=1) * scale1
    return mkpts0_c, mkpts1_c
  

def make_matching_figure(
        img0, img1, mkpts0, mkpts1, color = None,
        kpts0=None, kpts1=None, text=None, dpi=75, path=None):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1, 2, figsize=(12, 8), dpi=dpi)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')
    for i in range(2):   # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    
    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                            (fkpts0[i, 1], fkpts1[i, 1]),
                                            transform=fig.transFigure, c=color[i], linewidth=1, alpha=0.4)
                                        for i in range(len(mkpts0))]
        
        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)

    # put txts
    if text != None:
        txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
        fig.text(
            0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
            fontsize=15, va='top', ha='left', color=txt_color)

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return fig