import torch
import torch.nn as nn
import torch.nn.functional as F

from einops.einops import rearrange
from kornia.geometry.epipolar import find_fundamental
from kornia.geometry.epipolar import sampson_epipolar_distance

class CoarseMatching(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # threshold and boder removal
        self.thr = config["thr"]
        self.iter_thr = config["iter_thr"]
        self.border_rm = config["border_rm"]

        # for training fine-level
        self.train_coarse_percent = config["train_coarse_percent"]
        self.train_pad_num_gt_min = config["train_pad_num_gt_min"]

        self.match_type = config["match_type"]
        self.temperature = config["dsmax_temperature"]
        
        self.k = config['k']
        
        self.dist_thr = config['dist_thr']

        self.iter = config['iter']

        self.dist_weight = config['dist_weight']

    def forward(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None):
        """
        Args:
            feat_c0 (tensor): [N, L, C]
            feat_c1 (tensor): [N, S, C]
            data (dict)
            mask_c0 (tensor): [N, L]
            mask_c1 (tensor): [N, S]

        Update: data: {
            'b_ids' (tensor),
            'i_ids' (tensor),
            'j_ids' (tensor),
            'gt_mask' (tensor),
            'mkpts0_c' (tensor),
            'mkpts1_c' (tensor),
            'mconf' (tensor)
        }
        """
        L, S = feat_c0.size(1), feat_c1.size(1)
        INF = 1e9
    
        # normalize, divide by the sqrt of the channel dimension, blance the scale of features
        feat_c0, feat_c1 = map(
            lambda feat: feat / feat.shape[-1] ** 0.5, [feat_c0, feat_c1]
        )
    
        # dual_softmax
        # compute similarity matrix, self.temperature control sharpness of the softmax
        sim_matrix = (
            torch.einsum("nlc, nsc -> nls", feat_c0, feat_c1) / self.temperature
        )
    
        # if mask out certain features
        if mask_c0 is not None:
            sim_matrix.masked_fill_(
                ~(mask_c0[..., None] * mask_c1[:, None]).bool(), -INF
            )
    
        # Computes the confidence matrix by applying softmax on the similarity matrix along both dimension
        # first part calculates soft prob for each row.  Each row represents the similarity scores between
        # a point in feat_c0 and all points in feat_c1 a point in feat_c0 and all points in feat_c1.  Second part for the same for column
        conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(
            sim_matrix, 2
        )  # [N, (h0c * w0c), (h1c * w1c)]
    
        data.update({"conf_matrix": conf_matrix})
    
        # predict coarse matches from conf_matrix and fundamental matrix
        self.get_matches(conf_matrix, data)

    @torch.no_grad()
    def get_matches(self, conf_matrix, data):
        """
        Args:
            conf_matrix (tensor): [N, L, S]
            data (dict)
        return:
            coarse_matches (dict): {
                'b_ids' (tensor),
                'i_ids' (tensor),
                'j_ids' (tensor),
                'gt_mask' (tensor),
                'm_bids' (tensor),
                'mkpts0_c' (tensor),
                'mkpts1_c' (tensor),
                'mconf' (tensor)
            }
        """

        axes_lengths = {
            "h0c": data["hw0_c"][0],
            "w0c": data["hw0_c"][1],
            "h1c": data["hw1_c"][0],
            "w1c": data["hw1_c"][1],
        }
        device = conf_matrix.device
        
        if self.training:
            iter = 5
        else:
            iter = self.iter
        
        # obtain matches iteratively when inferencing
        for i in range(iter):
            
            # 1. confidence thresholding based on a threshold value
            if i == self.iter - 1:
                mask = conf_matrix > self.thr
            else:
                mask = conf_matrix > self.iter_thr
            mask = rearrange(
                mask, "b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c", **axes_lengths
            )
    
            # removing border regions using a mask
            if "mask0" not in data:
                mask_border(mask, self.border_rm, False)
            else:
                mask_border_with_padding(
                    mask, self.border_rm, False, data["mask0"], data["mask1"]
                )
    
            # rearrange the dimesions of mask back to original
            mask = rearrange(
                mask, "b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)", **axes_lengths
            )
    
            # 2. mutual nearest
            # applying mutual nearest neighbor selection based on the confidence matrix
            # it selects the elements in mask that correspond to the max values in their respective rows and columns
            # The resulting mask will have True values only where the corresponding elements in conf_matrix
            # are both max in their respective row and column
            mask = (
                mask
                * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0])
                * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])
            )
    
            # 3. final all valid coarse matches based on the mask
            # mask_v contains the max values along each row of the mask
            # all__j_ids contains the corresponding indices of the max values along each row
            mask_v, all_j_ids = mask.max(dim=2)
    
            # find the indices of True values in the mask_v, b_ids and i_ids are indices of rows and colums
            # j_ids is the indices of the maximum values along in the 3rd dimensions. i_ids for image0 and j_ids for image1
            b_ids, i_ids = torch.where(mask_v)
            j_ids = all_j_ids[b_ids, i_ids]
            mconf = conf_matrix[b_ids, i_ids, j_ids]
    
            # 4. random sampling of training samples for fine-level LoFTR
            # (optional) pad samples with gt coarse-level matches
            if self.training:
                # the sampling is performed across all pairs in a batch without manually balancing
                # the number of samples for fine-level increases w.r.t batch size
                if "mask0" not in data:
                    num_candidates_max = mask.size(0) * max(mask.size(1), mask.size(2))
                else:
                    num_candidates_max = compute_max_candidates(
                        data["mask0"], data["mask1"]
                    )
    
                num_matches_train = int(num_candidates_max * self.train_coarse_percent)
                num_matches_pred = len(b_ids)
                assert (
                    self.train_pad_num_gt_min < num_matches_train
                ), "min-num-gt-pad should be less than num_matches_train"
    
                # select indices for predictions (sampling)
                if num_matches_pred <= num_matches_train - self.train_pad_num_gt_min:
                    pred_indices = torch.arange(num_matches_pred, device=device)
                else:
                    pred_indices = torch.randint(
                        num_matches_pred,
                        (num_matches_train - self.train_pad_num_gt_min,),
                        device=device,
                    )
    
                # gt_pad_indices is to select from gt padding, selecting indices for padding with group true
                gt_pad_indices = torch.randint(
                    len(data["spv_b_ids"]),
                    (max(num_matches_train - num_matches_pred, self.train_pad_num_gt_min),),
                    device=device,
                )
                mconf_gt = torch.zeros(len(data['spv_b_ids']), device=device)
    
                # concatenating and updating the indices and confidences
                b_ids, i_ids, j_ids, mconf = map(
                    lambda x, y: torch.cat([x[pred_indices], y[gt_pad_indices]], dim=0),
                    *zip([b_ids, data['spv_b_ids']], [i_ids, data['spv_i_ids']],
                         [j_ids, data['spv_j_ids']], [mconf, mconf_gt]))
                
            # these matches select patches that feed into fine-level network
            coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}
    
            # 4. update with matches in original image resolution
            # scale the matches to the original image resolution
            scale = data['hw0_i'][0] / data['hw0_c'][0]
            scale0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
            scale1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale
    
            # computes the modified keypoint positions by applying the scale to the keypoint
            mkpts0_c = torch.stack([i_ids % data['hw0_c'][1], i_ids // data['hw0_c'][1]], dim=1) * scale0
            mkpts1_c = torch.stack([j_ids % data['hw1_c'][1], j_ids // data['hw1_c'][1]], dim=1) * scale1
            
            data.update({'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids, 'mconf': mconf, 'mkpts0_c':mkpts0_c, 'mkpts1_c': mkpts0_c})
            
            stop = False
            if i < self.iter - 1:
                self.get_top_k_pairs(data)
                if data['topk_mkpts0'] is None or data['topk_mkpts1'] is None:
                    stop = True
                elif data['topk_mkpts0'].size(0) == 100:
                    stop = True
                else:
                    self.get_fundamental(data)
                    
                    dist_prob = self.cal_sampson_distance_prob(mkpts0_c, mkpts1_c, data)

                    # 5 update conf by multiply the dist_prob
                    temp  = conf_matrix[b_ids, i_ids, j_ids] * dist_prob * 1.2
                    temp[temp > 1] = 0.99
                    conf_matrix[b_ids, i_ids, j_ids] = temp

            # These matches is the current prediction (for visualization)
            data.update({
                'gt_mask': mconf == 0,
                'm_bids': b_ids[mconf != 0],  # mconf == 0 => gt matches
                'mkpts0_c': mkpts0_c[mconf != 0],
                'mkpts1_c': mkpts1_c[mconf != 0],
                'mconf': mconf[mconf != 0]
            })

            if stop:
                break

    @torch.no_grad()
    def get_top_k_pairs(self, data):
        # assume batch size equals to 1

        mconf, b_ids, i_ids, j_ids = data['mconf'], data['b_ids'], data['i_ids'], data['j_ids']

        # get pairs ready for rotation and translation estimate
        scale = data['hw0_i'][0] / data['hw0_c'][0]
        
        n = mconf.shape[0]
        topk = int(n*self.k) if int(n*self.k) >100 else 100
        
        if n == 0:
            topk_mkpts0 = None
            topk_mkpts1 = None
        else:
            if mconf.shape[0] < topk :
                mconf = mconf.repeat(topk)
                b_ids = b_ids.repeat(topk)
                i_ids = i_ids.repeat(topk)
                j_ids = j_ids.repeat(topk)

            top_k_value, top_k_index = torch.topk(mconf, topk, dim=0,largest=True, sorted=True)

            scale0 = scale * data['scale0'][b_ids[top_k_index]] if 'scale0' in data else scale
            scale1 = scale * data['scale1'][b_ids[top_k_index]] if 'scale1' in data else scale

            topk_mkpts0 = torch.stack([i_ids[top_k_index] % data["hw0_c"][1], i_ids[top_k_index] // data["hw0_c"][0]], dim = 1)* scale0

            topk_mkpts1 = torch.stack([j_ids[top_k_index] % data["hw1_c"][1], j_ids[top_k_index] // data["hw1_c"][0]], dim = 1)* scale1
            

        data.update({
            'topk_mkpts0': topk_mkpts0,
            'topk_mkpts1': topk_mkpts1
        })
    
    @torch.no_grad()
    def cal_sampson_distance_prob(self, mkpts0, mkpts1, data):
        Fm = data['Fm']
        dist = sampson_epipolar_distance(mkpts0, mkpts1, Fm)

        dist_prob = nn.ReLU()(self.dist_thr - dist) / (self.dist_thr /5)
        dist_prob = nn.Sigmoid()(dist_prob)
        return torch.nan_to_num(dist_prob)

    @torch.no_grad()
    def get_fundamental(self, data):

        mkpts0 = data['topk_mkpts0']
        mkpts1 = data['topk_mkpts1']

        if mkpts0.dim() < 3:
            mkpts0 = mkpts0.unsqueeze(0)
            mkpts1 = mkpts1.unsqueeze(0)
        Fm = find_fundamental(mkpts0, mkpts1)

        data.update({
            'Fm': Fm
        })
        
    

def mask_border(m, b, v):
    """
    Mask borders with value
    Args:
        m (tensor): [N, H0, W0, H1, W1]
        b (int)
        v (bool)
    """
    if b <= 0:
        return

    m[:, :b] = v
    m[:, -b:] = v

    m[:, :, :b] = v
    m[:, :, -b:] = v

    m[:, :, :, :b] = v
    m[:, :, :, -b:] = v

    m[:, :, :, :, :b] = v
    m[:, :, :, :, -b:] = v


def mask_border_with_padding(m, b, v, p_m0, p_m1):
    """
    Mask borders with value
    Args:
        m (tensor): [N, H0, W0, H1, W1]
        b (int)
        v (bool)
        p_m0 (tensor): [N H0 W0]
        p_m1 (tensor): [N H1 W1]
    """

    if b <= 0:
        return

    m[:, :b] = v
    m[:, :, :b] = v
    m[:, :, :, :b] = v
    m[:, :, :, :, :b] = v

    # find h, w of the mask, and mask the border of the original
    h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
    h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()
    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        m[b_idx, h0 - b :] = v
        m[b_idx, :, w0 - b :] = v
        m[b_idx, :, :, h1 - b :] = v
        m[b_idx, :, :, :, w1 - b :] = v


def compute_max_candidates(p_m0, p_m1):
    """
    Compute the max candidates of all pairs within a batch

    Args:
        p_m0, p_m1 (tensor): padded masks [N, H, W]
    """

    h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
    h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()
    max_cand = torch.sum(torch.min(torch.stack([h0s * w0s, h1s * w1s], -1), -1)[0])
    return max_cand
