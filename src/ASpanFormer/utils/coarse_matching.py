import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange

from kornia.geometry.epipolar import find_fundamental
from kornia.geometry.epipolar import sampson_epipolar_distance
from time import time

INF = 1e9

def mask_border(m, b: int, v):
    """ Mask borders with value
    Args:
        m (torch.Tensor): [N, H0, W0, H1, W1]
        b (int)
        v (m.dtype)
    """
    if b <= 0:
        return

    m[:, :b] = v
    m[:, :, :b] = v
    m[:, :, :, :b] = v
    m[:, :, :, :, :b] = v
    m[:, -b:] = v
    m[:, :, -b:] = v
    m[:, :, :, -b:] = v
    m[:, :, :, :, -b:] = v


def mask_border_with_padding(m, bd, v, p_m0, p_m1):
    if bd <= 0:
        return

    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd] = v
    m[:, :, :, :, :bd] = v

    h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
    h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()
    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        m[b_idx, h0 - bd:] = v
        m[b_idx, :, w0 - bd:] = v
        m[b_idx, :, :, h1 - bd:] = v
        m[b_idx, :, :, :, w1 - bd:] = v


def compute_max_candidates(p_m0, p_m1):
    """Compute the max candidates of all pairs within a batch
    
    Args:
        p_m0, p_m1 (torch.Tensor): padded masks
    """
    h0s, w0s = p_m0.sum(1).max(-1)[0], p_m0.sum(-1).max(-1)[0]
    h1s, w1s = p_m1.sum(1).max(-1)[0], p_m1.sum(-1).max(-1)[0]
    max_cand = torch.sum(
        torch.min(torch.stack([h0s * w0s, h1s * w1s], -1), -1)[0])
    return max_cand


class CoarseMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # general config
        self.thr = config['thr']
        self.border_rm = config['border_rm']
        self.iter_thr = config["iter_thr"]
        # -- # for trainig fine-level LoFTR
        self.train_coarse_percent = config['train_coarse_percent']
        self.train_pad_num_gt_min = config['train_pad_num_gt_min']
        
        # we provide 2 options for differentiable matching
        self.match_type = config['match_type']
        if self.match_type == 'dual_softmax':
            self.temperature=nn.parameter.Parameter(torch.tensor(10.), requires_grad=True)
        elif self.match_type == 'sinkhorn':
            try:
                from .superglue import log_optimal_transport
            except ImportError:
                raise ImportError("download superglue.py first!")
            self.log_optimal_transport = log_optimal_transport
            self.bin_score = nn.Parameter(
                torch.tensor(config['skh_init_bin_score'], requires_grad=True))
            self.skh_iters = config['skh_iters']
            self.skh_prefilter = config['skh_prefilter']
        else:
            raise NotImplementedError()

        self.k = config['k']
        
        self.dist_thr = config['dist_thr']

        self.iter = config['iter']

        self.dist_weight = config['dist_weight']
     
    def forward(self, feat_c0, feat_c1, flow_list, data, mask_c0=None, mask_c1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            offset: [layer, B, H, W, 4] (*2)
            data (dict)
            mask_c0 (torch.Tensor): [N, L] (optional)
            mask_c1 (torch.Tensor): [N, S] (optional)
        Update:
            data (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
            NOTE: M' != M during training.
        """
        N, L, S, C = feat_c0.size(0), feat_c0.size(1), feat_c1.size(1), feat_c0.size(2)
        # normalize
        feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1]**.5,
                               [feat_c0, feat_c1])

        if self.match_type == 'dual_softmax':
            sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0,
                                      feat_c1) * self.temperature
            if mask_c0 is not None:
                sim_matrix.masked_fill_(
                    ~(mask_c0[..., None] * mask_c1[:, None]).bool(),
                    -INF)
            conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)
            
        elif self.match_type == 'sinkhorn':
            # sinkhorn, dustbin included
            sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1)
            if mask_c0 is not None:
                sim_matrix[:, :L, :S].masked_fill_(
                    ~(mask_c0[..., None] * mask_c1[:, None]).bool(),
                    -INF)

            # build uniform prior & use sinkhorn
            log_assign_matrix = self.log_optimal_transport(
                sim_matrix, self.bin_score, self.skh_iters)
            assign_matrix = log_assign_matrix.exp()
            conf_matrix = assign_matrix[:, :-1, :-1]

            # filter prediction with dustbin score (only in evaluation mode)
            if not self.training and self.skh_prefilter:
                filter0 = (assign_matrix.max(dim=2)[1] == S)[:, :-1]  # [N, L]
                filter1 = (assign_matrix.max(dim=1)[1] == L)[:, :-1]  # [N, S]
                conf_matrix[filter0[..., None].repeat(1, 1, S)] = 0
                conf_matrix[filter1[:, None].repeat(1, L, 1)] = 0

            if self.config['sparse_spvs']:
                data.update({'conf_matrix_with_bin': assign_matrix.clone()})

        data.update({'conf_matrix': conf_matrix})
        # predict coarse matches from conf_matrix
        self.get_coarse_match(conf_matrix, data)

        #update predicted offset
        if flow_list[0].shape[2]==flow_list[1].shape[2] and flow_list[0].shape[3]==flow_list[1].shape[3]:
            flow_list=torch.stack(flow_list,dim=0)
        data.update({'predict_flow':flow_list}) #[2*L*B*H*W*4]
        self.get_offset_match(flow_list,data,mask_c0,mask_c1)

    @torch.no_grad()
    def get_coarse_match(self, conf_matrix, data):
        """
        Args:
            conf_matrix (torch.Tensor): [N, L, S]
            data (dict): with keys ['hw0_i', 'hw1_i', 'hw0_c', 'hw1_c']
        Returns:
            coarse_matches (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'm_bids' (torch.Tensor): [M],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
        """
        axes_lengths = {
            'h0c': data['hw0_c'][0],
            'w0c': data['hw0_c'][1],
            'h1c': data['hw1_c'][0],
            'w1c': data['hw1_c'][1]
        }
        _device = conf_matrix.device

        # obtain matches iteratively
        for i in range(self.iter):

            # 1. confidence thresholding
            mask = conf_matrix > self.thr
            mask = rearrange(mask, 'b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c',
                            **axes_lengths)
            if 'mask0' not in data:
                mask_border(mask, self.border_rm, False)
            else:
                mask_border_with_padding(mask, self.border_rm, False,
                                        data['mask0'], data['mask1'])
            mask = rearrange(mask, 'b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)',
                            **axes_lengths)

            # 2. mutual nearest
            mask = mask \
                * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]) \
                * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])

            # 3. find all valid coarse matches
            # this only works when at most one `True` in each row
            mask_v, all_j_ids = mask.max(dim=2)
            b_ids, i_ids = torch.where(mask_v)
            j_ids = all_j_ids[b_ids, i_ids]
            mconf = conf_matrix[b_ids, i_ids, j_ids]

            # 4. Random sampling of training samples for fine-level LoFTR
            # (optional) pad samples with gt coarse-level matches
            if self.training:
                # NOTE:
                # The sampling is performed across all pairs in a batch without manually balancing
                # #samples for fine-level increases w.r.t. batch_size
                if 'mask0' not in data:
                    num_candidates_max = mask.size(0) * max(
                        mask.size(1), mask.size(2))
                else:
                    num_candidates_max = compute_max_candidates(
                        data['mask0'], data['mask1'])
                num_matches_train = int(num_candidates_max *
                                        self.train_coarse_percent)
                num_matches_pred = len(b_ids)
                assert self.train_pad_num_gt_min < num_matches_train, "min-num-gt-pad should be less than num-train-matches"
                
                # pred_indices is to select from prediction
                if num_matches_pred <= num_matches_train - self.train_pad_num_gt_min:
                    pred_indices = torch.arange(num_matches_pred, device=_device)
                else:
                    pred_indices = torch.randint(
                        num_matches_pred,
                        (num_matches_train - self.train_pad_num_gt_min, ),
                        device=_device)

                # gt_pad_indices is to select from gt padding. e.g. max(3787-4800, 200)
                gt_pad_indices = torch.randint(
                        len(data['spv_b_ids']),
                        (max(num_matches_train - num_matches_pred,
                            self.train_pad_num_gt_min), ),
                        device=_device)
                mconf_gt = torch.zeros(len(data['spv_b_ids']), device=_device)  # set conf of gt paddings to all zero

                b_ids, i_ids, j_ids, mconf = map(
                    lambda x, y: torch.cat([x[pred_indices], y[gt_pad_indices]],
                                        dim=0),
                    *zip([b_ids, data['spv_b_ids']], [i_ids, data['spv_i_ids']],
                        [j_ids, data['spv_j_ids']], [mconf, mconf_gt]))

            # These matches select patches that feed into fine-level network
            coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}

            # 4. Update with matches in original image resolution
            scale = data['hw0_i'][0] / data['hw0_c'][0]
            scale0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
            scale1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale
            mkpts0_c = torch.stack(
                [i_ids % data['hw0_c'][1], i_ids // data['hw0_c'][1]],
                dim=1) * scale0
            mkpts1_c = torch.stack(
                [j_ids % data['hw1_c'][1], j_ids // data['hw1_c'][1]],
                dim=1) * scale1
            
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
        
        dist_prob = nn.ReLU()(self.dist_thr - dist)
        dist_prob = nn.Sigmoid()(dist_prob)
        return torch.nan_to_num(dist_prob)

    @torch.no_grad()
    def get_fundamental(self, data):

        mkpts0 = data['topk_mkpts0']
        mkpts1 = data['topk_mkpts1']

        if mkpts0.dim() < 3:
            mkpts0 = mkpts0.unsqueeze(0)
            mkpts1 = mkpts1.unsqueeze(0)
        F = find_fundamental(mkpts0, mkpts1)

        data.update({
            'Fm': F
        })

    @torch.no_grad()
    def get_offset_match(self, flow_list, data,mask1,mask2):
        """
        Args:
            offset (torch.Tensor): [L, B, H, W, 2]
            data (dict): with keys ['hw0_i', 'hw1_i', 'hw0_c', 'hw1_c']
        Returns:
            coarse_matches (dict): {
                'm_bids' (torch.Tensor): [M],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
        """
        offset1=flow_list[0]
        bs,layer_num=offset1.shape[1],offset1.shape[0]
        
        #left side
        offset1=offset1.view(layer_num,bs,-1,4)
        conf1=offset1[:,:,:,2:].mean(dim=-1)
        if mask1 is not None:
            conf1.masked_fill_(~mask1.bool()[None].expand(layer_num,-1,-1),100)
        offset1=offset1[:,:,:,:2]
        self.get_offset_match_work(offset1,conf1,data,'left')

        #rihgt side
        if len(flow_list)==2:
            offset2=flow_list[1].view(layer_num,bs,-1,4)
            conf2=offset2[:,:,:,2:].mean(dim=-1)
            if mask2 is not None:
                conf2.masked_fill_(~mask2.bool()[None].expand(layer_num,-1,-1),100)
            offset2=offset2[:,:,:,:2]
            self.get_offset_match_work(offset2,conf2,data,'right')


    @torch.no_grad()
    def get_offset_match_work(self, offset,conf, data,side):
        bs,layer_num=offset.shape[1],offset.shape[0]
        # 1. confidence thresholding
        mask_conf= conf<2
        for index in range(bs):
            mask_conf[:,index,0]=True #safe guard in case that no match survives
        # 3. find offset matches
        scale = data['hw0_i'][0] / data['hw0_c'][0]
        l_ids,b_ids,i_ids = torch.where(mask_conf)
        j_coor=offset[l_ids,b_ids,i_ids,:2] *scale#[N,2]
        i_coor=torch.stack([i_ids%data['hw0_c'][1],i_ids//data['hw0_c'][1]],dim=1)*scale
        #i_coor=torch.as_tensor([[index%data['hw0_c'][1],index//data['hw0_c'][1]] for index in i_ids]).cuda().float()*scale #[N,2]
        # These matches is the current prediction (for visualization)
        data.update({
            'offset_bids_'+side: b_ids,  # mconf == 0 => gt matches
            'offset_lids_'+side: l_ids,
            'conf'+side: conf[mask_conf]
        })
        
        if side=='right':
            data.update({'offset_kpts0_f_'+side: j_coor.detach(),
            'offset_kpts1_f_'+side: i_coor})
        else:
            data.update({'offset_kpts0_f_'+side: i_coor,
            'offset_kpts1_f_'+side: j_coor.detach()})

    
