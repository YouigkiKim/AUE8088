# ---------- loss.py (요약 패치) ----------
import torch
import torch.nn.functional as F
from utils.metrics import bbox_iou, pairwise_bbox_iou  # <- helper (아래 참고)
from utils.metrics import wh_iou          # wh_iou 이미 metrics.py에 존재

# 1) SimOTA (dynamic-k) -------------------------------------------------
def dynamic_k_matching(cost, pair_wise_iou, gt_classes, num_gt, fg_mask):
    """
    cost        : [num_gt, num_in_boxes_anchor]   (작을수록 좋음)
    pair_wise_iou: same shape, IoU 값
    gt_classes  : [num_gt]  (0 == 'people')
    fg_mask     : [N] 현재 layer 안에서 in-box 로 필터링된 anchor mask
    return: assigned_gt_idx  [N]  (-1 = negative)
    """
    N = cost.shape[1]
    if num_gt == 0:
        return torch.full((N,), -1, dtype=torch.long, device=cost.device)

    matching_matrix = torch.zeros_like(cost)
    # ①  각 GT마다 IoU 상위 k 계산
    for gt_idx in range(num_gt):
        # k = IoU 합의 정수값(최소 1, 최대 10)
        n_candidate_k = min(10, int(pair_wise_iou[gt_idx].sum().item()))
        n_candidate_k = max(n_candidate_k, 1)
        # cost top-k anchor 선택
        _, pos_idx = torch.topk(cost[gt_idx], k=n_candidate_k, largest=False)
        matching_matrix[gt_idx, pos_idx] = 1.0

    # ②  하나의 anchor가 여러 GT와 매칭되면, cost가 가장 낮은 GT만 유지
    anchor_matching_gt = matching_matrix.sum(0)
    if (anchor_matching_gt > 1).sum() > 0:
        multiple_match_idx = anchor_matching_gt.nonzero(as_tuple=False).squeeze(1)
        cost_min, min_cost_gt_idx = torch.min(cost[:, multiple_match_idx], dim=0)
        matching_matrix[:, multiple_match_idx] *= 0
        matching_matrix[min_cost_gt_idx, multiple_match_idx] = 1.0

    # ③  결과 정리
    fg_mask_inboxes = fg_mask.clone()
    assigned_gt_idx = matching_matrix.argmax(0)           # [N]
    assigned_gt_idx[anchor_matching_gt == 0] = -1         # negative
    return assigned_gt_idx


# 2) build_targets() → SimOTA ------------------------------------------
def build_targets_simota(self, p, targets):
    """
    기존 anchor-t/offset 대신 SimOTA를 써서
    indices, tbox, tcls, anchors 반환
    """
    na, _, device = self.na, targets.shape[0], self.device
    tcls, tbox, indices, anch = [], [], [], []
    for layer_idx, pi in enumerate(p):
        # (1) layer 정보
        bs, _, ny, nx = pi.shape[:4]
        na = self.na
        n_grid = ny * nx
        device = self.device
        anchors = self.anchors[layer_idx]                # (na, 2)

        # (2) 그리드 중심 좌표 테이블 (n_grid, 2)
        gj, gi = torch.meshgrid(
                torch.arange(ny, device=device),
                torch.arange(nx, device=device),
                indexing='ij')                            # torch>=1.10
        grid_xy = torch.stack((gi, gj), -1).view(-1, 2).float() + 0.5  # 0.5는 중심 보정

        # (3) GT 복사 & 스케일
        gt = targets.clone()
        gt[:, 2:4] *= nx       # x,y  → grid 단위
        gt[:, 4:6] *= nx       # w,h  → grid 단위  (정사각형 가정)

        # (4) (GT, anchor) IoU  → (M, na*n_grid)
        pair_iou = pairwise_bbox_iou(gt[:, 2:6], anchors)      # (M, na)
        pair_iou = pair_iou.repeat_interleave(n_grid, dim=1)   # (M, na*n_grid)

        # (5) GT-to-grid 중심거리  → (M, n_grid) → (M, na*n_grid)
        center_dist = ((gt[:, 2:4].unsqueeze(1) - grid_xy)**2).sum(-1).sqrt()  # (M, n_grid)
        center_dist = center_dist.repeat(1, na)                                # (M, na*n_grid)

        # (6) cost 계산
        lambda_center = 2.5
        cost = -pair_iou + lambda_center * center_dist / max(ny, nx)


        num_anchor = na * n_grid
        fg_mask = torch.ones(num_anchor, dtype=torch.bool, device=device)

        # (4) SimOTA dynamic-k 매칭
        gt_idx = dynamic_k_matching(cost, pair_iou, gt[:, 1].long(), gt.shape[0], fg_mask)

        # (5) positive anchor만 인덱스화
        pos_mask = gt_idx >= 0
        if pos_mask.sum() == 0:          # no positive
            tcls.append(torch.tensor([], device=device))
            tbox.append(torch.tensor([], device=device))
            indices.append((torch.tensor([], device=device),) * 4)
            anch.append(torch.tensor([], device=device))
            continue

        matched_gt = gt[gt_idx[pos_mask]]
        # grid indices
        gi, gj = (matched_gt[:, 2:4]).long().T
        a = torch.arange(na, device=device).repeat(nx*ny)[pos_mask] % na
        b = matched_gt[:, 0].long()      # batch id
        indices.append((b, a, gj, gi))
        tbox.append(matched_gt[:, 2:6] - torch.stack([gi, gj, gi*0, gj*0], 1))  # offset form
        anch.append(self.anchors[layer_idx][a])
        tcls.append(matched_gt[:, 1].long())
    return tcls, tbox, indices, anch


# 3) ComputeLoss.__call__ 내부
#  tcls, tbox, indices, anchors = self.build_targets_simota(p, targets)
