import torch
import torch.nn as nn
import torch.nn.functional as F


# Supervised Contrastive Loss (Khosla et al., 2020).
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        """
        Args:
            - temperature: Scaling factor for logits.
            - contrast_mode: 'all' or 'one', determines anchor selection.
            - base_temperature: Normalization factor for loss scaling.
        """
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Compute loss for model. Falls back to SimCLR-style unsupervised loss when labels/mask are None.

        Args:
            - features: [bsz, n_views, ...]
            - labels: [bsz]
            - mask: [bsz, bsz], mask_{i,j}=1 if sample j shares class with i
        Returns:
            A loss scalar.
        """
        device = features.device

        # shape check
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            # flatten feature dims -> [bsz, n_views, dim]
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size, n_views, dim = features.shape

        # L2-normalize for stable cosine-like similarity
        features = F.normalize(features, p=2, dim=-1)

        # build mask
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)  # binary mask: 1 if same class, else 0
        else:
            mask = mask.float().to(device)

        contrast_count = n_views
        # stack views into single batch: [bsz*n_views, dim]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            # use only the first view as anchors: [bsz, dim]
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            # use all views as anchors: [bsz*n_views, dim]
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute scaled dot-product logits: [bsz*anchor_count, bsz*contrast_count]
        logits = torch.matmul(anchor_feature, contrast_feature.T) / self.temperature

        # numerical stability
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        # tile mask to match views
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast
        logits_mask = torch.ones_like(mask, device=device)
        idx = torch.arange(batch_size * anchor_count, device=device).view(-1, 1)
        logits_mask.scatter_(1, idx, 0.0)
        mask = mask * logits_mask

        # log-softmax over contrasts
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # average log-likelihood over positives (avoid divide-by-zero)
        mask_pos_pairs = mask.sum(dim=1)
        denom = mask_pos_pairs.clamp_min(1.0)
        mean_log_prob_pos = (mask * log_prob).sum(1) / denom

        # temperature scaling and mean over anchors
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
