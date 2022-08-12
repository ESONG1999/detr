# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer
from .transformer_BEV import build_transformer_BEV


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, transformer_BEV, num_classes, num_queries, aux_loss=False, n_depth_bins = 10):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.transformer_BEV = transformer_BEV
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_embed_B = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.bev_embed = MLP(hidden_dim, hidden_dim, 2, 2)
        self.angle_embed = MLP(hidden_dim, hidden_dim, 24, 2)
        self.dim_embed = MLP(hidden_dim, hidden_dim, 2, 2)
        # depth predicted as => d_bin_id* bin_width + bin_width/2 + correction
        # -bin_width/2 <= correction < bin_width/2
        self.n_depth_bins = n_depth_bins
        self.depth_delta =  nn.Linear(hidden_dim, 1) # regression for finding delta
        self.depth_bin = nn.Linear(hidden_dim, self.n_depth_bins)
        self.calib_cam_to_bev = nn.Linear(2, 2)


    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        # print(query_B.size())
        # Depth and 2D detection part
        output_depth_bin = F.softmax(self.depth_bin(hs), dim = 3)
        output_depth_delta = torch.tanh(self.depth_delta(hs))/2
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        
        # head and feet part
        mid_2d = (outputs_coord[:, :, :, 0] + outputs_coord[:, :, :, 2]) / 2
        head_2d = torch.stack((mid_2d, outputs_coord[:, :, :, 1]), dim = 3)
        feet_2d = torch.stack((mid_2d, outputs_coord[:, :, :, 3]), dim = 3)
        # print(head_2d.size())
        # print(feet_2d.size())
        head_bev = self.calib_cam_to_bev(head_2d).sigmoid()
        feet_bev = self.calib_cam_to_bev(feet_2d).sigmoid()

        # print(output_depth_bin[-1].size())
        # print(output_depth_delta[-1].size())
        # print(outputs_coord[-1].size())
        # a = output_depth_bin + outputs_class

        # BEV stage
        hs_B = self.transformer_BEV(self.input_proj(src), mask, self.query_embed_B.weight, pos[-1], output_depth_bin[-1], output_depth_delta[-1], head_bev[-1], feet_bev[-1])[0]
        outputs_bev = self.bev_embed(hs_B).sigmoid()
        outputs_dim = self.dim_embed(hs_B)
        outputs_angle = self.angle_embed(hs_B)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_bev': outputs_bev[-1], 'pred_dim': outputs_dim[-1], 'pred_angle': outputs_angle[-1], 'pred_depth_bin': output_depth_bin[-1], 'pred_depth_delta': output_depth_delta[-1], 'pred_head_bev': head_bev[-1], 'pred_feet_bev': feet_bev[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_bev, outputs_dim, outputs_angle, output_depth_bin, output_depth_delta, head_bev, feet_bev)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_bev, outputs_dim, outputs_angle, output_depth_bin, output_depth_delta, head_bev, feet_bev):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_bev': c, 'pred_dim': d, 'pred_angle': e, 'pred_depth_bin': f, 'pred_depth_delta': g, 'pred_head_bev': h, 'pred_feet_bev': i}
                for a, b, c, d, e, f, g, h, i in zip(outputs_class[:-1], outputs_coord[:-1], outputs_bev[:-1], outputs_dim[:-1], outputs_angle[:-1], output_depth_bin[:-1], output_depth_delta[:-1], head_bev[:-1], feet_bev[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, n_depth_bins = None, depth_bin_res = None):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.n_depth_bins = n_depth_bins
        self.depth_bin_res = depth_bin_res

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_depth_multi_bin(self, outputs, targets, indices, num_boxes):
        """Classification loss (NLL)
        targets dicts must contain the key "depth" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_depth_bin' in outputs
        assert 'pred_depth_delta' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_depth_bin = outputs['pred_depth_bin'][idx].squeeze()
        src_depth_delta = outputs['pred_depth_delta'][idx].squeeze()

        target_depth = torch.cat([t['depth'][i] for t, (_, i) in zip(targets, indices)])
        target_depth_bin_cls = target_depth/self.depth_bin_res
        target_depth_bin_cls = target_depth_bin_cls.int()
        target_depth_delta = target_depth - target_depth_bin_cls*self.depth_bin_res - self.depth_bin_res/2
        target_depth_bin = torch.zeros_like(src_depth_bin)
        target_depth_bin[torch.arange(src_depth_bin.size()[0]),target_depth_bin_cls.long()] = 1

        loss_depth_reg = F.mse_loss(src_depth_delta, target_depth_delta)
        loss_depth_class = F.cross_entropy(src_depth_bin, target_depth_bin)
        # TODO - confirm the formula
        loss = loss_depth_reg/2 + loss_depth_class/2
        losses = {}
        losses['loss_depth'] = loss
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def loss_bev(self, outputs, targets, indices, num_boxes):
        assert 'pred_bev' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_bev = outputs['pred_bev'][idx].squeeze()
        target_bev = torch.cat([t['bev'][i] for t, (_, i) in zip(targets, indices)])
        loss = F.l1_loss(src_bev, target_bev, reduction='none')
        losses = {}
        losses['loss_bev'] = loss.sum() / num_boxes
        return losses
    
    def loss_head(self, outputs, targets, indices, num_boxes):
        assert 'pred_head_bev' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_bev = outputs['pred_head_bev'][idx].squeeze()
        target_bev = torch.cat([t['bev'][i] for t, (_, i) in zip(targets, indices)])
        loss = F.l1_loss(src_bev, target_bev, reduction='none')
        losses = {}
        losses['loss_head_bev'] = loss.sum() / num_boxes
        return losses

    def loss_feet(self, outputs, targets, indices, num_boxes):
        assert 'pred_feet_bev' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_bev = outputs['pred_feet_bev'][idx].squeeze()
        target_bev = torch.cat([t['bev'][i] for t, (_, i) in zip(targets, indices)])
        loss = F.l1_loss(src_bev, target_bev, reduction='none')
        losses = {}
        losses['loss_feet_bev'] = loss.sum() / num_boxes
        return losses

    def loss_dims(self, outputs, targets, indices, num_boxes):
        assert 'pred_dim' in outputs
        # idx = self._get_src_permutation_idx(indices)
        # src_dim = outputs['pred_dim'][idx].squeeze()
        # target_dim = torch.cat([t['dim'][i] for t, (_, i) in zip(targets, indices)])
        # loss = F.mse_loss(src_dim, target_dim)
        # losses = {'loss_bev' : loss}

        idx = self._get_src_permutation_idx(indices)
        src_dims = outputs['pred_dim'][idx]
        target_dims = torch.cat([t['dim'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        dimension = target_dims.clone().detach()
        dim_loss = torch.abs(src_dims - target_dims)
        dim_loss /= dimension
        with torch.no_grad():
            compensation_weight = F.l1_loss(src_dims, target_dims) / dim_loss.mean()
        dim_loss *= compensation_weight
        losses = {}
        losses['loss_dim'] = dim_loss.sum() / num_boxes

        return losses
    
    def loss_angles(self, outputs, targets, indices, num_boxes):  

        idx = self._get_src_permutation_idx(indices)
        heading_input = outputs['pred_angle'][idx]
        target_heading_cls = torch.cat([t['heading_bin'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_heading_res = torch.cat([t['heading_res'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        heading_input = heading_input.view(-1, 24)
        heading_target_cls = target_heading_cls.view(-1).long()
        heading_target_res = target_heading_res.view(-1)

        # classification loss
        heading_input_cls = heading_input[:, 0:12]
        cls_loss = F.cross_entropy(heading_input_cls, heading_target_cls, reduction='none')

        # regression loss
        heading_input_res = heading_input[:, 12:24]
        cls_onehot = torch.zeros(heading_target_cls.shape[0], 12).cuda().scatter_(dim=1, index=heading_target_cls.view(-1, 1), value=1)
        heading_input_res = torch.sum(heading_input_res * cls_onehot, 1)
        reg_loss = F.l1_loss(heading_input_res, heading_target_res, reduction='none')
        
        angle_loss = cls_loss + reg_loss
        losses = {}
        losses['loss_angle'] = angle_loss.sum() / num_boxes 
        return losses

    

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        # loss_map = {
        #     'labels': self.loss_labels,
        #     'cardinality': self.loss_cardinality,
        #     'boxes': self.loss_boxes,
        #     'masks': self.loss_masks
        # }
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'bev': self.loss_bev,
            'dim': self.loss_dims,
            'angle': self.loss_angles,
            'depth': self.loss_depth_multi_bin,
            'head': self.loss_head,
            'feet': self.loss_feet
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    # num_classes = 20 if args.dataset_file != 'coco' else 91
    num_classes = args.num_classes+1 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)
    transformer_BEV = build_transformer_BEV(args)

    n_depth_bins = args.num_depth_bins # 9
    depth_bin_res = args.depth_bin_res # 10

    model = DETR(
        backbone,
        transformer,
        transformer_BEV,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        n_depth_bins = n_depth_bins
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    weight_dict['loss_bev'] = args.bev_loss_coef
    weight_dict['loss_dim'] = args.dim_loss_coef
    weight_dict['loss_angle'] = args.angle_loss_coef
    weight_dict['loss_depth'] = args.depth_loss_coef
    weight_dict['loss_head_bev'] = args.head_loss_coef
    weight_dict['loss_feet_bev'] = args.feet_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality', 'bev', 'dim', 'angle', 'depth', 'head', 'feet']
    # losses = ['bev', 'dim', 'angle']

    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses, n_depth_bins = n_depth_bins, depth_bin_res = depth_bin_res)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
