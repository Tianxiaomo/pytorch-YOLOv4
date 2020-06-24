import torch.nn as nn
import torch.nn.functional as F
from .torch_utils import *


def yolo_forward_alternative(output, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1,
                              validation=False):
    # Output would be invalid if it does not satisfy this assert
    # assert (output.size(1) == (5 + num_classes) * num_anchors)

    # print(output.size())

    # Slice the second dimension (channel) of output into:
    # [ 2, 2, 1, num_classes, 2, 2, 1, num_classes, 2, 2, 1, num_classes ]
    # And then into
    # bxy = [ 6 ] bwh = [ 6 ] det_conf = [ 3 ] cls_conf = [ num_classes * 3 ]
    batch = output.size(0)
    H = output.size(2)
    W = output.size(3)

    device = None
    cuda_check = output.is_cuda
    if cuda_check:
        device = output.get_device()


    # Prepare C-x, C-y, P-w, P-h (None of them are torch related)
    grid_x = np.expand_dims(np.linspace(0, W - 1, W), axis=0).repeat(H, 0).reshape(1, 1, H * W).repeat(batch, 0).repeat(num_anchors, 1)
    grid_y = np.expand_dims(np.linspace(0, H - 1, H), axis=1).repeat(W, 1).reshape(1, 1, H * W).repeat(batch, 0).repeat(num_anchors, 1)
    # Shape: [batch, num_anchors, H * W]
    grid_x_tensor = torch.tensor(grid_x, device=device, dtype=torch.float32)
    grid_y_tensor = torch.tensor(grid_y, device=device, dtype=torch.float32)

    anchor_array = np.array(anchors).reshape(1, num_anchors, 2)
    anchor_array = anchor_array.repeat(batch, 0)
    anchor_array = np.expand_dims(anchor_array, axis=3).repeat(H * W, 3)
    # Shape: [batch, num_anchors, 2, H * W]
    anchor_tensor = torch.tensor(anchor_array, device=device, dtype=torch.float32)

    # normalize coordinates to [0, 1]
    normal_array = np.array([1.0 / W, 1.0 / H, 1.0 / W, 1.0 / H], dtype=np.float32).reshape(1, 1, 4)
    normal_array = normal_array.repeat(batch, 0)
    normal_array = normal_array.repeat(num_anchors * H * W, 1)
    # Shape: [batch, num_anchors * H * W, 4]
    normal_tensor = torch.tensor(normal_array, device=device, dtype=torch.float32)

    bxy_list = []
    bwh_list = []
    det_confs_list = []
    cls_confs_list = []

    for i in range(num_anchors):
        begin = i * (5 + num_classes)
        end = (i + 1) * (5 + num_classes)
        
        bxy_list.append(output[:, begin : begin + 2])
        bwh_list.append(output[:, begin + 2 : begin + 4])
        det_confs_list.append(output[:, begin + 4 : begin + 5])
        cls_confs_list.append(output[:, begin + 5 : end])

    # Shape: [batch, num_anchors * 2, H, W]
    bxy = torch.cat(bxy_list, dim=1)
    # Shape: [batch, num_anchors * 2, H, W]
    bwh = torch.cat(bwh_list, dim=1)

    # Shape: [batch, num_anchors, H, W]
    det_confs = torch.cat(det_confs_list, dim=1)
    # Shape: [batch, num_anchors * H * W]
    det_confs = det_confs.view(batch, num_anchors * H * W)

    # Shape: [batch, num_anchors * num_classes, H, W]
    cls_confs = torch.cat(cls_confs_list, dim=1)
    # Shape: [batch, num_anchors, num_classes, H * W]
    cls_confs = cls_confs.view(batch, num_anchors, num_classes, H * W)
    # Shape: [batch, num_anchors, num_classes, H * W] --> [batch, num_anchors * H * W, num_classes] 
    cls_confs = cls_confs.permute(0, 1, 3, 2).reshape(batch, num_anchors * H * W, num_classes)

    # Apply sigmoid(), exp() and softmax() to slices
    #
    bxy = torch.sigmoid(bxy)
    bwh = torch.exp(bwh)
    det_confs = torch.sigmoid(det_confs)
    cls_confs = torch.nn.Softmax(dim=2)(cls_confs)

    # Shape: [batch, num_anchors, 2, H * W]
    bxy = bxy.view(batch, num_anchors, 2, H * W)
    # Shape: [batch, num_anchors, 2, H * W]
    bwh = bwh.view(batch, num_anchors, 2, H * W)

    # Apply C-x, C-y, P-w, P-h
    bxy[:, :, 0] += grid_x_tensor
    bxy[:, :, 1] += grid_y_tensor

    print(anchor_tensor.size())
    bwh *= anchor_tensor

    # Shape: [batch, num_anchors, 4, H * W] --> [batch, num_anchors * H * W, 4]
    boxes = torch.cat((bxy, bwh), dim=2).permute(0, 1, 3, 2).reshape(batch, num_anchors * H * W, 4)

    print(normal_tensor.size())
    boxes *= normal_tensor

    det_confs = det_confs.view(batch, num_anchors * H * W, 1)
    confs = cls_confs * det_confs

    # boxes: [batch, num_anchors * H * W, 4]
    # confs: [batch, num_anchors * H * W, num_classes]

    return  boxes, confs



def yolo_forward(output, conf_thresh, num_classes, anchors, num_anchors, only_objectness=1,
                              validation=False):
    # Output would be invalid if it does not satisfy this assert
    # assert (output.size(1) == (5 + num_classes) * num_anchors)

    # print(output.size())

    # Slice the second dimension (channel) of output into:
    # [ 2, 2, 1, num_classes, 2, 2, 1, num_classes, 2, 2, 1, num_classes ]
    # And then into
    # bxy = [ 6 ] bwh = [ 6 ] det_conf = [ 3 ] cls_conf = [ num_classes * 3 ]
    batch = output.size(0)
    H = output.size(2)
    W = output.size(3)

    bxy_list = []
    bwh_list = []
    det_confs_list = []
    cls_confs_list = []

    for i in range(num_anchors):
        begin = i * (5 + num_classes)
        end = (i + 1) * (5 + num_classes)
        
        bxy_list.append(output[:, begin : begin + 2])
        bwh_list.append(output[:, begin + 2 : begin + 4])
        det_confs_list.append(output[:, begin + 4 : begin + 5])
        cls_confs_list.append(output[:, begin + 5 : end])

    # Shape: [batch, num_anchors * 2, H, W]
    bxy = torch.cat(bxy_list, dim=1)
    # Shape: [batch, num_anchors * 2, H, W]
    bwh = torch.cat(bwh_list, dim=1)

    # Shape: [batch, num_anchors, H, W]
    det_confs = torch.cat(det_confs_list, dim=1)
    # Shape: [batch, num_anchors * H * W]
    det_confs = det_confs.view(batch, num_anchors * H * W)

    # Shape: [batch, num_anchors * num_classes, H, W]
    cls_confs = torch.cat(cls_confs_list, dim=1)
    # Shape: [batch, num_anchors, num_classes, H * W]
    cls_confs = cls_confs.view(batch, num_anchors, num_classes, H * W)
    # Shape: [batch, num_anchors, num_classes, H * W] --> [batch, num_anchors * H * W, num_classes] 
    cls_confs = cls_confs.permute(0, 1, 3, 2).reshape(batch, num_anchors * H * W, num_classes)

    # Apply sigmoid(), exp() and softmax() to slices
    #
    bxy = torch.sigmoid(bxy)
    bwh = torch.exp(bwh)
    det_confs = torch.sigmoid(det_confs)
    cls_confs = torch.nn.Softmax(dim=2)(cls_confs)

    # Prepare C-x, C-y, P-w, P-h (None of them are torch related)
    grid_x = np.expand_dims(np.expand_dims(np.expand_dims(np.linspace(0, W - 1, W), axis=0).repeat(H, 0), axis=0), axis=0)
    grid_y = np.expand_dims(np.expand_dims(np.expand_dims(np.linspace(0, H - 1, H), axis=1).repeat(W, 1), axis=0), axis=0)
    # grid_x = torch.linspace(0, W - 1, W).reshape(1, 1, 1, W).repeat(1, 1, H, 1)
    # grid_y = torch.linspace(0, H - 1, H).reshape(1, 1, H, 1).repeat(1, 1, 1, W)

    anchor_w = []
    anchor_h = []
    for i in range(num_anchors):
        anchor_w.append(anchors[i * 2])
        anchor_h.append(anchors[i * 2 + 1])

    device = None
    cuda_check = output.is_cuda
    if cuda_check:
        device = output.get_device()

    bx_list = []
    by_list = []
    bw_list = []
    bh_list = []

    # Apply C-x, C-y, P-w, P-h
    for i in range(num_anchors):
        ii = i * 2
        # Shape: [batch, 1, H, W]
        bx = bxy[:, ii] + torch.tensor(grid_x, device=device, dtype=torch.float32) # grid_x.to(device=device, dtype=torch.float32)
        # Shape: [batch, 1, H, W]
        by = bxy[:, ii + 1] + torch.tensor(grid_y, device=device, dtype=torch.float32) # grid_y.to(device=device, dtype=torch.float32)
        # Shape: [batch, 1, H, W]
        bw = bwh[:, ii] * anchor_w[i]
        # Shape: [batch, 1, H, W]
        bh = bwh[:, ii + 1] * anchor_h[i]

        bx_list.append(bx)
        by_list.append(by)
        bw_list.append(bw)
        bh_list.append(bh)


    ########################################
    #   Figure out bboxes from slices     #
    ########################################
    
    # Shape: [batch, num_anchors, H, W]
    bx = torch.cat(bx_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    by = torch.cat(by_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    bw = torch.cat(bw_list, dim=1)
    # Shape: [batch, num_anchors, H, W]
    bh = torch.cat(bh_list, dim=1)

    # normalize coordinates to [0, 1]
    bx = bx / W
    by = by / H
    bw = bw / W
    bh = bh / H

    # Shape: [batch, num_anchors * H * W, 1]
    bx = bx.view(batch, num_anchors * H * W, 1)
    by = by.view(batch, num_anchors * H * W, 1)
    bw = bw.view(batch, num_anchors * H * W, 1)
    bh = bh.view(batch, num_anchors * H * W, 1)

    # Shape: [batch, num_anchors * h * w, 4]
    boxes = torch.cat((bx, by, bw, bh), dim=2).view(batch, num_anchors * H * W, 4)

    # boxes:     [batch, num_anchors * H * W, num_classes, 4]
    # cls_confs: [batch, num_anchors * H * W, num_classes]
    # det_confs: [batch, num_anchors * H * W]

    det_confs = det_confs.view(batch, num_anchors * H * W, 1)
    confs = cls_confs * det_confs

    # boxes: [batch, num_anchors * H * W, 4]
    # confs: [batch, num_anchors * H * W, num_classes]

    return  boxes, confs


class YoloLayer(nn.Module):
    ''' Yolo layer
    model_out: while inference,is post-processing inside or outside the model
        true:outside
    '''
    def __init__(self, anchor_mask=[], num_classes=0, anchors=[], num_anchors=1, stride=32, model_out=False):
        super(YoloLayer, self).__init__()
        self.anchor_mask = anchor_mask
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.anchor_step = len(anchors) // num_anchors
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.stride = stride
        self.seen = 0

        self.model_out = model_out

    def forward(self, output, target=None):

        masked_anchors = []
        for m in self.anchor_mask:
            masked_anchors += self.anchors[m * self.anchor_step:(m + 1) * self.anchor_step]
        masked_anchors = [anchor / self.stride for anchor in masked_anchors]

        return yolo_forward(output, self.thresh, self.num_classes, masked_anchors, len(self.anchor_mask))

