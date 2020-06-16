
class NMS(nn.Module):
    def __init__(self, size=2, stride=1):
        super(NMS, self).__init__()
        self.size = size
        self.stride = stride

    def forward(boxes, scores, _keep, overlap_threshold=0.5, min_mode=False):

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]

        scores = boxes[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        _, order = scores.sort(dim=0, descending=True)
        cnt = 0
        
        while order.size()[0] > 1 and cnt < _keep.shape[0]:
            _keep[cnt] = order[0]
            cnt += 1
            xx1 = torch.max(x1[order[0]], x1[order[1:]])
            yy1 = torch.max(y1[order[0]], y1[order[1:]])
            xx2 = torch.min(x2[order[0]], x2[order[1:]])
            yy2 = torch.min(y2[order[0]], y2[order[1:]])

            w = torch.clamp(xx2-xx1, min=0)
            h = torch.clamp(yy2-yy1, min=0)
            inter = w * h
            if min_mode:
                ovr = inter / torch.min(areas[order[0]], areas[order[1:]])
            else:
                ovr = inter / (areas[order[0]] + areas[order[1:]] - inter)

            inds = torch.nonzero(ovr <= overlap_threshold).squeeze()
            if inds.dim():
                order = order[inds + 1]
            else:
                break

        return _keep[:cnt]

    