# -*- coding: utf-8 -*-
'''
@Time          : 2020/05/06 21:09
@Author        : Tianxiaomo
@File          : dataset.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
from torch.utils.data.dataset import Dataset

import random
import cv2
import sys


def rand_uniform_strong(min, max):
    if min > max:
        swap = min
        min = max
        max = swap
    return random.random() * (max - min) + min


def rand_scale(s):
    scale = rand_uniform_strong(1, s)
    if random.randint(0, 1) % 2:
        return scale
    return 1. / scale


def rand_precalc_random(min, max, random_part):
    if max < min:
        swap = min
        min = max
        max = swap
    return (random_part * (max - min)) + min


check_mistakes = 0


def fill_truth_detection(path, num_boxes, truth, classes, flip, dx, dy, sx, sy, net_w, net_h):
    count = 0
    boxes = read_boxes(path, & count)
    min_w_h = 0
    lowest_w = 1. / net_w
    lowest_h = 1. / net_h
    randomize_boxes(boxes, count)
    correct_boxes(boxes, count, dx, dy, sx, sy, flip)
    if count > num_boxes:
        count = num_boxes
    sub = 0

    for i in range(count):
        x = boxes[i].x
        y = boxes[i].y
        w = boxes[i].w
        h = boxes[i].h
        id = boxes[i].id

        # not detect small objects
        # if ((w < 0.001F || h < 0.001F)) continue
        # if truth (box for object) is smaller than 1x1 pix
        if id >= classes:
            print("Wrong annotation: class_id = {}. But class_id should be [from 0 to {}], file: {}".format(id, (
                    classes - 1), path))
            # sprintf(buff, "echo %s \"Wrong annotation: class_id = %d. But class_id should be [from 0 to %d]\" >> bad_label.list", labelpath, id, (classes-1))
            # system(buff)
            if check_mistakes:
                input()
            ++sub
            continue

        if w < lowest_w or h < lowest_h:
            # sprintf(buff, "echo %s \"Very small object: w < lowest_w OR h < lowest_h\" >> bad_label.list", labelpath)
            # system(buff)
            ++sub
            continue

        if x == 999999 or y == 999999:
            print("Wrong annotation: x = 0, y = 0, < 0 or > 1, file: {}".format(path))
            # sprintf(buff, "echo %s \"Wrong annotation: x = 0 or y = 0\" >> bad_label.list", labelpath)
            # system(buff)
            # ++sub
            if check_mistakes:
                input()
            continue

        if x <= 0 or x > 1 or y <= 0 or y > 1:
            print("\n Wrong annotation: x = {}, y = {}, file: {}".format(x, y, path))
            # sprintf(buff, "echo %s \"Wrong annotation: x = %f, y = %f\" >> bad_label.list", labelpath, x, y)
            # system(buff)
            # ++sub
            if check_mistakes:
                input()
            continue

        if w > 1:
            print("\n Wrong annotation: w = {}, file: {}".format(w, path))
            # sprintf(buff, "echo %s \"Wrong annotation: w = %f\" >> bad_label.list", labelpath, w)
            # system(buff)
            w = 1
            if check_mistakes:
                input()

        if h > 1:
            print("\n Wrong annotation: h = {}, file: {}".format(h, path))
            # sprintf(buff, "echo %s \"Wrong annotation: h = %f\" >> bad_label.list", labelpath, h)
            # system(buff)
            h = 1
            if check_mistakes:
                input()

        if x == 0:
            x += lowest_w
        if y == 0:
            y += lowest_h

        truth[(i - sub) * 5 + 0] = x
        truth[(i - sub) * 5 + 1] = y
        truth[(i - sub) * 5 + 2] = w
        truth[(i - sub) * 5 + 3] = h
        truth[(i - sub) * 5 + 4] = id

        if min_w_h == 0:
            min_w_h = w * net_w
        if min_w_h > w * net_w:
            min_w_h = w * net_w
        if min_w_h > h * net_h:
            min_w_h = h * net_h
    return min_w_h


def image_data_augmentation(mat, w, h, pleft, ptop, swidth, sheight, flip, dhue, dsat, dexp, gaussian_noise, blur,
                            num_boxes, truth):
    try:
        img = mat

        # crop
        src_rect = (pleft, ptop, swidth, sheight)
        img_rect = (cv2.Point2i(0, 0), img.size())
        new_src_rect = src_rect & img_rect

        dst_rect = (cv2.Point2i(max(0, -pleft), max(0, -ptop)), new_src_rect.size())
        # cv2.Mat sized

        if (src_rect.x == 0 and src_rect.y == 0 and src_rect.size() == img.size()):
            cv2.resize(img, sized, cv2.Size(w, h), 0, 0, cv2.INTER_LINEAR)
        else:
            cropped = (src_rect.size(), img.type())
            # cropped.setTo(cv2.Scalar::all(0))
            cropped.setTo(cv2.mean(img))

            img(new_src_rect).copyTo(cropped(dst_rect))

            # resize
            cv2.resize(cropped, sized, cv2.Size(w, h), 0, 0, cv2.INTER_LINEAR)

        # flip
        if flip:
            # cv2.Mat cropped
            cv2.flip(sized, cropped, 1)  # 0 - x-axis, 1 - y-axis, -1 - both axes (x & y)
            sized = cropped.clone()

        # HSV augmentation
        # cv2.COLOR_BGR2HSV, cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2BGR, cv2.COLOR_HSV2RGB
        if dsat != 1 or dexp != 1 or dhue != 0:
            if img.channels() >= 3:
                cvtColor(sized, hsv_src, cv2.COLOR_RGB2HSV)  # RGB to HSV
                cv2.split(hsv_src, hsv)

                hsv[1] *= dsat
                hsv[2] *= dexp
                hsv[0] += 179 * dhue

                cv2.merge(hsv, hsv_src)

                cvtColor(hsv_src, sized, cv2.COLOR_HSV2RGB)  # HSV to RGB (the same as previous)
            else:
                sized *= dexp

        # std::stringstream window_name
        # window_name << "augmentation - " << ipl
        # cv2.imshow(window_name.str(), sized)
        # cv2.waitKey(0)

        if blur:
            cv2.Mat
            dst(sized.size(), sized.type())
            if blur == 1:
                cv2.GaussianBlur(sized, dst, cv2.Size(17, 17), 0)
                # cv2.bilateralFilter(sized, dst, 17, 75, 75)
            else:
                ksize = (blur / 2) * 2 + 1
                kernel_size = cv2.Size(ksize, ksize)
                cv2.GaussianBlur(sized, dst, kernel_size, 0)
                # cv2.medianBlur(sized, dst, ksize)
                # cv2.bilateralFilter(sized, dst, ksize, 75, 75)

                # sharpen
                # cv2.Mat img_tmp
                # cv2.GaussianBlur(dst, img_tmp, cv2.Size(), 3)
                # cv2.addWeighted(dst, 1.5, img_tmp, -0.5, 0, img_tmp)
                # dst = img_tmp

            # std::cout << " blur num_boxes = " << num_boxes << std::endl

            if blur == 1:
                cv2.Rect
                img_rect(0, 0, sized.cols, sized.rows)
                for (t = 0 t < num_boxes ++t):
                    b = float_to_box_stride(truth + t * (4 + 1), 1)
                    if not b.x:
                        break
                    left = (b.x - b.w / 2.) * sized.cols
                    width = b.w * sized.cols
                    top = (b.y - b.h / 2.) * sized.rows
                    height = b.h * sized.rows
                    roi(left, top, width, height)
                    roi = roi & img_rect

                    sized(roi).copyTo(dst(roi))

            dst.copyTo(sized)

        if gaussian_noise:
            noise = cv2.Mat(sized.size(), sized.type())
            gaussian_noise = min(gaussian_noise, 127)
            gaussian_noise = max(gaussian_noise, 0)
            cv2.randn(noise, 0, gaussian_noise)  # mean and variance
            sized_norm = sized + noise
            # cv2.normalize(sized_norm, sized_norm, 0.0, 255.0, cv2.NORM_MINMAX, sized.type())
            # cv2.imshow("source", sized)
            # cv2.imshow("gaussian noise", sized_norm)
            # cv2.waitKey(0)
            sized = sized_norm

        # char txt[100]
        # sprintf(txt, "blur = %d", blur)
        # cv2.putText(sized, txt, cv2.Point(100, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.7, CV_RGB(255, 0, 0), 1, CV_AA)

        # Mat -> image
        out = mat_to_image(sized)
    except:
        print("OpenCV can't augment image: " + str(w) + " x " + str(h))
        out = mat_to_image(mat)

    return out


# blend two images with (alpha and beta)
def blend_images_cv(new_img, alpha, old_img, beta):
    new_mat = (cv::Size(new_img.w, new_img.h), CV_32FC(new_img.c), new_img.data)  # , size_t step = AUTO_STEP)
    old_mat = (cv::Size(old_img.w, old_img.h), CV_32FC(old_img.c), old_img.data)
    cv2.addWeighted(new_mat, alpha, old_mat, beta, 0.0, new_mat)
    return new_mat


def blend_truth(new_truth, boxes, old_truth):
    t_size = 4 + 1
    count_new_truth = 0
    for (t = 0 t < boxes ++t):
        float
        x = new_truth[t * (4 + 1)]
        if not x:
            break
        count_new_truth += 1

    for (t = count_new_truth t < boxes ++t):
        new_truth_ptr = new_truth + t * t_size
        old_truth_ptr = old_truth + (t - count_new_truth) * t_size
        x = old_truth_ptr[0]
        if not x:
            break

        new_truth_ptr[0] = old_truth_ptr[0]
        new_truth_ptr[1] = old_truth_ptr[1]
        new_truth_ptr[2] = old_truth_ptr[2]
        new_truth_ptr[3] = old_truth_ptr[3]
        new_truth_ptr[4] = old_truth_ptr[4]
    return new_truth_ptr


def min_val_cmp(a, b):
    return a if a < b else b


def max_val_cmp(a, b):
    return a if a > b else b


def blend_truth_mosaic(new_truth, boxes, old_truth, w, h, cut_x, cut_y, i_mixup, left_shift, right_shift, top_shift,
                       bot_shift):
    t_size = 4 + 1
    count_new_truth = 0
    for (t = 0 t < boxes ++t):
        x = new_truth[t * (4 + 1)]
        if not x:
            break
        count_new_truth + +

    new_t = count_new_truth
    for (t = count_new_truth t < boxes ++t):
        *new_truth_ptr = new_truth + new_t * t_size
        new_truth_ptr[0] = 0
        *old_truth_ptr = old_truth + (t - count_new_truth) * t_size
        x = old_truth_ptr[0]
        if not x:
            break

        xb = old_truth_ptr[0]
        yb = old_truth_ptr[1]
        wb = old_truth_ptr[2]
        hb = old_truth_ptr[3]

        # shift 4 images
        if i_mixup == 0:
            xb = xb - (float)(w - cut_x - right_shift) / w
            yb = yb - (float)(h - cut_y - bot_shift) / h

        if i_mixup == 1:
            xb = xb + (float)(cut_x - left_shift) / w
            yb = yb - (float)(h - cut_y - bot_shift) / h

        if i_mixup == 2:
            xb = xb - (float)(w - cut_x - right_shift) / w
            yb = yb + (float)(cut_y - top_shift) / h

        if i_mixup == 3:
            xb = xb + (float)(cut_x - left_shift) / w
            yb = yb + (float)(cut_y - top_shift) / h

        left = (xb - wb / 2) * w
        right = (xb + wb / 2) * w
        top = (yb - hb / 2) * h
        bot = (yb + hb / 2) * h

        # fix out of bound
        if left < 0:
            diff = left / w
            xb = xb - diff / 2
            wb = wb + diff

        if right > w:
            diff = (right - w) / w
            xb = xb - diff / 2
            wb = wb - diff

        if top < 0:
            diff = (float)
            top / h
            yb = yb - diff / 2
            hb = hb + diff

        if bot > h:
            diff = (float)(bot - h) / h
            yb = yb - diff / 2
            hb = hb - diff

        left = (xb - wb / 2) * w
        right = (xb + wb / 2) * w
        top = (yb - hb / 2) * h
        bot = (yb + hb / 2) * h

        # leave only within the image
        if left >= 0 and right <= w and top >= 0 and bot <= h and wb > 0 and wb < 1 and hb > 0 and hb < 1 and xb > 0 and xb < 1 and yb > 0 and yb < 1:
            new_truth_ptr[0] = xb
            new_truth_ptr[1] = yb
            new_truth_ptr[2] = wb
            new_truth_ptr[3] = hb
            new_truth_ptr[4] = old_truth_ptr[4]
            new_t += 1
    return new_truth_ptr


class Yolo_dataset(Dataset):
    def __init__(self, cfg):
        super(Yolo_dataset, self).__init__()
        if cfg.mixup == 2:
            print("cutmix=1 - isn't supported for Detector")
            raise
        elif cfg.mixup == 2 and cfg.letter_box:
            print("Combination: letter_box=1 & mosaic=1 - isn't supported, use only 1 of these parameters")
            raise

        self.cfg = cfg

    def __len__(self):
        return 100

    def __getitem__(self, index):
        img_path = self.img_list[index]
        use_mixup = self.cfg.mixup
        if random.randint(0, 1):
            use_mixup = 0

        r1, r2, r3, r4, r_scale = 0, 0, 0, 0, 0
        dhue, dsat, dexp, flip, blur = 0, 0, 0, 0, 0
        augmentation_calculated, gaussian_noise = 0, 0

        for i in range(use_mixup + 1):
            if i:
                augmentation_calculated = 0
            img = cv2.imread(img_path)
            if img is None:
                continue
            oh, ow, oc = img.shape
            dh, dw, dc = [oh, ow, oc] * self.cfg.jitter

            if augmentation_calculated == 0 or self.cfg.track == 0:
                augmentation_calculated = 1
                r1 = random.random()
                r2 = random.random()
                r3 = random.random()
                r4 = random.random()

                r_scale = random.random()

                dhue = rand_uniform_strong(-self.cfg.hue, self.cfg.hue)
                dsat = rand_scale(self.cfg.saturation)
                dexp = rand_scale(self.cfg.exposure)

                flip = random.randint(0, 1) % 2 if self.cfg.flip else 0

                if (self.cfg.blur):
                    tmp_blur = random.randint(0, 2)  # 0 - disable, 1 - blur background, 2 - blur the whole image
                    if tmp_blur == 0:
                        blur = 0
                    elif tmp_blur == 1:
                        blur = 1
                    else:
                        blur = self.cfg.blur

                if self.cfg.gaussian and random.randint(0, 1):
                    gaussian_noise = self.cfg.gaussian
                else:
                    gaussian_noise = 0

            pleft = rand_precalc_random(-dw, dw, r1)
            pright = rand_precalc_random(-dw, dw, r2)
            ptop = rand_precalc_random(-dh, dh, r3)
            pbot = rand_precalc_random(-dh, dh, r4)

            if self.cfg.letter_box:
                img_ar = ow / oh
                net_ar = self.cfg.w / self.cfg.h
                result_ar = img_ar / net_ar
                # print(" ow = %d, oh = %d, w = %d, h = %d, img_ar = %f, net_ar = %f, result_ar = %f \n", ow, oh, w, h, img_ar, net_ar, result_ar);
                if result_ar > 1:  # sheight - should be increased
                    oh_tmp = ow / net_ar
                    delta_h = (oh_tmp - oh) / 2
                    ptop = ptop - delta_h
                    pbot = pbot - delta_h
                    # print(" result_ar = %f, oh_tmp = %f, delta_h = %d, ptop = %f, pbot = %f \n", result_ar, oh_tmp, delta_h, ptop, pbot);
                else:  # swidth - should be increased
                    ow_tmp = oh * net_ar
                    delta_w = (ow_tmp - ow) / 2
                    pleft = pleft - delta_w
                    pright = pright - delta_w
                    # printf(" result_ar = %f, ow_tmp = %f, delta_w = %d, pleft = %f, pright = %f \n", result_ar, ow_tmp, delta_w, pleft, pright);

            swidth = ow - pleft - pright
            sheight = oh - ptop - pbot

            sx = swidth / ow
            sy = sheight / oh

            dx = (pleft / ow) / sx
            dy = (ptop / oh) / sy

            min_w_h = fill_truth_detection(filename, self.cfg.boxes, truth, self.cfg.classes, flip, dx, dy, 1. / sx,
                                           1. / sy, self.cfg.w, self.cfg.h)
            if (min_w_h / 8) < blur and blur > 1:  # disable blur if one of the objects is too small
                blur = min_w_h / 8

            ai, truth = image_data_augmentation(img, self.cfg.w, self.cfg.h, pleft, ptop, swidth, sheight, flip,
                                                dhue,
                                                dsat,
                                                dexp,
                                                gaussian_noise, blur, self.cfg.boxes, truth)

            # if use_mixup == 0:
            #     d.X.vals[i] = ai.data
            #     memcpy(d.y.vals[i], truth, 5 * boxes * sizeof(float))
            if use_mixup == 1:
                if i == 0:
                    d.X.vals[i] = ai.data
                    old_img = ai.copy()
                    old_truth = truth.copy()
                elif i == 1:
                    blend_images_cv(ai, 0.5, old_img, 0.5)
                    blend_truth(old_truth, self.cfg.boxes, truth)
            elif use_mixup == 3:
                if i == 0:
                    # tmp_img = make_image(w, h, c)
                    d.X.vals[i] = tmp_img.data
                if flip:
                    tmp = pleft
                    pleft = pright
                    pright = tmp

                left_shift = min_val_cmp(cut_x[i], max_val_cmp(0, (-pleft * w / ow)))
                top_shift = min_val_cmp(cut_y[i], max_val_cmp(0, (-ptop * h / oh)))

                right_shift = min_val_cmp((w - cut_x[i]), max_val_cmp(0, (-pright * w / ow)))
                bot_shift = min_val_cmp(h - cut_y[i], max_val_cmp(0, (-pbot * h / oh)))

                for k in range(self.cfg.c):
                    for y in range(self.cfg.h):
                        j = y * self.cfg.w + k * self.cfg.w * self.cfg.h
                        if i == 0 and y < cut_y[i]:
                            j_src = (w - cut_x[i] - right_shift) + (y + h - cut_y[i] - bot_shift) * w + k * w * h
                        if i == 1 and y < cut_y[i]:
                            j_src = left_shift + (y + h - cut_y[i] - bot_shift) * w + k * w * h
                        if i == 2 and y >= cut_y[i]:
                            j_src = (w - cut_x[i] - right_shift) + (top_shift + y - cut_y[i]) * w + k * w * h
                        if i == 3 and y >= cut_y[i]:
                            j_src = left_shift + (top_shift + y - cut_y[i]) * w + k * w * h

                blend_truth_mosaic(d.y.vals[i], boxes, truth, w, h, cut_x[i], cut_y[i], i_mixup, left_shift,
                                   right_shift, top_shift, bot_shift)
