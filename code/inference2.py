# external imports
import torch
import torch.nn.functional as F
import os
import argparse
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from medpy import metric
import math

# internal imports
from utils import util
from utils import metrics
from dataloaders.PancreasMultiView import Pancreas, Rotate, ToTensor, CenterCrop
from networks.UNet3D import UNet3D
from valid import prob2img, save_volume
from dataloaders import Pancreas_processing

# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument("--gpu",
                    type=str,
                    default='5,6,7',
                    help="gpu id")

parser.add_argument("--iter_from",
                    type=int,
                    dest="iter_from",
                    default=0,
                    help="iteration number to start training from")

parser.add_argument("--data_path",
                    type=str,
                    dest="data_path",
                    default='/home/hra/dataset/Pancreas/Pancreas_region',
                    help="data path")

parser.add_argument("--model_path",
                    type=str,
                    dest="model_path",
                    # default='../model/Pancreas_model/UNet3D/2020-11-30/11:26:44',
                    # default='../model/Pancreas_model/UNet3D/2020-11-30/21:30:56',
                    # default='../model/Pancreas_model/UNet3D/2020-12-01/16:55:59/',
                    # default='../model/Pancreas_model/UNet3D/2020-12-01/20:32:18/',
                    default='../model/Pancreas_model/UNet3D/2020-12-13/15:40:38/',
                    help="Path to model")

arg = parser.parse_args()


def test_all_case(model_multi_views, image_list, num_classes, patch_size=(128, 128, 128), stride_xy=16, stride_z=16, save_result=True,
                  test_save_path=None, preproc_fn=None, metric_detail=0):
    total_metric = 0.0
    loader = tqdm(image_list) if not metric_detail else image_list
    ith = 0
    for image_path in loader:
        # id = image_path.split('/')[-2]
        # h5f = h5py.File(image_path, 'r')
        # image = h5f['image'][:]
        # label = h5f['label'][:]
        image = nib.load(image_path).get_fdata()
        label_path = image_path.replace('img', 'label')
        label = nib.load(label_path).get_fdata()
        name = os.path.basename(image_path).split('img')[-1].split('.')[0]

        if preproc_fn is not None:
            image = preproc_fn(image)
        # util.save_volume(image, '../data', 'demo_image.nii.gz')

        prediction, score_map = test_single_case(model_multi_views, image, stride_xy, stride_z, patch_size, num_classes=num_classes)

        if np.sum(prediction)==0:
            single_metric = (0,0,0,0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:])
        if metric_detail:
            print('%s,\t%.5f, %.5f, %.5f, %.5f' % (name, single_metric[0], single_metric[1], single_metric[2], single_metric[3]))


        total_metric += np.asarray(single_metric)

        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path + "%spred.nii.gz" % name)
            nib.save(nib.Nifti1Image(image.astype(np.float32), np.eye(4)), test_save_path + "%simg.nii.gz" % name)
            nib.save(nib.Nifti1Image(label.astype(np.float32), np.eye(4)), test_save_path + "%sgt.nii.gz" % name)
            # fuse_seg = util.fuse_pred_label(prediction, label)
            # nib.save(nib.Nifti1Image(fuse_seg.astype(np.float32), np.eye(4)), test_save_path + "%sfuse.nii.gz" % name)
        ith += 1

    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))

    return avg_metric


def test_single_case(model_multi_views, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in tqdm(range(0, sx)):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy*y, hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z*z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                volume_input_views = [torch.from_numpy(test_patch).to("cuda:%d" % i).float() for i in range(len(model_multi_views))]
                for i in range(len(model_multi_views)):
                    volume_input_views[i] = torch.unsqueeze(volume_input_views[i], dim=0)
                    volume_input_views[i] = torch.unsqueeze(volume_input_views[i], dim=1)
                    if i != 0:
                        volume_input_views[i] = volume_input_views[i].rot90(dims=(i+1, 4))


                with torch.no_grad():
                    y1 = inference_fuse(model_multi_views, volume_input_views)
                    # ensemble
                    y = F.softmax(y1, dim=1)

                y = y.cpu().data.numpy()
                y = y[0,:,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd


def inference_fuse(model_multi_views, volume_input_views, T=10):
    view_num = len(model_multi_views)

    # run 3D U-Net model on unlabeled data with multi-views
    pred_views = [model_multi_views[i](volume_input_views[i]) for i in range(view_num)]
    pred_softmax_views = [F.softmax(pred_views[i], dim=1) for i in range(view_num)]

    # run 3D U-Net bayesian model on input data with multi-views
    volume_input_repeat_views = [volume_input_views[i].repeat(2, 1, 1, 1, 1) for i in range(view_num)]
    stride = volume_input_repeat_views[0].shape[0] // 2
    pred_bayes_views = [torch.zeros([stride * T, 2, 128, 128, 128]).to('cuda:%d' % i) for i in range(view_num)]
    for t in range(T // 2):
        noise_input_volume_views = [volume_input_repeat_views[i] +
                                        torch.clamp(torch.rand_like(volume_input_repeat_views[i]) * 0.1,
                                                    -0.2, 0.2).to('cuda:%d' % i) for i in range(view_num)]
        with torch.no_grad():
            for i in range(view_num):
                pred_bayes_views[i][2 * stride * t:2 * stride * (t + 1)] = model_multi_views[i](
                    noise_input_volume_views[i])

    pred_bayes_views = [F.softmax(pred_bayes_views[i], dim=1) for i in range(view_num)]
    pred_bayes_views = [pred_bayes_views[i].reshape(T, stride, 2, 128, 128, 128) for i in range(view_num)]
    pred_bayes_views = [torch.mean(pred_bayes_views[i], dim=0) for i in range(view_num)]  # [view_num * (bs, 2, 128, 128, 128)]
    uncertainty_views = [
        -1.0 * torch.sum(pred_bayes_views[i] * torch.log(pred_bayes_views[i] + 1e-6),
                         dim=1, keepdim=True) for i in range(view_num)]  # [view_num * (bs, 1, 128, 128, 128)]

    # turn uncertainty and pred into the same view
    rotate_axes = [(2, 4), (3, 4)]
    for i in range(view_num):
        if i == 0:
            continue
        else:
            pred_softmax_views[i] = pred_softmax_views[i].rot90(dims=rotate_axes[i-1], k=-1)
            uncertainty_views[i] = uncertainty_views[i].rot90(dims=rotate_axes[i-1], k=-1)

    # inference fusion
    top = torch.zeros_like(pred_softmax_views[i]).to('cuda:%d' % i)
    down = torch.zeros_like(uncertainty_views[i]).to('cuda:%d' % i)
    for j in range(view_num):
        top += (pred_softmax_views[j] / uncertainty_views[j]).to('cuda:%d' % i)
        down += 1 / uncertainty_views[j].to('cuda:%d' % i)
    fusion_label = top / down
    # fusion_label = torch.argmax(fusion_label, dim=1)
    return fusion_label


def preproc_fn(image):
    image = Pancreas_processing.CT_window(image)
    image = Pancreas_processing.norm_0mean_1var(image)
    return image


def test_calculate_metric(model_multi_views, test_save_path, preproc_fn):
    avg_metric = test_all_case(model_multi_views, valid_volume_path, num_classes=2,
                           patch_size=(128, 128, 128), stride_xy=10, stride_z=10,
                           save_result=True, test_save_path=test_save_path,
                           metric_detail=True, preproc_fn=preproc_fn)

    return avg_metric


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu
    # load all data path
    total_volume_path, total_label_path = util.gen_Pancreas_data_path(arg.data_path)

    train_volume_path, train_label_path, valid_volume_path, valid_label_path \
        = util.divide_data2train_valid(total_volume_path, total_label_path, 62, 1997)

    valid_volume_path = [i.replace('region', 'region2') for i in valid_volume_path]
    valid_label_path = [i.replace('region', 'region2') for i in valid_label_path]

    model_multi_views = []
    for i in range(3):
        model_view = UNet3D(1, 2, has_dropout=True).to("cuda:%d" % i)
        model_multi_views.append(model_view)

    for i in range(3):
        model_multi_views[i].load_state_dict(torch.load("%s/model_view%d_%d.ckpt" % (arg.model_path, i + 1, arg.iter_from)))
        model_multi_views[i].eval()

    test_save_path = '../data/UNet3D_pred_views/' + arg.model_path.split('UNet3D/')[-1] + \
               '/' + str(arg.iter_from) + '/fused/'
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
    avg_metric = test_calculate_metric(model_multi_views, test_save_path, None)

    print(avg_metric)


