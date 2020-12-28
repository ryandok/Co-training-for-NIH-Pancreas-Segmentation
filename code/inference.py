# external imports
import torch
import torch.nn.functional as F
import os
import argparse
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader
from torchvision import transforms


# internal imports
from utils import util
from utils import metrics
from dataloaders.PancreasMultiView import Pancreas, Rotate, ToTensor, CenterCrop
from networks.UNet3D import UNet3D
from valid import prob2img, save_volume


# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument("--gpu",
                    type=str,
                    default='5,6,7',
                    help="gpu id")

parser.add_argument("--iter_from",
                    type=int,
                    dest="iter_from",
                    default=24000,
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


def inference(model_multi_views, volume_input_views, T=10):
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
    fusion_label = torch.argmax(fusion_label, dim=1)
    return fusion_label


def validate(valid_dataset, model_multi_views, save_dir):
    label_list = [0, 1]

    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    final_score = []

    for valid_index, valid_batch_sample in enumerate(valid_dataloader):

        volume_input_views = []
        for i, sample_batch in enumerate(valid_batch_sample):
            volume_input_views.append(sample_batch['volume'].to("cuda:%d" % i).float())
        valid_seg = valid_batch_sample[0]['label'].to("cuda:0").float()


        valid_name = valid_batch_sample[0]['name']

        # image_shape = (valid_seg.shape[2], valid_seg.shape[3], valid_seg.shape[4])

        # Increase the dimension
        # valid_input = torch.unsqueeze(valid_input, 1)
        # valid_seg = valid_seg.cpu().detach().numpy()
        # valid_seg = np.squeeze(valid_seg, 1)

        pred_seg = inference(model_multi_views, volume_input_views)
        pred_seg = pred_seg.cpu().detach().numpy()
        valid_seg = valid_seg.cpu().detach().numpy()
        valid_seg = np.squeeze(valid_seg, axis=1)

        # calculate dice
        dice_dict, average_dice = metrics.dice(pred_seg, valid_seg, label_list)
        print("validating %2d/%d, valid_name: %-10s, final score: %f"
              % (valid_index+1, len(valid_dataset), valid_name, average_dice))
        # print(dice_dict)

        fuse_seg = util.fuse_pred_label(valid_seg, pred_seg)

        save_pred_name = 'pred' + valid_name[0].split('img')[-1] + '.nii.gz'
        save_fuse_name = 'fuse' + valid_name[0].split('img')[-1] + '.nii.gz'
        # save_dir = '../data/UNet3D_pred/' + arg.model_path.split('UNet3D/')[-1] + '/20500/view2/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_volume(pred_seg, save_dir, save_pred_name)
        save_volume(fuse_seg, save_dir, save_fuse_name)

        final_score.append(average_dice)

    valid_average_score = np.average(final_score)
    print("validating done")
    print('validation score :%f' % valid_average_score)

    return


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

    valid_dataset = Pancreas(valid_volume_path, valid_label_path,
                                    transform_views=[transforms.Compose([CenterCrop((128,128,128)), Rotate((0,0)),   ToTensor()]),
                                                     transforms.Compose([CenterCrop((128,128,128)), Rotate((0,2)),  ToTensor()]),
                                                     transforms.Compose([CenterCrop((128,128,128)), Rotate((1,2)),  ToTensor()])])


    save_dir = '../data/UNet3D_pred_views/' + arg.model_path.split('UNet3D/')[-1] + \
               '/' + str(arg.iter_from) + '/view%d/' % (i + 1)
    validate(valid_dataset, model_multi_views, save_dir)

