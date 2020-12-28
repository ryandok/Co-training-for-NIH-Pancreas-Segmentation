# external imports
import torch
import torch.nn.functional as nnf
import os
import argparse
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader
from torchvision import transforms
from medpy import metric

# internal imports
from utils import util
from utils import metrics
from dataloaders.Pancreas import Pancreas
from dataloaders.Pancreas import Rotate, RandomNoise, ToTensor, Flip, CenterCrop
from networks.UNet3D import UNet3D
# from networks.UNet3D_v2 import UNet3D

# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument("--gpu",
                    type=str,
                    default='6',
                    help="gpu id")

parser.add_argument("--option",
                    type=str,
                    default='baseline',
                    choices=['baseline', 'upper bound', 'multi-views'],
                    help="option for baseline/ upper bound/ multi-views")

parser.add_argument("--iter_from",
                    type=int,
                    dest="iter_from",
                    default=21000,
                    help="iteration number to start training from")
# baseline and upper bound (without dropout)
# 30000  20% labeled 69.5005%
# 15000 100% labeled 79.6547%

# baseline and upper bound (with dropout)
# 30000  20% labeled 68.4014%
# 15000 100% labeled 81.7602%

# multi-views
# view-wise batch size = 1
# co training batch size = 2
# 20000 view-wise   : 62.1490% 64.9011% 65.0780%
# 40000 co-training : 69.6453% 70.1073% 69.1510%

# multi-views
# view-wise batch size = 3
# co training batch size = 3(2 for labeled, 1 for unlabeled)
# 5000 view-wise    : 69.8117% 68.5434% 65.6873%
# 8000 co-training  : 73.8309% 72.2007% 70.7330%


parser.add_argument("--data_path",
                    type=str,
                    dest="data_path",
                    default='/home/hra/dataset/Pancreas/Pancreas_region',
                    help="data path")

parser.add_argument("--model_path",
                    type=str,
                    dest="model_path",
                    # default='../model/Pancreas_model/UNet3D/2020-11-24/14:50:36/',
                    # default='../model/Pancreas_model/UNet3D/2020-11-28/12:15:46/',
                    # default='../model/Pancreas_model/UNet3D/2020-11-28/16:19:54/',
                    # default='../model/Pancreas_model/UNet3D/2020-11-30/15:08:33/',
                    # default='../model/Pancreas_model/UNet3D/2020-12-01/11:09:09/',
                    # default='../model/Pancreas_model/UNet3D/2020-12-01/11:09:53/',
                    # default='../model/Pancreas_model/UNet3D/2020-12-01/15:32:50/',  # 2000 view1
                    # default='../model/Pancreas_model/UNet3D/2020-12-01/14:36:32/',  # 2500 view2
                    # default='../model/Pancreas_model/UNet3D/2020-12-01/14:36:58/',  # 2500 view3
                    # default='../model/Pancreas_model/UNet3D/2020-12-10/22:05:10',
                    # default='../model/Pancreas_model/UNet3D/2020-12-10/14:32:26',
                    # default='../model/Pancreas_model/UNet3D/2020-12-11/14:18:03',
                    # default='../model/Pancreas_model/UNet3D/2020-12-11/14:59:42',
                    # default='../model/Pancreas_model/UNet3D/2020-12-11/15:59:03', # view1
                    # default='../model/Pancreas_model/UNet3D/2020-12-12/11:13:16',
                    default='../model/Pancreas_model/UNet3D/2020-12-12/11:13:35',

                    # default='../model/Pancreas_model/UNet3D/2020-11-24/14:50:41/',
                    # default='../model/Pancreas_model/UNet3D/2020-11-28/12:16:04/',
                    # default='../model/Pancreas_model/UNet3D/2020-11-30/15:08:35/',
                    # default='../model/Pancreas_model/UNet3D/2020-12-01/16:41:31',
                    # default='../model/Pancreas_model/UNet3D/2020-12-02/10:47:19',
                    # default='../model/Pancreas_model/UNet3D/2020-12-10/14:32:28',
                    # default='../model/Pancreas_model/UNet3D/2020-12-10/22:04:14',
                    # default='../model/Pancreas_model/UNet3D/2020-12-11/17:35:38',


                    # default='../model/Pancreas_model/UNet3D/2020-11-23/18:49:58/',
                    # default='../model/Pancreas_model/UNet3D/2020-11-27/18:14:53',
                    # default='../model/Pancreas_model/UNet3D/2020-11-28/11:56:44',
                    # default='../model/Pancreas_model/UNet3D/2020-11-28/16:10:19',
                    # default='../model/Pancreas_model/UNet3D/2020-11-30/11:26:44',
                    # default='../model/Pancreas_model/UNet3D/2020-11-30/21:30:56',
                    # default='../model/Pancreas_model/UNet3D/2020-12-01/16:55:59/',
                    # default='../model/Pancreas_model/UNet3D/2020-12-01/20:32:18/',
                    help="Path to model")

arg = parser.parse_args()

def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    # jc = metric.binary.jc(pred, gt)
    # hd = metric.binary.hd95(pred, gt)
    # asd = metric.binary.asd(pred, gt)
    jc = 0
    hd = 0
    asd = 0

    return dice, jc, hd, asd

def prob2img(pred, output_shape, label_list):
    """
    turn possibility to image
    :param pred: model' output -- prediction
    :param output_shape: image shape
    :param label_list: include 0
    :return:
    """

    pred = pred.cpu().detach().numpy()
    # output2 = pred.argmax(axis=1) 也可以用这个输出，更简单一些

    pred = np.squeeze(pred)
    output = np.zeros(output_shape)
    max = np.max(pred, axis=0)
    for label_index, label_value in enumerate(label_list):
        label_seg = np.array(pred[label_index] == max).astype(int) * label_value
        output += label_seg
    output = output[np.newaxis, ...]
    return output


def save_volume(pred_array, save_dir, name):
    pred_array = np.squeeze(pred_array)
    pseudo_seg_image = nib.Nifti1Image(pred_array, np.eye(4))
    nib.save(pseudo_seg_image, os.path.join(save_dir, name))
    print('%-20s saving done' % name)


def validate(valid_dataset, model, save_dir):
    label_list = [0, 1]
    model.eval()

    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    final_score = []
    total_metric = 0.0

    for valid_index, valid_sample in enumerate(valid_dataloader):

        valid_name = valid_sample['name']
        valid_input = valid_sample['volume'].to('cuda').float()
        valid_seg = valid_sample['label'].to('cuda').float()

        image_shape = (valid_seg.shape[2], valid_seg.shape[3], valid_seg.shape[4])

        # Increase the dimension
        # valid_input = torch.unsqueeze(valid_input, 1)
        valid_seg = valid_seg.cpu().detach().numpy()
        valid_seg = np.squeeze(valid_seg, 1)

        pred = model(valid_input)
        pred_softmax = nnf.softmax(pred, dim=1)
        pred_seg = prob2img(pred_softmax, image_shape, label_list)

        # calculate result
        single_metric = calculate_metric_percase(pred_seg, valid_seg)
        total_metric += np.asarray(single_metric)

        # calculate dice
        # dice_dict, average_dice = metrics.dice(pred_seg, valid_seg, label_list)
        print("validating %2d/%d, valid_name: %-10s, Dice: %.5f, Jaccard: %.5f, 95HD: %.5f, ASD: %.5f"
              % (valid_index+1, len(valid_dataset), valid_name,
                 single_metric[0], single_metric[1], single_metric[2], single_metric[3]))
        # print(dice_dict)

        fuse_seg = util.fuse_pred_label(valid_seg, pred_seg)

        save_pred_name = 'pred' + valid_name[0].split('img')[-1] + '.nii.gz'
        save_fuse_name = 'fuse' + valid_name[0].split('img')[-1] + '.nii.gz'
        save_img_name = valid_name[0] + '.nii.gz'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        input_volume = valid_input.cpu().detach().numpy()
        save_volume(input_volume, save_dir, save_img_name)
        save_volume(pred_seg, save_dir, save_pred_name)
        save_volume(fuse_seg, save_dir, save_fuse_name)

        # final_score.append(average_dice)

    # valid_average_score = np.average(final_score)
    print("validating done")
    # print('validation score :%f' % valid_average_score)
    avg_metric = total_metric / len(valid_dataloader)
    print('average metric is Dice: %.5f, Jaccard: %.5f, 95HD: %.5f, ASD: %.5f'
          % (avg_metric[0], avg_metric[1], avg_metric[2], avg_metric[3]))

    return


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu
    # load all data path
    total_volume_path, total_label_path = util.gen_Pancreas_data_path(arg.data_path)
    total_volume_path = [i.replace('region', 'region2') for i in total_volume_path]
    total_label_path = [i.replace('region', 'region2') for i in total_label_path]

    train_volume_path, train_label_path, valid_volume_path, valid_label_path \
        = util.divide_data2train_valid(total_volume_path, total_label_path, 62, 1997)

    option = arg.option
    valid_options = ['baseline', 'upper bound', 'multi-views']
    if option == valid_options[0]:
        modelS = UNet3D(1, 2).cuda()
        valid_dataset = Pancreas(valid_volume_path, valid_label_path,
                                 transform=transforms.Compose([CenterCrop((128, 128, 128)), Rotate((1, 2)), ToTensor()]))
        # valid_dataset = Pancreas(valid_volume_path, valid_label_path, transform=transforms.Compose([Rotate((0,2)), ToTensor()]))
        # valid_dataset = Pancreas(valid_volume_path, valid_label_path, transform=transforms.Compose([Rotate((1,2)), ToTensor()]))
        # train_dataset = Pancreas(train_volume_path[0:12], train_label_path[0:12], transform=transforms.Compose([Rotate((0,2)), ToTensor()]))
        modelS.load_state_dict(torch.load("%s/modelS_%d.ckpt" % (arg.model_path, arg.iter_from)))
        save_dir = '../data/UNet3D_pred_baseline/' + arg.model_path.split('UNet3D/')[-1] + \
                   '/' + str(arg.iter_from) + '/'
        print('\n\n baseline testing .........')
        validate(valid_dataset, modelS, save_dir)
        # validate(train_dataset, modelS, save_dir)

    elif option == valid_options[1]:
        modelS = UNet3D(1, 2).cuda()
        valid_dataset = Pancreas(valid_volume_path, valid_label_path,
                                 transform=transforms.Compose([CenterCrop((128, 128, 128)), ToTensor()]))
        # train_dataset = Pancreas(train_volume_path, train_label_path, transform=transforms.Compose([ToTensor()]))
        modelS.load_state_dict(torch.load("%s/modelS_%d.ckpt" % (arg.model_path, arg.iter_from)))
        save_dir = '../data/UNet3D_pred_upper/' + arg.model_path.split('UNet3D/')[-1] + \
                   '/' + str(arg.iter_from) + '/'
        print('\n\n upper bound testing .........')
        validate(valid_dataset, modelS, save_dir)
        # validate(train_dataset, modelS, save_dir)

    elif option == valid_options[2]:
        modelS = UNet3D(1, 2).cuda()
        modelS.eval()
        rotate_list = [Rotate((0,0)), Rotate((0,2)), Rotate((1,2))]
        for i in range(3):
            valid_dataset = Pancreas(valid_volume_path, valid_label_path,
                                     transform=transforms.Compose([rotate_list[i], ToTensor()]))

            modelS.load_state_dict(torch.load("%s/model_view%d_%d.ckpt" % (arg.model_path, i + 1, arg.iter_from),
                                              map_location={'cuda:%d' % i: 'cuda:0'}))
            save_dir = '../data/UNet3D_pred_views/' + arg.model_path.split('UNet3D/')[-1] + \
                       '/' + str(arg.iter_from) + '/view%d/' % (i + 1)
            print('\n\n view %d testing .........' % (i + 1))
            validate(valid_dataset, modelS, save_dir)

    else:
        print('option wrong')




