"""
Training multi-view co-training 3D U-Net
"""
# TODO: dataloader
# TODO: two model
# TODO: loss
# TODO: updata
# TODO: two gpu parallel
# external imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
import sys
import time
import datetime
import codecs
import logging
import random
from tensorboardX import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader

# internal imports
from utils import losses
from utils import util
import valid
from utils.losses import dice_loss
from dataloaders.PancreasMultiView import Pancreas, Rotate, RandomNoise, ToTensor, Flip, TwoStreamBatchSampler
from networks.UNet3D import UNet3D


# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument("--gpu",
                    type=str,
                    default='0,1,4',
                    help="gpu id")

parser.add_argument("--batch_size",
                    type=int,
                    default='3',
                    help="batch size per gpu")

parser.add_argument("--labeled_bs",
                    type=int,
                    default=2,
                    help="labeled batch size per gpu")

parser.add_argument("--view_num",
                    type=int,
                    default=3,
                    help="number of views")

parser.add_argument("--n_iter_view_wise",
                    type=int,
                    default=20000,
                    help="number of iterations for view-wise training")

parser.add_argument("--seed1",
                    type=int,
                    default='1997',
                    help="seed 1")

parser.add_argument("--seed2",
                    type=int,
                    default='1124',
                    help="seed 1")

parser.add_argument("--iter_from",
                    type=int,
                    dest="iter_from",
                    default=0,
                    help="iteration number to start training from")

parser.add_argument("--n_total_iter_from",
                    type=int,
                    dest="n_total_iter_from",
                    default=0,
                    help="iteration number(used for continuing training)")

parser.add_argument("--n_epochs",
                    type=int,
                    dest="n_epochs",
                    default=10000,
                    help="number of epochs of training")

parser.add_argument("--lr_stage1",
                    type=float,
                    dest="lr_stage1",
                    default=7e-3,
                    help="SGD: learning rate for view-wise traning")

parser.add_argument("--lr_stage2",
                    type=float,
                    dest="lr_stage2",
                    default=1e-3,
                    help="SGD: learning rate for co-training")

parser.add_argument("--n_save_iter",
                    type=int,
                    dest="n_save_iter",
                    default=100,
                    help="Save the model every time")

parser.add_argument("--data_path",
                    type=str,
                    dest="data_path",
                    default='/home/hra/dataset/Pancreas/Pancreas_region',
                    help="data path")

parser.add_argument("--model_dir_root_path",
                    type=str,
                    dest="model_dir_root_path",
                    default='../model/Pancreas_model/UNet3D/',
                    help="root path to save the UNet3D model")

parser.add_argument("--model_pre_trained_dir",
                    type=str,
                    dest="model_pre_trained_dir",
                    # default='../model/Pancreas_model/UNet3D/2020-11-22/16:54:40/',
                    default=None,
                    help="Path to pre-trained model")

parser.add_argument("--note",
                    type=str,
                    dest="note",
                    default="multi view on 3D U-Net",
                    # default=None,
                    help="note")

arg = parser.parse_args()


def train(gpu,
          batch_size,
          labeled_bs,
          view_num,
          n_iter_view_wise,
          seed1,
          seed2,
          iter_from,
          n_total_iter_from,
          n_epochs,
          lr_stage1,
          lr_stage2,
          n_save_iter,
          data_path,
          model_dir_root_path,
          model_pre_trained_dir,
          note):
    """
    Training 3D U-Net
    :param gpu: gpu id
    :param batch_size: batch size
    :param labeled_bs: labeled batch size
    :param view_num: number of views
    :param n_iter_view_wise: number of iterations for view-wise training
    :param seed1: seed 1
    :param seed2: seed 2
    :param iter_from: iter_from to start training from
    :param n_total_iter_from: used for continuing training
    :param n_epochs: number of training epochs
    :param lr_stage1: for view_wise training
    :param lr_stage2: for co-training
    :param n_save_iter: Determines how many epochs before saving model version
    :param data_path: data path
    :param model_dir_root_path: the model directory root path to save to
    :param model_pre_trained_dir: Path to pre-trained model
    :param note:
    :return:
    """

    """ setting """
    # gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    # time
    now = time.localtime()
    now_format = time.strftime("%Y-%m-%d %H:%M:%S", now)  # time format
    date_now = now_format.split(' ')[0]
    time_now = now_format.split(' ')[1]

    # save model path
    save_path = os.path.join(model_dir_root_path, date_now, time_now)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # print setting
    print("----------------------------------setting-------------------------------------")
    print("lr for stage1 :%f" % lr_stage1)
    print("lr for stage2 :%f" % lr_stage2)
    if model_pre_trained_dir is None:
        print("pre-trained dir is None")
    else:
        print("pre-trained dir:%s" % model_pre_trained_dir)
    print("path of saving model:%s" % save_path)
    print("----------------------------------setting-------------------------------------")

    # save parameters to TXT.
    parameter_dict = {"gpu": gpu,
                      "model_pre_trained_dir": model_pre_trained_dir,
                      "iter_from": iter_from,
                      "lr_stage1": lr_stage1,
                      "lr_stage2": lr_stage2,
                      "save_path": save_path,
                      'note': note}
    txt_name = 'parameter_log.txt'
    path = os.path.join(save_path, txt_name)
    with codecs.open(path, mode='a', encoding='utf-8') as file_txt:
        for key, value in parameter_dict.items():
            file_txt.write(str(key) + ':' + str(value) + '\n')

    # logging
    logging.basicConfig(filename=save_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(parameter_dict)

    # tensorboardX
    writer = SummaryWriter(log_dir=save_path)

    # label_dict
    label_list = [0, 1]

    """ data generator """
    # load all data path
    total_volume_path, total_label_path = util.gen_Pancreas_data_path(data_path)

    # 82 data -> 62 data for training
    # 82 data -> 20 data for validation
    train_volume_path, train_label_path, valid_volume_path, valid_label_path \
        = util.divide_data2train_valid(total_volume_path, total_label_path, 62, seed1)

    # dataset
    # training
    labeled_idxs = list(range(12))
    unlabeled_idxs = list(range(12, 62))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)
    train_dataset = Pancreas(train_volume_path, train_label_path,
                             transform_views=[transforms.Compose([RandomNoise(), ToTensor()]),
                                              transforms.Compose([RandomNoise(), Rotate((0, 2)), ToTensor()]),
                                              transforms.Compose([RandomNoise(), Rotate((1, 2)), ToTensor()])])
    # validation
    # valid_dataset = Pancreas(valid_volume_path, valid_label_path,
    #                          transform=transforms.Compose([ToTensor]))

    # dataloader
    def worker_init_fn(worker_id):
        random.seed(seed1 + worker_id)
    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=8,
                                  pin_memory=True, worker_init_fn=worker_init_fn)

    """ model, optimizer, loss """
    model_multi_views = []
    for i in range(view_num):
        model_view = UNet3D(1, 2, has_dropout=True).to("cuda:%d" % i)
        model_multi_views.append(model_view)
    if iter_from != 0:
        for i in range(view_num):
            model_multi_views[i].load_state_dict(torch.load("%s/model_view%d_%d.ckpt"% (model_pre_trained_dir, i+1, iter_from)))

    optimizer_stage1_multi_views = []
    optimizer_stage2_multi_views = []
    for i in range(view_num):
        # view-wise training stage, lr=7e-3, m=0.9, weight decay = 4e-5, iterations = 20k
        # optimizer_stage1_view = torch.optim.SGD(model_multi_views[i].parameters(), lr=lr_stage1, momentum=0.9, weight_decay=4e-5)
        optimizer_stage1_view = torch.optim.Adam(model_multi_views[i].parameters(), lr=lr_stage2)
        optimizer_stage1_multi_views.append(optimizer_stage1_view)
        # co-training stage, constant lr=1e-3, iterations = 5k
        optimizer_stage2_view = torch.optim.SGD(model_multi_views[i].parameters(), lr=lr_stage2)
        optimizer_stage2_multi_views.append(optimizer_stage2_view)

    criterion1 = nn.CrossEntropyLoss()
    criterion2 = dice_loss

    n_total_iter = 0
    if n_total_iter_from != 0:
        n_total_iter = n_total_iter_from
    if n_total_iter < n_iter_view_wise:
        optimizer_multi_views = optimizer_stage1_multi_views
    else:
        optimizer_multi_views = optimizer_stage2_multi_views

    for epoch in range(n_epochs):

        for batch_index, sample_batch_views in enumerate(train_dataloader):
            # start_time
            start = time.time()

            # loading data
            for i, sample_batch in enumerate(sample_batch_views):
                sample_batch['volume'] = sample_batch['volume'].to("cuda:%d" % i).float()
                sample_batch['label'] = sample_batch['label'].to("cuda:%d" % i).float()

            # labeled data(volume, label)
            # [view_num * (bs, 1, 128, 128, 128)]
            labeled_volume_batch_views = [sample_batch_views[i]['volume'][:labeled_bs] for i in range(view_num)]

            # [view_num * (bs, 1, 128, 128, 128)]
            label_batch_views = [sample_batch_views[i]['label'][:labeled_bs] for i in range(view_num)]

            # unlabeled data (volume)
            # [view_num * (bs, 1, 128, 128, 128)]
            unlabeled_volume_batch_views = [sample_batch_views[i]['volume'][labeled_bs:] for i in range(view_num)]

            # ------------------
            #    Train model
            # ------------------
            # zeros the parameter gradients
            for i in range(view_num):
                optimizer_multi_views[i].zero_grad()

            # run 3D U-Net model on labeled data with multi-views
            # [view_num * (bs, 2, 128, 128, 128)]
            pred_labeled_views = [model_multi_views[i](labeled_volume_batch_views[i]) for i in range(view_num)]
            pred_labeled_softmax_views = [F.softmax(pred_labeled_views[i], dim=1) for i in range(view_num)]

            # run 3D U-Net model on unlabeled data with multi-views
            pred_unlabeled_views = [model_multi_views[i](unlabeled_volume_batch_views[i]) for i in range(view_num)]
            pred_unlabeled_softmax_views = [F.softmax(pred_unlabeled_views[i], dim=1) for i in range(view_num)]

            # run 3D U-Net bayesian model on unlabeled data with multi-views
            T = 8
            unlabeled_volume_batch_repeat_views = [unlabeled_volume_batch_views[i].repeat(2, 1, 1, 1, 1) for i in range(view_num)]
            stride = unlabeled_volume_batch_repeat_views[0].shape[0] // 2
            pred_unlabeled_bayes_views = [torch.zeros([stride*T, 2, 128, 128, 128]).to('cuda:%d' % i) for i in range(view_num)]
            for t in range(T//2):
                noise_unlabeled_volume_views = [unlabeled_volume_batch_repeat_views[i] +
                                                torch.clamp(torch.rand_like(unlabeled_volume_batch_repeat_views[i])*0.1,
                                                            -0.2, 0.2).to('cuda:%d' % i) for i in range(view_num)]
                with torch.no_grad():
                    for i in range(view_num):
                        pred_unlabeled_bayes_views[i][2*stride*t:2*stride*(t+1)] = model_multi_views[i](noise_unlabeled_volume_views[i])

            pred_unlabeled_bayes_views = [F.softmax(pred_unlabeled_bayes_views[i], dim=1) for i in range(view_num)]
            pred_unlabeled_bayes_views = [pred_unlabeled_bayes_views[i].reshape(T, stride, 2, 128, 128, 128) for i in range(view_num)]
            pred_unlabeled_bayes_views = [torch.mean(pred_unlabeled_bayes_views[i], dim=0) for i in range(view_num)]  # [view_num * (bs, 2, 128, 128, 128)]
            uncertainty_views = [-1.0 * torch.sum(pred_unlabeled_bayes_views[i] * torch.log(pred_unlabeled_bayes_views[i] + 1e-6),
                                                  dim=1, keepdim=True) for i in range(view_num)]  # [view_num * (bs, 1, 128, 128, 128)]

            # turn uncertainty and pred_unlabeled into the same view
            rotate_axes = [(0, 2), (1, 2)]
            for i in range(view_num):
                if i == 0:
                    continue
                else:
                    pred_unlabeled_softmax_views[i] = pred_unlabeled_softmax_views[i].rot90(dims=rotate_axes[i + 1],
                                                                                            k=-1)
                    pred_unlabeled_views[i] = pred_unlabeled_views[i].rot90(dims=rotate_axes[i + 1], k=-1)
                    uncertainty_views[i] = uncertainty_views[i].rot90(dims=rotate_axes[i + 1], k=-1)

            # label fusion
            pseudo_label_views = []
            for i in range(view_num):
                top = torch.zeros_like(pred_unlabeled_softmax_views[i]).to('cuda:%d' % i)
                down = torch.zeros_like(uncertainty_views[i]).to('cuda:%d' % i)
                for j in range(view_num):
                    if i == j:
                        continue
                    else:
                        top += (pred_unlabeled_softmax_views[j] / uncertainty_views[j]).to('cuda:%d' % i)
                        down += 1 / uncertainty_views[j].to('cuda:%d' % i)
                pseudo_label = top / down
                pseudo_label = torch.argmax(pseudo_label, dim=1)
                pseudo_label_views.append(pseudo_label)  # [view_num * (bs, 128, 128, 128)]

            # supervised loss
            loss_sup1_views = []
            loss_sup2_views = []
            loss_sup_total_views = []
            for i in range(view_num):
                label_batch_stand_view = util.standardized_seg(label_batch_views[i], label_list, "cuda:%d" % i)
                loss_sup1 = criterion1(pred_labeled_views[i], label_batch_stand_view)
                loss_sup1_views.append(loss_sup1)

                label_batch_one_hot_view = util.onehot(label_batch_views[i], label_list, "cuda:%d" % i)
                loss_sup2 = criterion2(pred_labeled_softmax_views[i][:, 1, :, :, :], label_batch_one_hot_view[:, 1, :, :, :])
                loss_sup2_views.append(loss_sup2)

                loss_sup_total_views.append(0.5*loss_sup1+0.5*loss_sup2)

            # co-training loss
            loss_cot1_views = []
            loss_cot2_views = []
            loss_cot_total_views = []
            for i in range(view_num):
                loss_cot1 = criterion1(pred_unlabeled_views[i], pseudo_label_views[i])
                loss_cot1_views.append(loss_cot1)

                pseudo_label_view = pseudo_label_views[i].unsqueeze(dim=1)  # (bs, 1, 128, 128, 128)
                pseudo_label_one_hot_view = util.onehot(pseudo_label_view, label_list, "cuda:%d" % i)
                loss_cot2 = criterion2(pred_unlabeled_softmax_views[i][:, 1, :, :, :], pseudo_label_one_hot_view[:, 1, :, :, :])
                loss_cot2_views.append(loss_cot2)

                loss_cot_total_views.append(0.5 * loss_cot1 + 0.5 * loss_cot2)

            # backwards and optimize
            # before n_iter_view_wise iterations -- train separately; after n_iter_view_wise iterations -- co-training
            if n_total_iter < n_iter_view_wise:
                loss_total_views = loss_sup_total_views
            else:
                optimizer_multi_views = optimizer_stage2_multi_views
                loss_total_views = [loss_sup_total_views[i] + 0.2*loss_cot_total_views[i] for i in range(view_num)]

            for i in range(view_num):
                loss_total_views[i].backward()
                optimizer_multi_views[i].step()

            # logging, tensorboard
            # ---------------------
            #     Print log
            # ---------------------
            # Determine approximate time left
            end = time.time()
            iter_left = (n_epochs - epoch) * (len(train_dataloader) - batch_index)
            time_left = datetime.timedelta(seconds=iter_left * (end - start))

            # print log
            logging.info("[Epoch: %4d/%d] [n_total_iter: %5d] [labeled volume index: %2d/%d] "
                         "[loss view1: %f] [loss view2: %f] [loss view3: %f][ETA: %s]"
                         % (epoch, n_epochs, n_total_iter + 1, batch_index + 1, len(train_dataloader),
                            loss_total_views[0].item(), loss_total_views[1].item(), loss_total_views[2].item(), time_left))

            # tensorboardX log writer
            for i in range(view_num):
                writer.add_scalar("loss_view%d/loss_SupCot_total" % i, loss_total_views[i].item(), global_step=n_total_iter)

                writer.add_scalar("loss_view%d/loss_sup_total" % i, loss_sup_total_views[i].item(), global_step=n_total_iter)
                writer.add_scalar("loss_view%d/loss_sup_CrossEntropy" % i, loss_sup1_views[i].item(), global_step=n_total_iter)
                writer.add_scalar("loss_view%d/loss_sup_Dice" % i, loss_sup2_views[i].item(), global_step=n_total_iter)

                writer.add_scalar("loss_view%d/loss_cot_total" % i, loss_cot_total_views[i].item(), global_step=n_total_iter)
                writer.add_scalar("loss_view%d/loss_cot_CrossEntropy" % i, loss_cot1_views[i].item(), global_step=n_total_iter)
                writer.add_scalar("loss_view%d/loss_cot_Dice" % i, loss_cot2_views[i].item(), global_step=n_total_iter)

            # save model
            if n_total_iter % n_save_iter == 0:
                # Save model checkpoints
                for i in range(view_num):
                    torch.save(model_multi_views[i].state_dict(), "%s/model_view%d_%d.ckpt" % (save_path, i+1, n_total_iter))
                    logging.info("save model : %s/model_view%d_%d.ckpt" % (save_path, i+1, n_total_iter))
            n_total_iter += 1

    for i in range(view_num):
        torch.save(model_multi_views[i].state_dict(), "%s/model_view%d_%d.ckpt" % (save_path, i+1, n_total_iter))
        logging.info("save model : %s/model_view%d_%d.ckpt" % (save_path, i+1, n_total_iter))
    writer.close()


if __name__ == "__main__":
    train(**vars(arg))
