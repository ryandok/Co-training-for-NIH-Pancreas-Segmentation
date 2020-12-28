"""
Training multi-view co-training 3D U-Net
wrong
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
from utils.losses import dice_loss
from dataloaders.PancreasMultiView import Pancreas, Rotate, RandomNoise, ToTensor, Flip, TwoStreamBatchSampler
from networks.UNet3D import UNet3D


# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument("--gpu",
                    type=str,
                    default='0,1',
                    help="gpu id")

parser.add_argument("--batch_size",
                    type=int,
                    default='4',
                    help="batch size per gpu")

parser.add_argument("--labeled_bs",
                    type=int,
                    default=2,
                    help="labeled batch size per gpu")

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

parser.add_argument("--lr",
                    type=float,
                    dest="lr",
                    default=0.001,
                    help="adam: learning rate")

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
                    # default='',
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
          seed1,
          seed2,
          iter_from,
          n_total_iter_from,
          n_epochs,
          lr,
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
    :param seed1: seed 1
    :param seed2: seed 2
    :param iter_from: iter_from to start training from
    :param n_total_iter_from: used for continuing training
    :param n_epochs: number of training epochs
    :param lr: learning rate
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
    print("lr:%f" % lr)
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
                      "lr": lr,
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

    # 80 data -> 60 data for training
    # 80 data -> 20 data for validation
    train_volume_path, train_label_path, valid_volume_path, valid_label_path \
        = util.divide_data2train_valid(total_volume_path, total_label_path, 60, seed1)

    # dataset
    # training
    labeled_idxs = list(range(12))
    unlabeled_idxs = list(range(12, 60))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)
    train_dataset = Pancreas(train_volume_path, train_label_path,
                             transform_views=[transforms.Compose([RandomNoise(), Rotate(0), ToTensor()]),
                                              transforms.Compose([RandomNoise(), Rotate(180), ToTensor()])])
    # validation
    # valid_dataset = Pancreas(valid_volume_path, valid_label_path,
    #                          transform=transforms.Compose([ToTensor]))

    # dataloader
    def worker_init_fn(worker_id):
        random.seed(seed1 + worker_id)
    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=8,
                                  pin_memory=True, worker_init_fn=worker_init_fn)

    """ model, optimizer, loss """
    model_view1 = UNet3D(1, 2, has_dropout=True).to("cuda:0")
    model_view2 = UNet3D(1, 2, has_dropout=True).to("cuda:1")
    # model_view3 = UNet3D(1, 2).cuda()
    if iter_from != 0:
        model_view1.load_state_dict(torch.load("%s/model_view1_%d.ckpt" % (model_pre_trained_dir, iter_from)))
        model_view2.load_state_dict(torch.load("%s/model_view2_%d.ckpt" % (model_pre_trained_dir, iter_from)))
        # model_view3.load_state_dict(torch.load("%s/model_view3_%d.ckpt" % (model_pre_trained_dir, iter_from)))

    optimizer_view1 = torch.optim.Adam(model_view1.parameters(), lr=lr)
    optimizer_view2 = torch.optim.Adam(model_view2.parameters(), lr=lr)
    # optimizer_view3 = torch.optim.Adam(model_view3.parameters(), lr=lr)

    criterion1 = nn.CrossEntropyLoss()
    criterion2 = dice_loss

    """ training loop """
    # TODO: do label fusion (T^-1)

    n_total_iter = 0
    if n_total_iter_from != 0:
        n_total_iter = n_total_iter_from

    for epoch in range(n_epochs):

        for batch_index, sample_batch_views in enumerate(train_dataloader):
            # start_time
            start = time.time()

            # loading data
            for i, sample_batch in enumerate(sample_batch_views):
                # sample_batch_name = sample_batch['name']
                if i == 0:
                    device = "cuda:0"
                else:
                    device = "cuda:1"
                sample_batch['volume'] = sample_batch['volume'].to(device).float()
                sample_batch['label'] = sample_batch['label'].to(device).float()

            # labeled data(volume, label)
            labeled_volume_batch_view1 = sample_batch_views[0]['volume'][:labeled_bs]
            labeled_volume_batch_view2 = sample_batch_views[1]['volume'][:labeled_bs]

            label_batch_view1 = sample_batch_views[0]['label'][:labeled_bs]
            label_batch_view2 = sample_batch_views[1]['label'][:labeled_bs]

            # unlabeled data (volume)
            unlabeled_volume_batch_view1 = sample_batch_views[0]['volume'][labeled_bs:]
            unlabeled_volume_batch_view2 = sample_batch_views[1]['volume'][labeled_bs:]

            # put noise into unlabeled data
            noise1 = torch.clamp(torch.rand_like(unlabeled_volume_batch_view1)*0.1, -0.2, 0.2).to('cuda:0')
            noise2 = torch.clamp(torch.rand_like(unlabeled_volume_batch_view1) * 0.1, -0.2, 0.2).to('cuda:1')
            # noise_unlabeled_volume_batch_view1 = unlabeled_volume_batch_view1 + noise1
            # noise_unlabeled_volume_batch_view2 = unlabeled_volume_batch_view2 + noise2

            # ------------------
            #    Train model
            # ------------------
            # zeros the parameter gradients
            optimizer_view1.zero_grad()
            optimizer_view2.zero_grad()

            # run 3D U-Net model on labeled data with view1 & view2
            pred_labeled_view1 = model_view1(labeled_volume_batch_view1)
            pred_labeled_view2 = model_view2(labeled_volume_batch_view2)

            # run 3D U-Net model on unlabeled data with view1 & view2 (Bayesian)
            T = 8
            unlabeled_volume_batch_r_view1 = unlabeled_volume_batch_view1.repeat(2, 1, 1, 1, 1)
            unlabeled_volume_batch_r_view2 = unlabeled_volume_batch_view2.repeat(2, 1, 1, 1, 1)
            stride = unlabeled_volume_batch_r_view1.shape[0] // 2
            pred_unlabeled_view1 = torch.zeros([stride * T, 2, 128, 128, 128]).to('cuda:0')
            pred_unlabeled_view2 = torch.zeros([stride * T, 2, 128, 128, 128]).to('cuda:1')
            for i in range(T//2):
                noise_unlabeled_volume_batch_view1 = unlabeled_volume_batch_r_view1 \
                                                     + torch.clamp(torch.rand_like(unlabeled_volume_batch_r_view1) * 0.1, -0.2, 0.2).to('cuda:0')
                noise_unlabeled_volume_batch_view2 = unlabeled_volume_batch_r_view2 \
                                                     + torch.clamp(torch.rand_like(unlabeled_volume_batch_r_view1) * 0.1, -0.2, 0.2).to('cuda:1')
                with torch.no_grad():
                    pred_unlabeled_view1[2*stride*i:2*stride*(i+1)] = model_view1(noise_unlabeled_volume_batch_view1)
                    pred_unlabeled_view2[2*stride*i:2*stride*(i+1)] = model_view2(noise_unlabeled_volume_batch_view2)

            pred_unlabeled_view1 = F.softmax(pred_unlabeled_view1, dim=1)
            pred_unlabeled_view1 = pred_unlabeled_view1.reshape(T, stride, 2, 128, 128, 128)
            pred_unlabeled_view1 = torch.mean(pred_unlabeled_view1, dim=0) # (batch, 2, 128, 128, 128)
            uncertainty_view1 = -1.0*torch.sum(pred_unlabeled_view1*torch.log(pred_unlabeled_view1 + 1e-6), dim=1,
                                               keepdim=True) # (batch, 1, 128, 128, 128)

            pred_unlabeled_view2 = F.softmax(pred_unlabeled_view2, dim=1)
            pred_unlabeled_view2 = pred_unlabeled_view2.reshape(T, stride, 2, 128, 128, 128)
            pred_unlabeled_view2 = torch.mean(pred_unlabeled_view2, dim=0)  # (batch, 2, 128, 128, 128)
            uncertainty_view2 = -1.0 * torch.sum(pred_unlabeled_view2 * torch.log(pred_unlabeled_view2 + 1e-6), dim=1,
                                                 keepdim=True)  # (batch, 1, 128, 128, 128)

            # TODO: label fusion(先要转为同一视角)
            # label_fusion_view1 =

            # Calculate loss
            label_batch_stand_view1 = util.standardized_seg(label_batch_view1, label_list, "cuda:0")
            label_batch_stand_view2 = util.standardized_seg(label_batch_view2, label_list, "cuda:1")

            loss_1_view1 = criterion1(pred_labeled_view1, label_batch_stand_view1)
            loss_1_view2 = criterion1(pred_labeled_view2, label_batch_stand_view2)

            label_batch_one_hot_view1 = util.onehot(label_batch_view1, label_list, "cuda:0")
            label_batch_one_hot_view2 = util.onehot(label_batch_view2, label_list, "cuda:1")

            pred_labeled_softmax_view1 = F.softmax(pred_labeled_view1, dim=1)
            pred_labeled_softmax_view2 = F.softmax(pred_labeled_view2, dim=1)

            loss_2_view1 = criterion2(pred_labeled_softmax_view1, label_batch_one_hot_view1)
            loss_2_view2 = criterion2(pred_labeled_softmax_view2, label_batch_one_hot_view2)

            loss_view1 = 0.5 * (loss_1_view1 + loss_2_view1)
            loss_view2 = 0.5 * (loss_1_view2 + loss_2_view2)

            # backwards and optimize
            loss_view1.backward()
            optimizer_view1.step()

            loss_view2.backward()
            optimizer_view2.step()

            # ---------------------
            #     Print log
            # ---------------------
            # Determine approximate time left
            end = time.time()
            iter_left = (n_epochs - epoch) * (len(train_dataloader) - batch_index)
            time_left = datetime.timedelta(seconds=iter_left * (end - start))

            # print log
            logging.info("[Epoch: %4d/%d] [n_total_iter: %5d] [Total index: %2d/%d] "
                         "[loss view1: %f] [loss view2: %f] [ETA: %s]"
                         % (epoch, n_epochs, n_total_iter+1, batch_index+1, len(train_dataloader),
                            loss_view1.item(), loss_view2.item(), time_left))

            # tensorboardX log writer
            writer.add_scalar("loss_view1/loss",              loss_view1.item(),   global_step=n_total_iter)
            writer.add_scalar("loss_view1/loss_CrossEntropy", loss_1_view1.item(), global_step=n_total_iter)
            writer.add_scalar("loss_view1/loss_Dice",         loss_2_view1.item(), global_step=n_total_iter)

            writer.add_scalar("loss_view2/loss",              loss_view2.item(), global_step=n_total_iter)
            writer.add_scalar("loss_view2/loss_CrossEntropy", loss_1_view2.item(), global_step=n_total_iter)
            writer.add_scalar("loss_view2/loss_Dice",         loss_2_view2.item(), global_step=n_total_iter)

            if n_total_iter % n_save_iter == 0:
                # Save model checkpoints
                torch.save(model_view1.state_dict(), "%s/model_view1_%d.ckpt" % (save_path, n_total_iter))
                torch.save(model_view2.state_dict(), "%s/model_view2_%d.ckpt" % (save_path, n_total_iter))
                logging.info("save model : %s/model_view1_%d.ckpt" % (save_path, n_total_iter))
                logging.info("save model : %s/model_view2_%d.ckpt" % (save_path, n_total_iter))
            n_total_iter += 1

    torch.save(model_view1.state_dict(), "%s/model_view1_%d.ckpt" % (save_path, n_total_iter))
    torch.save(model_view2.state_dict(), "%s/model_view2_%d.ckpt" % (save_path, n_total_iter))
    logging.info("save model : %s/model_view1_%d.ckpt" % (save_path, n_total_iter))
    logging.info("save model : %s/model_view2_%d.ckpt" % (save_path, n_total_iter))
    writer.close()



if __name__ == "__main__":
    train(**vars(arg))

