"""
Training 3D U-Net
"""
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
from dataloaders.Pancreas import Pancreas
from dataloaders.Pancreas import Rotate, RandomNoise, ToTensor, RandomRotFlip, RandomCrop
from networks.UNet3D import UNet3D
# from networks.UNet3D_v2 import UNet3D

# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument("--gpu",
                    type=str,
                    default='7',
                    help="gpu id")

parser.add_argument("--batch_size",
                    type=int,
                    default='4',
                    help="batch size")

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
                    default="upper bound on segmentation with 3D UNet",
                    # default=None,
                    help="note")

arg = parser.parse_args()


def train(gpu,
          batch_size,
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
                      "batch size": batch_size,
                      "model_pre_trained_dir": model_pre_trained_dir,
                      "data path": data_path,
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

    # patch size
    patch_size = (128, 128, 128)

    """ data generator """
    # load all data path
    total_volume_path, total_label_path = util.gen_Pancreas_data_path(data_path)

    # 82 data -> 62 data for training
    # 82 data -> 20 data for validation
    train_volume_path, train_label_path, valid_volume_path, valid_label_path \
        = util.divide_data2train_valid(total_volume_path, total_label_path, 62, seed1)

    train_volume_path = [i.replace('region', 'region2') for i in train_volume_path]
    train_label_path = [i.replace('region', 'region2') for i in train_label_path]

    # dataset
    # training
    # todo: validation
    train_volume_path.append(train_volume_path[-1])
    train_label_path.append(train_label_path[-1])
    total_dataset = Pancreas(train_volume_path, train_label_path,
                             transform=transforms.Compose([RandomCrop(patch_size),
                                                           RandomNoise(), ToTensor()]))
    # validation
    valid_dataset = Pancreas(valid_volume_path, valid_label_path,
                             transform=transforms.Compose([ToTensor]))

    # dataloader
    def worker_init_fn(worker_id):
        random.seed(seed1 + worker_id)
    total_dataloader = DataLoader(total_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                  pin_memory=True, worker_init_fn=worker_init_fn)

    """ model, optimizer, loss """
    modelS = UNet3D(1, 2, has_dropout=True).cuda()
    if iter_from != 0:
        modelS.load_state_dict(torch.load("%s/modelS_%d.ckpt" % (model_pre_trained_dir, iter_from)))

    optimizer_S = torch.optim.Adam(modelS.parameters(), lr=lr)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = dice_loss

    """ training loop """
    n_total_iter = 0
    if n_total_iter_from != 0:
        n_total_iter = n_total_iter_from

    for epoch in range(n_epochs):

        for total_index, total_sample in enumerate(total_dataloader):
            # start_time
            start = time.time()

            # generate moving data
            total_name = total_sample['name']
            total_input = total_sample['volume'].to('cuda').float()
            total_seg = total_sample['label'].to('cuda').float()

            # ------------------
            #    Train model
            # ------------------
            # zeros the parameter gradients
            optimizer_S.zero_grad()

            # run 3D U-Net model
            pred = modelS(total_input)

            # Calculate loss
            # todo : if wrong, check label_list
            total_seg_stand = util.standardized_seg(total_seg, label_list)
            loss_1 = criterion1(pred, total_seg_stand)
            total_seg_one_hot = util.onehot(total_seg, label_list)
            pred_softmax = F.softmax(pred, dim=1)
            loss_2 = criterion2(pred_softmax[:,1,:,:,:], total_seg_one_hot[:,1,:,:,:])
            loss = 0.5 * (loss_1 + loss_2)

            # backwards and optimize
            loss.backward()
            optimizer_S.step()

            # ---------------------
            #     Print log
            # ---------------------
            # Determine approximate time left
            end = time.time()
            iter_left = (n_epochs - epoch) * (len(total_dataloader) - total_index)
            time_left = datetime.timedelta(seconds=iter_left * (end - start))

            # print log
            logging.info("[Epoch: %4d/%d] [n_total_iter: %5d] [Total index: %2d/%d] [loss: %f] [ETA: %s]"
                         % (epoch, n_epochs, n_total_iter+1, total_index+1, len(total_dataloader), loss.item(), time_left))

            # tensorboardX log writer
            writer.add_scalar("loss/loss",              loss.item(),   global_step=n_total_iter)
            writer.add_scalar("loss/loss_CrossEntropy", loss_1.item(), global_step=n_total_iter)
            writer.add_scalar("loss/loss_Dice",         loss_2.item(), global_step=n_total_iter)

            if n_total_iter % n_save_iter == 0:
                # Save model checkpoints
                torch.save(modelS.state_dict(), "%s/modelS_%d.ckpt" % (save_path, n_total_iter))
                logging.info("save model : %s/modelS_%d.ckpt" % (save_path, n_total_iter))
            n_total_iter += 1

    torch.save(modelS.state_dict(), "%s/modelS_%d.ckpt" % (save_path, n_total_iter))
    logging.info("save model : %s/modelS_%d.ckpt" % (save_path, n_total_iter))
    writer.close()



if __name__ == "__main__":
    train(**vars(arg))

