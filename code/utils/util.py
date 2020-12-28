'''
utils
is used for hra@172.21.127.70
is used for train_v1/v2
root path : /data1/hra
'''
import nibabel as nib
import xlwt
import xlrd
import os
import numpy as np
import torch
import random


def dice_numpy2excel(dice_npy, dice_xls):
    """
    This function is used to save various dice values of each test data to an XLS file
    :param dice_npy: The path to the NPY file (inside which is dict) of the dice value
    :param dice_xls: Path to output XLS
    """
    each_volume_dice = np.load(dice_npy, allow_pickle=True).item()
    workbook = xlwt.Workbook()
    for key, item in each_volume_dice.items():
        name = key.split('/')[-2].split('_')[0] + '-' + key.split('/')[-1].split('_')[0]
        worksheet = workbook.add_sheet(name, cell_overwrite_ok=True)
        i = 0
        worksheet.write(i, 0, 'label_class')
        worksheet.write(i, 1, 'dice_value')
        i += 1
        for each_class, each_dice in item.items():
            worksheet.write(i, 0, each_class)
            worksheet.write(i, 1, each_dice)
            i += 1
    workbook.save(dice_xls)
    print(dice_xls, 'done')


def gen_label_class_xls():
    PATH = '/data1/hra/dataset/Pancreas/label'
    # PATH = "D:\\000RyanWong\LAB\data\CANDI\label"
    FILE_NAME = 'label0001.nii.gz'
    file = os.path.join(PATH, FILE_NAME)
    label_nii = nib.load(file).get_fdata()
    label_np = np.array(label_nii)
    size = label_np.shape
    value_dict = {}
    label_list = []
    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(size[2]):
                if label_np[i, j, k] != 0:
                    print('坐标:', i, j, k, 'label:', label_np[i, j, k])
                    if label_np[i, j, k] not in value_dict:
                        value_dict[int(label_np[i, j, k])] = 1
                        label_list.append(int(label_np[i, j, k]))
                    else:
                        value_dict[int(label_np[i, j, k])] += 1

    label_list.sort()

    # # 输出到excel
    book = xlwt.Workbook()
    sheet = book.add_sheet('sheet1', cell_overwrite_ok=True)
    for i in range(len(label_list)):
        sheet.write(i, 0, label_list[i])
    book.save('../../data/Pancreas-label-classes.xls')
    print('done')

# gen_label_class_xls()
# gen_label_class_xls(True)

def gen_labels_list(label_xls):
    """
    The Label list is generated based on the category content in the Label XLS file
    :param label_xls: path to label_xls
    :return: labels list
    """
    workbook = xlrd.open_workbook(label_xls)
    worksheet = workbook.sheet_by_index(0)
    nrows = worksheet.nrows
    labels = []
    for i in range(nrows):
        value = worksheet.row_values(i)
        value = value[0]
        value = int(value)
        labels.append(value)
    return labels


def gen_CANDI_data_path(data_path):
    """
    from root path to generate total_volume_path, total_label_path
    :return: total_volume_path, total_label_path
    """
    total_volume_path = []
    total_label_path = []

    volume_root_path = data_path + '/img'
    label_root_path = data_path + '/label'

    volume_name_list = os.listdir(volume_root_path)
    # label_name_list = os.listdir(label_root_path)

    for volume_name in volume_name_list:
        volume_path = os.path.join(volume_root_path, volume_name)
        total_volume_path.append(volume_path)

        label_name = volume_name.split('img')[0] + 'seg.nii.gz'
        label_path = os.path.join(label_root_path, label_name)
        total_label_path.append(label_path)

    return total_volume_path, total_label_path


def gen_Pancreas_data_path(data_path):
    """
    from data path to generate total_volume_path, total_label_path
    :return: total_volume_path, total_label_path
    """
    total_volume_path = []
    total_label_path = []

    volume_root_path = data_path + '/img'
    label_root_path = data_path + '/label'

    volume_name_list = os.listdir(volume_root_path)

    for volume_name in volume_name_list:
        volume_path = os.path.join(volume_root_path, volume_name)
        total_volume_path.append(volume_path)

        label_name = 'label' + volume_name.split('img')[1]
        label_path = os.path.join(label_root_path, label_name)
        total_label_path.append(label_path)

    return total_volume_path, total_label_path


def divide_data2train_valid(total_volume_path, total_label_path, index, seed):
    """
    used to divide all data into training dataset and validation dataset
    :param total_volume_path: the path to total volume
    :param total_label_path: the path to total label
    :param index: indicate how many data set as training dataset
    :param seed: for random
    :return:
    """
    random.seed(seed)
    random.shuffle(total_volume_path)
    random.seed(seed)
    random.shuffle(total_label_path)

    train_volume_path = total_volume_path[0:index]
    train_label_path = total_label_path[0:index]

    valid_volume_path = total_volume_path[index:]
    valid_label_path = total_label_path[index:]

    return train_volume_path, train_label_path, valid_volume_path, valid_label_path


def divide_data2unlabeled_labeled(volume_path, label_path, index, seed):
    """
    used to divide data into fixed data(unlabeled data) and moving data(labeled data)
    :param volume_path: path to volume
    :param label_path: path to label
    :param index: indicate how many data set as moving data
    :param seed: for random
    :return:
    """
    random.seed(seed)
    random.shuffle(volume_path)
    random.seed(seed)
    random.shuffle(label_path)

    labeled_volume_path = volume_path[0:index]
    labeled_label_path = label_path[0:index]

    unlabled_volume_path = volume_path[index:]
    unlabled_label_path = label_path[index:]

    return labeled_volume_path, labeled_label_path, unlabled_volume_path, unlabled_label_path


def onehot(tensor, label_list, device="cuda:0"):
    """
    one hot encoder
    :param tensor:
    :param label_list:
    :param device: cuda:?
    :return:
    """
    tensor = tensor.float()
    shape = list(tensor.shape)
    shape[1] = len(label_list)
    result = torch.zeros(shape).to(device)
    for index, label_class in enumerate(label_list):
        label_mask = torch.full(size=list(tensor.shape), fill_value=label_class).to(device)
        label_seg = (label_mask == tensor).float()
        result[:, index, :, :, :] = label_seg.squeeze(dim=1)
    return result


def gen_pseudo_label_path(volumes_path):
    pseudo_label_path = []
    root_path = '/data/hra/dataset/CANDI/pseudo'
    for single_path in volumes_path:
        name = single_path.split('/')[-1].split('.')[0] + '_pseudo.nii.gz'
        pseudo_single_path = os.path.join(root_path, name)
        pseudo_label_path.append(pseudo_single_path)
    return pseudo_label_path


def standardized_seg(seg, label_list, device="cuda:0"):
    """
    standardized seg_tensor with label list to generate a tensor,
    which can be put into nn.CrossEntropy(input, target) as "target"
    :param seg:
    :param label_list: (include 0)
    :param device: cuda device
    :return:
    """
    seg = torch.squeeze(seg, dim=1)
    result = torch.zeros(seg.shape, dtype=torch.long).to(device)
    for index, label_class in enumerate(label_list):
        label_mask = torch.full(size=list(seg.shape), fill_value=label_class).to(device)
        label_seg = (label_mask == seg).long()
        label_seg = label_seg * index
        result = torch.add(result, label_seg)
    return result


def fuse_pred_label(label, pred):
    """
    fuse prediction and label into one array
    label       -- 1
    prediction  -- 2
    overlap     -- 3
    :param label:
    :param pred:
    :return:
    """
    fusion = np.zeros_like(label)
    pred = 2 * pred
    fusion += label + pred
    return fusion


def save_volume(img_array, save_dir, name):
    img_array = np.squeeze(img_array)
    img_nii = nib.Nifti1Image(img_array, np.eye(4))
    nib.save(img_nii, os.path.join(save_dir, name))
    print('%10s saving done' % name)


def save_pred_demo(pred_tensor, name, save_dir='../data'):
    data = pred_tensor.cpu().detach().numpy()
    data = np.argmax(data, axis=1)
    save_volume(data, save_dir, name)


