from utils import util
from dataloaders import PancreasMultiView
import nibabel as nib
import os


path1 = '/home/hra/dataset/Pancreas/Pancreas_region2'
seed = 1997

total_volume_path1, total_label_path1 = util.gen_Pancreas_data_path(path1)
train_volume_path1, train_label_path1, valid_volume_path1, valid_label_path1 = util.divide_data2train_valid(total_volume_path1, total_label_path1, 62, seed)

volume = nib.load(train_label_path1[0]).get_fdata()
label = nib.load(train_label_path1[0]).get_fdata()
name = os.path.basename(train_label_path1[0])
sample = {"volume": volume, "label": label, "name": name}
RC = PancreasMultiView.RandomCrop((128, 128, 128))
r = RC(sample)

print('d')
