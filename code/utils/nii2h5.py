import glob
import os
import random
import json
from tqdm import tqdm
import SimpleITK as sitk
import nibabel as nib
import numpy as np
import h5py


def Process_Pancreas_forSeg_toh5():
    root_path = '/home/hra/dataset/Pancreas/Pancreas_region/'  # 该文件夹下的数据已经处理成128*128*128
    volume_paths = glob.glob(os.path.join(root_path, "img", '*'))
    prefix_list = [p.split('.', 1)[0] for p in sorted(volume_paths)]
    random.shuffle(prefix_list)
    prefix_train_list = prefix_list[:60]
    prefix_eval_list = prefix_list[60:]

    train_json = json.dumps({'prefix': prefix_train_list})
    eval_json = json.dumps({'prefix': prefix_eval_list})

    save_path = '/home/hra/dataset/Pancreas/Pancreas-h5'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_path = os.path.join(save_path, 'prefix_train_list.json')
    eval_path = os.path.join(save_path, 'prefix_eval_list.json')
    with open(train_path, 'w') as f:
        f.write(train_json)
    with open(eval_path, 'w') as f:
        f.write(eval_json)

    for prefix in tqdm(prefix_list):
        h5_save_path = os.path.join(save_path, os.path.basename(prefix) + ".h5")
        volume_path = prefix + '.nii.gz'
        label_path = volume_path.replace('img', 'label')
        label_path = label_path.replace('/image/', '/label/')

        volume_array = nib.load(volume_path).get_fdata()
        label_array = nib.load(label_path).get_fdata()

        array_2_h5(['image', 'label'], [volume_array, label_array], h5_save_path, 'w')


def array_2_h5(datasetnames, arrays, filename, mode):
    """
    :param datasetnames:(list[str]) 要写入文件的多维数组的名字,与array对应
    :param arrays:(list[ndarray]) 要写入文件的多维数组
    :param filename:(str) 被写文件名
    :param mode:(str) 文件读写的参数：w,a
    """
    assert mode in ['w', 'a']

    try:
        print(f'\nbuilding {filename}...')
        with h5py.File(filename, mode) as file:
            for name, array in zip(datasetnames, arrays):
                if isinstance(array, dict):
                    grp = file.create_group(name)
                    for k, v in array.items():
                        grp.create_dataset(k, data=v)
                else:
                    file.create_dataset(name, data=array)
    except (IOError, RuntimeError) as e:
        print(e)
        raise e


if __name__ == '__main__':
    Process_Pancreas_forSeg_toh5()
