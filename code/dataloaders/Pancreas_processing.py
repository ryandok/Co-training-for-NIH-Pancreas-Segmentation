"""
NIH pancreas dataset contains 80(82?) abdominal 3D CT
size with 512 × 512 × D (181 <= D <= 466)
The CT scans have resolutions of 512x512 pixels with varying pixel sizes and slice thickness between 1.5 − 2.5 mm,
acquired on Philips and Siemens MDCT scanners (120 kVp tube voltage).
https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT#2251404082af2dca8f2443b1bef1e85ac73acd44

pre-processing:
[1]:
use the soft tissue CT window range of [−125, 275] HU.
The intensities of each slice are then rescaled to [0.0, 255.0].

[2]:
randomly crop 96×96×96 3D CT Pancreas sub-volume as the network input.

[3]:
we did not normalize the spatial resolution into the same one
since we wanted to impose the networks to learn to deal with the variations between different volumetric cases

[4]:
we use the soft tissue CT window range of [−125, 275] HU (Zhou et al. 2019),
and resample all images to an isotropic resolution of 1.0 × 1.0 × 1.0mm.
Finally, we crop the images centering at the pancreas region based on the ground truth with enlarged margins (25 voxels)
and normalize them as zero mean and unit variance.

[5]:
all CT scans were cropped to a size of [192, 240], which
still can fully cover the pancreas in the CT scans.
"""
# TODO: 看看别人是怎么对胰腺数据进行预处理的
# TODO: 窗宽 窗位 标准化 [-125,275] [0~255]/[mean=0, std=1]
# TODO: 空间分辨率（设置相同？） 整个腹部区域/胰腺部分？
# TODO: 重采样；设置相同尺寸大小（random_crop，padding？）

import os
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from skimage.transform import resize
from skimage.util import pad
from glob import glob
from tqdm import tqdm

# todo: read
# todo: resample
# todo: CT window
# todo: crop ROI
# todo: resize and padding
# todo: norm?
img_root_path = '/home/hra/dataset/Pancreas/Pancreas_original/img/'
label_root_path = '/home/hra/dataset/Pancreas/Pancreas_original/label/'


def read_nii(img_path):
    """
    read img and label according to img_file
    :param img_path: image path
    :return: img and label (itk class)
    """
    label_path = label_root_path + 'label' + img_path.split('img')[-1]
    img_itk = sitk.ReadImage(img_path)
    label_itk = sitk.ReadImage(label_path)
    return img_itk, label_itk


def resample(img_itk, is_label=False):
    """
    resample all images to an isotropic resolution of 1*1*1
    :param img_itk:
    :param is_label:
    :return:
    """
    original_size = img_itk.GetSize()
    original_spacing = img_itk.GetSpacing()
    new_spacing = [1, 1, 1]

    new_size = [int(round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
                int(round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
                int(round(original_size[2] * (original_spacing[2] / new_spacing[2])))]
    if is_label:
        resamplemethod = sitk.sitkNearestNeighbor
    else:
        resamplemethod = sitk.sitkLinear

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img_itk)  # 需要重新采样的目标图像
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    imgResampled_itk = resampler.Execute(img_itk)  # 得到重新采样后的图像
    imgResampled_array = sitk.GetArrayFromImage(imgResampled_itk).transpose([2,1,0])
    return imgResampled_itk, imgResampled_array


def resize_in_same(img, output_shape=[128, 128, 128], is_label=False):
    """
    resize all the image into same_shape
    :param img:
    :param same_shape:
    :return:
    """
    origin_shape = img.shape
    rate = np.zeros(3)
    rate[0] = origin_shape[0] / output_shape[0]
    rate[1] = origin_shape[1] / output_shape[1]
    rate[2] = origin_shape[2] / output_shape[2]
    max_rate = rate.max()
    # keep scale
    keep_scale_shape = (int(origin_shape[0]/max_rate), int(origin_shape[1]/max_rate), int(origin_shape[2]/max_rate))
    if is_label:
        order = 0
    else:
        order = 1
    # keep scale resize
    img_resized_keep_prop = resize(img, keep_scale_shape, order=order,
                                   preserve_range=True, clip=False, anti_aliasing=False)

    # padding to the output shape
    x1 = int((output_shape[0] - keep_scale_shape[0]) / 2)
    x2 = output_shape[0] - keep_scale_shape[0] - x1
    y1 = int((output_shape[1] - keep_scale_shape[1]) / 2)
    y2 = output_shape[1] - keep_scale_shape[1] - y1
    z1 = int((output_shape[2] - keep_scale_shape[2]) / 2)
    z2 = output_shape[2] - keep_scale_shape[2] - z1
    min_intensity = np.min(img_resized_keep_prop)
    output_resized_padded = pad(img_resized_keep_prop, ((x1, x2), (y1, y2), (z1, z2)),
                                'constant', constant_values=min_intensity)

    return output_resized_padded


def save_image(data, affine, output_path):
    data_nii = nib.Nifti1Image(data, affine)
    nib.save(data_nii, output_path)
    print("Saving done : {}".format(output_path))
    return 1


def find_region_margin(label):
    """
    find the margin according to the GT
    :param label:
    :return:
    """
    shape = label.shape
    z1 = y1 = x1 = 0
    x2 = y2 = 512 - 1
    z2 = shape[2] - 1

    for z in range(shape[2]):
        plane = label[:, :, z]
        if 1 in plane:
            # print('z1=%d' % z)
            z1 = z
            break
    for z in range(shape[2] - 1, -1, -1):
        plane = label[:, :, z]
        if 1 in plane:
            # print('z2=%d' % z)
            z2 = z
            break

    for y in range(shape[1]):
        plane = label[:, y, :]
        if 1 in plane:
            # print('y1=%d' % y)
            y1 = y
            break
    for y in range(shape[1] - 1, -1, -1):
        plane = label[:, y, :]
        if 1 in plane:
            # print('y2=%d' % y)
            y2 = y
            break

    for x in range(shape[0]):
        plane = label[x, :, :]
        if 1 in plane:
            # print('x1=%d' % x)
            x1 = x
            break
    for x in range(shape[0] - 1, -1, -1):
        plane = label[x, :, :]
        if 1 in plane:
            # print('x2=%d' % x)
            x2 = x
            break

    margin = (x1, x2, y1, y2, z1, z2)

    return margin


def crop_ROI(img, label):
    """
    crop ROI according to label
    crop the images centering at the pancreas region
    based on the ground truth with enlarged margins (25 voxels)
    :param img:
    :param label:
    :return: crop_img, crop_label
    """
    margin = find_region_margin(label)
    enlarged_margin = []
    shape = label.shape
    for index, mar in enumerate(margin):
        if index % 2 == 0:
            enl_mar = mar - 25
            if enl_mar <= 0:
                enl_mar = 0
            enlarged_margin.append(enl_mar)
        else:
            enl_mar = mar + 25
            if enl_mar >= shape[(index-1)//2]:
                enl_mar = shape[(index-1)//2]
            enlarged_margin.append(enl_mar)

    crop_img = img[enlarged_margin[0]:enlarged_margin[1], enlarged_margin[2]:enlarged_margin[3],
               enlarged_margin[4]:enlarged_margin[5]]
    crop_label = label[enlarged_margin[0]:enlarged_margin[1], enlarged_margin[2]:enlarged_margin[3],
               enlarged_margin[4]:enlarged_margin[5]]

    return crop_img, crop_label


def CT_window(img):
    """
    soft thissue CT window range of [-125, 275] HU
    :param img:
    :return:
    """
    img_clipped = np.clip(img, -125, 275)
    return img_clipped


def norm_0mean_1var(img):
    """
    zero mean and unit variance
    :param img:
    :return:
    """
    mean = np.mean(img)
    std = np.std(img)
    img_norm = (img - mean) / std
    return img_norm


# if __name__ == '__main__':
#     """
#     pre-processing data
#     """
#     output_img_root = '/home/hra/dataset/Pancreas/Pancreas_region2/img/'
#     output_label_root = '/home/hra/dataset/Pancreas/Pancreas_region2/label/'
#     # output_img_root = '../../data/'
#     # output_label_root = '../../data/'
#
#     img_files = os.listdir(img_root_path)
#
#     affine = np.array([[-1, 0, 0, 0],
#                        [0, 1, 0, 0],
#                        [0, 0, -1, 0],
#                        [0, 0, 0, 1]])
#     # shape = np.zeros([80, 3])
#     img_files = [img_files[-2]]
#     for i, img_file in enumerate(img_files):
#         img_name = img_file
#         label_name = 'label' + img_name.split('img')[-1]
#         img_file = os.path.join(img_root_path, img_file)
#
#         # read
#         img_itk, label_itk = read_nii(img_file)
#
#         # resample
#         _, img_resampled = resample(img_itk, is_label=False)
#         _, label_resampled = resample(label_itk, is_label=True)
#
#         # CT window [-125, 275]
#         img_windowed = CT_window(img_resampled)
#         label_windowed = CT_window(label_resampled)
#
#         # crop region
#         img_crop, label_crop = crop_ROI(img_windowed, label_windowed)
#         crop_shape = img_crop.shape
#         # shape[i, :] = crop_shape
#
#         # normalize
#         img_crop = norm_0mean_1var(img_crop)
#
#         # resized into same size
#         img_crop_resized = resize_in_same(img_crop, is_label=False)
#         label_crop_resized = resize_in_same(label_crop, is_label=True)
#
#         # save
#         save_image(img_crop_resized, affine, output_img_root+img_name)
#         save_image(label_crop_resized, affine, output_label_root+label_name)
#         print("saving ····  %2d/82" % (i+1))
#
#     print('done')


if __name__ == '__main__':
    """
       pre-processing data
       """
    output_img_root = '/home/hra/dataset/Pancreas/Pancreas_norm/img/'
    output_label_root = '/home/hra/dataset/Pancreas/Pancreas_norm/label/'

    if not os.path.exists(output_img_root):
        os.makedirs(output_img_root)
    if not os.path.exists(output_label_root):
        os.makedirs(output_label_root)

    img_list = glob(img_root_path + '*nii.gz')
    img_list = sorted(img_list)
    for img_path in tqdm(img_list):
        img = nib.load(img_path)
        img_affine = img.affine
        label_path = img_path.replace('img', 'label')
        label = nib.load(label_path)
        label_affine = label.affine

        img_array = img.get_fdata()
        label_array = label.get_fdata()

        img_array = CT_window(img_array)
        label_array = CT_window(label_array)

        # img_array = norm_0mean_1var(img_array)

        # img_array, label_array = crop_ROI(img_array, label_array)

        save_image(img_array, img_affine, output_img_root + os.path.basename(img_path))
        save_image(label_array, label_affine, output_label_root + os.path.basename(label_path))

        # print('s')








