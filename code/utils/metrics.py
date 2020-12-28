import numpy as np


def dice(vol1_seg, vol2_seg, label_list):
    """
    used to evaluate the registration quality based on Dice
    :param vol1_seg：predict(moving volume segmentation)
    :param vol2_seg：GT(fixed volume segmentation)
    :param label_list: label class (include 0)
    :returns dice_value: is a dict(key=label_name, value=dice value) (not including 0)
    :returns average: average_dice
    """
    shape = vol1_seg.shape[1:]
    local_label_list = label_list[1:]
    dice_value = {}
    top_total = 0
    bottom_total = 0
    for label_class in local_label_list:
        label_array = np.empty(shape, dtype=np.float32)
        label_array.fill(label_class)
        v1 = vol1_seg == label_array
        v2 = vol2_seg == label_array
        top = 2 * np.sum(np.logical_and(v1, v2), axis=(1,2,3))
        bottom = np.sum(v1, axis=(1,2,3)) + np.sum(v2, axis=(1,2,3))
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dice_each_class = top / bottom
        dice_value[label_class] = dice_each_class

        top_total += top
        bottom_total += bottom

    average = top_total/bottom_total
    average = np.average(average)

    return dice_value, average

