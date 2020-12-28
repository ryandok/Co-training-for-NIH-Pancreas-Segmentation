import pydicom
import SimpleITK as sitk
import os


def readdcm(filepath):
    series_id = sitk.ImageSeriesReader.GetGDCMSeriesIDs(filepath)
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(filepath, series_id[0])
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    images = series_reader.Execute()
    return images


if __name__ == '__main__':
    root_path = "/home/hra/dataset/Pancreas/Pancreas_CT_dcm/image/"
    output_path = "/home/hra/dataset/Pancreas/Pancreas_original/img"
    PANCREAS_ID_list = []
    index = 1
    for root, subdirs, files in os.walk(root_path):
        print("第", index, "层")
        index += 1
        for filepath in files:
            print(os.path.join(root, filepath))
            PANCREAS_ID_list.append(root)
            break
        for sub in subdirs:
            print(os.path.join(root, sub))

    for index, file_path in enumerate(PANCREAS_ID_list):
        print("processing %2d/%d ..." % (index+1, len(PANCREAS_ID_list)))
        dcm_images = readdcm(file_path)
        output_file_name = output_path + '/' + 'img' + file_path.split('/')[-3].split('_')[-1] + '.nii.gz'
        sitk.WriteImage(dcm_images, output_file_name)


