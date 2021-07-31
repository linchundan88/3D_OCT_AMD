
import os
import sys
sys.path.append(os.path.abspath('..'))


#region DICOM->images
'''
dir_source = '/disk1/3D_OCT_DME/Topocon_dicom/'
from libs.dicom.my_dicom import dicom_save_dirs
dir_dest = '/disk1/3D_OCT_DME/original/Topocon/'
for label_str in ['M0', 'M1', 'M2']:
    dir_source_tmp = os.path.join(dir_source, label_str)
    dir_dest_tmp = os.path.join(dir_dest, label_str)
    dicom_save_dirs(dir_source_tmp, dir_dest_tmp, save_npy=True, save_image_files=True)
'''
#endregion

from libs.img_preprocess.my_image_helper import resize_images
from libs.dicom.my_dicom import slices_to_npy
dir_source = '/disk1/3D_OCT_AMD/2021_4_22/original/M0/Topocon/'
dir_dest = '/disk1/3D_OCT_AMD/2021_4_22/preprocess/128_128_128/M0/Topocon/'
resize_images(dir_source, dir_dest,  image_shape=(128, 128))
slices_to_npy(dir_dest)

dir_source = '/disk1/3D_OCT_AMD/2021_4_22/original/M1/Topocon/'
dir_dest = '/disk1/3D_OCT_AMD/2021_4_22/preprocess/128_128_128/M1/Topocon/'
resize_images(dir_source, dir_dest,  image_shape=(128, 128))
slices_to_npy(dir_dest)


print('OK')