
import os
import sys
sys.path.append(os.path.abspath('..'))
from libs.img_preprocess.my_image_helper import resize_images
from libs.dicom.my_dicom import slices_to_npy

if __name__ == '__main__':
    dir_source = '/disk1/3D_OCT_AMD/2021_4_22/original/M0/ZEISS/'
    dir_dest = '/disk1/3D_OCT_AMD/2021_4_22/preprocess/128_128_128/M0/ZEISS'
    resize_images(dir_source, dir_dest, p_image_to_square=True, image_shape=(128, 128))
    slices_to_npy(dir_dest)

    dir_source = '/disk1/3D_OCT_AMD/2021_4_22/original/M1/ZEISS/'
    dir_dest = '/disk1/3D_OCT_AMD/2021_4_22/preprocess/128_128_128/M1/ZEISS'
    resize_images(dir_source, dir_dest, p_image_to_square=True, image_shape=(128, 128))
    slices_to_npy(dir_dest)


    '''
    dest_dir = '/disk1/3D_OCT_DME/preprocess/64_64_64/ZEISS/'
    resize_images_dir(source_dir, dest_dir,
                      p_image_to_square=True, image_shape=(64, 64))
    # save_npy(dest_dir, depth_ratio=1, remainder=0)
    save_npy(dest_dir, depth_ratio=2, remainder=0)
    save_npy(dest_dir, depth_ratio=2, remainder=1)
    '''

    print('OK')



