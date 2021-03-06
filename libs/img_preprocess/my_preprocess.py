'''
create by jji, email:jji@stu.edu.cn
modified at 2019_5_23_12:00

del_black_or_white:  delete borders of fundus images

detect_xyr:  using HoughCircles detect circle, if not detected  suppose the center of the image is the center of the circle.

my_crop_xyz:  crop the image based on circle detected

add_black_margin: add some black margin areas, so that img aug(random rotate clip) will not delete meaningful edge region

my_preprocess: the main entrance of fundus images preprocess

multi_crop: code_test time img aug,  multi crop

get_fundus_border: only used in Fovea_center/DR0_4_inconsistent_400/crop_images

'''

import cv2
import numpy as np
import os
from imgaug import augmenters as iaa


DEL_PADDING_RATIO = 0.02  #used for del_black_or_white
CROP_PADDING_RATIO = 0.02  #used for my_crop_xyr

# del_black_or_white margin
THRETHOLD_LOW = 7
THRETHOLD_HIGH = 180

# HoughCircles
MIN_REDIUS_RATIO = 0.33
MAX_REDIUS_RATIO = 0.6

#illegal image
IMG_SMALL_THRETHOLD = 80

def del_black_or_white(img1):
    if img1.ndim == 2:
        img1 = np.expand_dims(img1, axis=-1)

    height, width = img1.shape[:2]

    (left, bottom) = (0, 0)
    (right, top) = (width, height)

    padding = int(min(width, height) * DEL_PADDING_RATIO)


    for i in range(width):
        array1 = img1[:, i, :]  #array1.shape[1]=3 RGB
        if np.sum(array1) > THRETHOLD_LOW * array1.shape[0] * array1.shape[1] and \
                np.sum(array1) < THRETHOLD_HIGH * array1.shape[0] * array1.shape[1]:
            left = i
            break
    left = max(0, left-padding)

    for i in range(width - 1, 0 - 1, -1):
        array1 = img1[:, i, :]
        if np.sum(array1) > THRETHOLD_LOW * array1.shape[0] * array1.shape[1] and \
                np.sum(array1) < THRETHOLD_HIGH * array1.shape[0] * array1.shape[1]:
            right = i
            break
    right = min(width, right + padding)

    for i in range(height):
        array1 = img1[i, :, :]
        if np.sum(array1) > THRETHOLD_LOW * array1.shape[0] * array1.shape[1] and \
                np.sum(array1) < THRETHOLD_HIGH * array1.shape[0] * array1.shape[1]:
            bottom = i
            break
    bottom = max(0, bottom - padding)

    for i in range(height - 1, 0 - 1, -1):
        array1 = img1[i, :, :]
        if np.sum(array1) > THRETHOLD_LOW * array1.shape[0] * array1.shape[1] and \
                np.sum(array1) < THRETHOLD_HIGH * array1.shape[0] * array1.shape[1]:
            top = i
            break
    top = min(height, top + padding)

    img2 = img1[bottom:top, left:right, :]

    return img2

def detect_xyr(img_source):
    if isinstance(img_source, str):
        try:
            img1 = cv2.imread(img_source)
        except:
            # Corrupt JPEG data1: 19 extraneous bytes before marker 0xc4
            raise Exception("image file not found:" + img_source)
        if img1 is None:
            raise Exception("image file error:" + img_source)
    else:
        img1 = img_source

    height, width = img1.shape[:2]

    myMinWidthHeight = min(width, height)  # ????????????1600 ??????????????????,???????????????????????????>??? train/22054_left.jpeg ??????
    myMinRadius = round(myMinWidthHeight * MIN_REDIUS_RATIO)
    myMaxRadius = round(myMinWidthHeight * MAX_REDIUS_RATIO)

    '''
    parameters of HoughCircles
    minDist?????????????????????????????????x,y???????????????????????????????????????minDist????????????????????????????????????????????????????????????minDist????????????????????????????????????????????????
    minDist??????????????????????????????????????????
    param1????????????????????????????????????????????????
    param2???cv2.HOUGH_GRADIENT?????????????????????????????????????????????????????????????????????
    
    According to our code_test about fundus images, param2 = 30 is enough, too high will miss some circles
    '''

    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=450, param1=120, param2=32,
                               minRadius=myMinRadius, maxRadius=myMaxRadius)

    found_circle = False

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        if (circles is not None) and (len(circles == 1)):
            # ??????????????????????????? 25.Hard exudates/chen_liang quan_05041954_clq19540405_557410.jpg

            x, y, r = circles[0]
            if x > (2 / 5 * width) and x < (3 / 5 * width) \
                    and y > (2 / 5 * height) and y < (3 / 5 * height):
                found_circle = True

    if not found_circle:
        # suppose the center of the image is the center of the circle.
        x = img1.shape[1] // 2
        y = img1.shape[0] // 2

        # get radius  according to the distribution of pixels of the middle line
        temp_x = img1[int(img1.shape[0] / 2), :, :].sum(1)
        r = int((temp_x > temp_x.mean() / 12).sum() / 2)

    return (found_circle, x, y, r)

def my_crop_xyr(img_source, x, y, r, crop_size=None):
    if isinstance(img_source, str):
        try:
            img1 = cv2.imread(img_source)
        except:
            # Corrupt JPEG data1: 19 extraneous bytes before marker 0xc4
            raise Exception("image file not found:" + img_source)
    else:
        img1 = img_source

    if img1 is None:
        raise Exception("image file error:" + img_source)

    height, width = img1.shape[:2]

    #  ???????????? ??????????????????  ??????????????????  ??????????????????,??????0???width
    # ???????????????,???????????????  r?????????

    img_padding = int(min(width, height) * CROP_PADDING_RATIO)

    image_left = int(max(0, x - r - img_padding))
    image_right = int(min(x + r + img_padding, width - 1))
    image_bottom = int(max(0, y - r - img_padding))
    image_top = int(min(y + r + img_padding, height - 1))

    if width >= height:  # ??????????????????
        if height >= 2 * (r + img_padding):
            # ???????????????
            img1 = img1[image_bottom: image_top, image_left:image_right]
        else:
            # ????????????????????????,?????????????????????
            img1 = img1[:, image_left:image_right]
    else:  # ??????????????????
        if width >= 2 * (r + img_padding):
            # ???????????????
            img1 = img1[image_bottom: image_top, image_left:image_right]
        else:
            img1 = img1[image_bottom:image_top, :]

    if crop_size is not None:
        img1 = cv2.resize(img1, (crop_size, crop_size))

    return img1

def add_black_margin(img_source, add_black_pixel_ratio):
    if isinstance(img_source, str):
        try:
            img1 = cv2.imread(img_source)
        except:
            # Corrupt JPEG data1: 19 extraneous bytes before marker 0xc4
            raise Exception("image file not found:" + img_source)
    else:
        img1 = img_source

    if img1 is None:
        raise Exception("image file error:" + img_source)

    height, width = img1.shape[:2]
    add_black_pixel = int(min(height, width) * add_black_pixel_ratio)

    img_h = np.zeros((add_black_pixel, width, 3))
    img_v = np.zeros((height + add_black_pixel*2, add_black_pixel, 3))

    img1 = np.concatenate((img_h, img1, img_h), axis=0)
    img1 = np.concatenate((img_v, img1, img_v), axis=1)

    return img1


def do_preprocess(img_source, crop_size, img_file_dest=None, add_black_pixel_ratio=0):
    if isinstance(img_source, str):
        try:
            img1 = cv2.imread(img_source)
        except:
            # Corrupt JPEG data1: 19 extraneous bytes before marker 0xc4
            raise Exception("image file not found:" + img_source)
    else:
        img1 = img_source

    if img1 is None:  #file not exists or other errors
        raise Exception("image file error:" + img_source)

    img1 = del_black_or_white(img1)

    # after delete black margin, image may be too small
    min_width_height = min(img1.shape[:2])
    if min_width_height < IMG_SMALL_THRETHOLD:
        return None

    #image too big, resize
    resize_size = crop_size * 2.5
    if min_width_height > resize_size:
        crop_ratio = resize_size / min_width_height
        img1 = cv2.resize(img1, None, fx=crop_ratio, fy=crop_ratio)

    (found_circle, x, y, r) = detect_xyr(img1)

    if add_black_pixel_ratio > 0:
        img1 = my_crop_xyr(img1, x, y, r)
        # add some black margin, for fear that duing img aug(random rotate crop) delete useful areas
        img1 = add_black_margin(img1, add_black_pixel_ratio=add_black_pixel_ratio)
        img1 = cv2.resize(img1, (crop_size, crop_size))
    else:
        img1 = my_crop_xyr(img1, x, y, r, crop_size)

    if img_file_dest is not None:
        if not os.path.exists(os.path.dirname(img_file_dest)):
            os.makedirs(os.path.dirname(img_file_dest))

        cv2.imwrite(img_file_dest, img1)

    return img1

def multi_crop(img_source, gen_times=5, add_black=True):
    if isinstance(img_source, str):
        try:
            image1 = cv2.imread(img_source)
        except:
            # Corrupt JPEG data1: 19 extraneous bytes before marker 0xc4
            raise Exception("image file not found:" + img_source)
    else:
        image1 = img_source

    if image1 is None:
        raise Exception("image file error:" + img_source)

    if add_black:
        image1 = add_black(img_source)

    list_image = [image1]

    # sometimes = lambda aug: iaa.Sometimes(0.96, aug)
    seq = iaa.Sequential([
        iaa.Crop(px=(0, min(image1.shape[0], image1.shape[1]) // 20)),  # crop images from each side by 0 to 16px (randomly chosen)
        # iaa.GaussianBlur(sigma=(0, 3.0)),  # blur images with a sigma of 0 to 3.0,
        # iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
        # sometimes(iaa.Crop(percent=(0, 0.1))),  # crop images by 0-10% of their height/width
        # shuortcut for CropAndPad

        # improve or worsen the contrast  If PCH is set to true, the process happens channel-wise with possibly different S.
        # sometimes1(iaa.ContrastNormalization((0.9, 1.1), per_channel=0.5), ),
        # change brightness of images (by -5 to 5 of original value)
        # sometimes1(iaa.Add((-6, 6), per_channel=0.5),),
        # sometimes(iaa.Affine(
        #     # scale={"x": (0.92, 1.08), "y": (0.92, 1.08)},
        #     # scale images to 80-120% of their size, individually per axis
        #     # Translation Shifts the pixels of the image by the specified amounts in the x and y directions
        #     translate_percent={"x": (-0.08, 0.08), "y": (-0.06, 0.06)},
        #     # translate by -20 to +20 percent (per axis)
        #     rotate=(0, 360),  # rotate by -45 to +45 degrees
        #     # shear=(-16, 16),  # shear by -16 to +16 degrees
        #     # order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
        #     # cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
        #     # mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        # )),
    ])

    img_results = []

    for i in range(gen_times):
        images_aug = seq.augment_images(list_image)
        img_results.append(images_aug[0])

    return img_results

def get_fundus_border(img1, threthold1 = 5, threthold2 = 180, padding = 13):
    if isinstance(img1, str):
        img1 = cv2.imread(img1)

    if img1.ndim == 2:
        img1 = np.expand_dims(img1, axis=-1)

    (found_circle, x, y, r) = detect_xyr(img1)
    if found_circle:
        left = x - r
        left = max(0, left)
        right = x + r
        right = min(right, img1.shape[1])
        bottom = y - r
        bottom = min(bottom, img1.shape[0])
        bottom = max(0, bottom)
        top = y + r

        return bottom, top, left, right

    width, height = (img1.shape[1], img1.shape[0])

    (left, bottom) = (0, 0)
    (right, top) = (img1.shape[1], img1.shape[0])

    for i in range(width):
        array1 = img1[:, i, :]
        if np.sum(array1) > threthold1 * array1.shape[0] * array1.shape[1] and \
                np.sum(array1) < threthold2 * array1.shape[0] * array1.shape[1]:
            left = i
            break
    left = max(0, left - padding)  # ???????????????

    for i in range(width - 1, 0 - 1, -1):
        array1 = img1[:, i, :]
        if np.sum(array1) > threthold1 * array1.shape[0] * array1.shape[1] and \
                np.sum(array1) < threthold2 * array1.shape[0] * array1.shape[1]:
            right = i
            break
    right = min(width, right + padding)  # ???????????????

    for i in range(height):
        array1 = img1[i, :, :]
        if np.sum(array1) > threthold1 * array1.shape[0] * array1.shape[1] and \
                np.sum(array1) < threthold2 * array1.shape[0] * array1.shape[1]:
            bottom = i
            break
    bottom = max(0, bottom - padding)

    for i in range(height - 1, 0 - 1, -1):
        array1 = img1[i, :, :]
        if np.sum(array1) > threthold1 * array1.shape[0] * array1.shape[1] and \
                np.sum(array1) < threthold2 * array1.shape[0] * array1.shape[1]:
            top = i
            break

    top = min(height, top + padding)


    return  bottom, top, left, right


# simple demo code
if __name__ == '__main__':
    img_file = '/tmp1/img2.jpg'
    if os.path.exists(img_file):
        img_processed = do_preprocess(img_file, crop_size=384)
        cv2.imwrite('/tmp1/tmp2_preprocess.jpg', img_processed)
        print('OK')
    else:
        print('file not exists!')
