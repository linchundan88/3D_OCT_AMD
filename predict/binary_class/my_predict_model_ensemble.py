import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import sys
sys.path.append(os.path.abspath('../..'))
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from libs.dataset.my_dataset_torchio import Dataset_CSV_test
from libs.neural_networks.helper.my_predict_binary_class import predict_multiple_models
import shutil
from libs.neural_networks.model.my_get_model import get_model

filename_csv = os.path.join(os.path.abspath('../..'), 'datafiles', 'v1', '3D_OCT_AMD.csv')
dir_original = '/disk1/3D_OCT_AMD/2021_4_22/original/'
dir_preprocess = '/disk1/3D_OCT_AMD/2021_4_22/preprocess/128_128_128/'
dir_dest = '/tmp2/3D_OCT_AMD/3D_OCT_AMD_confusion_files_2021_7_30/'
export_confusion_files = False

threshold = 0.5

models_dicts = []

'''
model_name = 'cls_3d'
# model_file = '/tmp2/2021_6_6/v2/ModelsGenesis/0/epoch13.pth'
model_file = os.path.join(os.path.abspath('../..'), 'trained_models', 'binary_class', 'cls_3d.pth')
model = get_model(model_name, 1, model_file=model_file)
image_shape = (64, 64)
ds_test = Dataset_CSV_test(csv_file=filename_csv, image_shape=image_shape,
                           depth_start=0, depth_interval=2, test_mode=True)
loader_test = DataLoader(ds_test, batch_size=32, pin_memory=True, num_workers=4)
model_dict = {'model': model, 'weight': 1, 'dataloader': loader_test}
models_dicts.append(model_dict)
'''

model_name = 'medical_net_resnet50'
model_file = os.path.join(os.path.abspath('../..'), 'trained_models', 'binary_class', 'medical_net_resnet50.pth')
model = get_model(model_name, 1, model_file=model_file)
image_shape = (64, 64)
ds_test = Dataset_CSV_test(csv_file=filename_csv, image_shape=image_shape,
                           depth_start=0, depth_interval=2, test_mode=True)
loader_test = DataLoader(ds_test, batch_size=32, pin_memory=True, num_workers=4)
model_dict = {'model': model, 'weight': 1, 'dataloader': loader_test}
models_dicts.append(model_dict)

probs, probs_ensembling = predict_multiple_models(models_dicts)
labels_pd = np.array(probs_ensembling)
labels_pd[labels_pd > threshold] = 1
labels_pd[labels_pd <= threshold] = 0

df = pd.read_csv(filename_csv)
(image_files, labels) = list(df['images']), list(df['labels'])
from sklearn.metrics import confusion_matrix
cf = confusion_matrix(labels, labels_pd)
print(cf)

if export_confusion_files:
    from libs.neural_networks.helper.my_export_confusion_files import export_confusion_files_binary_class
    export_confusion_files_binary_class(image_files, labels, probs_ensembling, dir_original, dir_preprocess, dir_dest, threshold)
    print('export confusion files ok!')


