import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import sys
sys.path.append(os.path.abspath('../..'))
from torch.utils.data import DataLoader
import pandas as pd
from libs.dataset.my_dataset_torchio import Dataset_CSV_test
from libs.neural_networks.helper.my_predict_binary_class import predict_single_model
from libs.neural_networks.model.my_get_model import get_model

filename_csv = os.path.join(os.path.abspath('../..'), 'datafiles', 'v1', '3D_OCT_AMD.csv')
dir_original = '/disk1/3D_OCT_AMD/2021_4_22/original/'
dir_preprocess = '/disk1/3D_OCT_AMD/2021_4_22/preprocess/128_128_128/'
dir_dest = '/tmp2/3D_OCT_AMD/3D_OCT_AMD_confusion_files_2021_7_30/'
export_confusion_files = False

threshold = 0.5

model_name = 'medical_net_resnet50'
model_file = os.path.join(os.path.abspath('../..'), 'trained_models', 'binary_class', 'medical_net_resnet50.pth')
# model_name = 'cls_3d'
# model_file = os.path.join(os.path.abspath('../..'), 'trained_models', 'binary_class', 'cls_3d.pth')
image_shape = (64, 64)
model = get_model(model_name, 1, model_file=model_file)

ds_test = Dataset_CSV_test(csv_file=filename_csv, image_shape=image_shape,
                           depth_start=0, depth_interval=2, test_mode=True)
loader_test = DataLoader(ds_test, batch_size=32, pin_memory=True, num_workers=4)

(probs, labels_pd) = predict_single_model(model, loader_test, activation='sigmoid', threshold=0.5)

df = pd.read_csv(filename_csv)
image_files, labels_gt = list(df['images']), list(df['labels'])
from sklearn.metrics import confusion_matrix
print(confusion_matrix(labels_gt, labels_pd))

if export_confusion_files:
    from libs.neural_networks.helper.my_export_confusion_files import export_confusion_files_binary_class
    export_confusion_files_binary_class(image_files, labels_gt, probs, dir_original, dir_preprocess, dir_dest, threshold)
    print('export confusion files ok!')