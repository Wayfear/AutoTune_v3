import yaml
import os
from os.path import join
import shutil

project_dir = os.path.dirname(os.getcwd())
with open(join(project_dir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg['base_conf']['gpu_num'])
import tensorflow as tf


import tensorflow as tf
from libs import facenet
import os
import cv2
import numpy as np
from shutil import copyfile, rmtree
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import re
import tools

import yaml
import pickle


add_people = 0

dest_parent_dir = cfg['base_conf']['god_train_path']
middle_data_path = cfg['base_conf']['middle_data_path']
thres = cfg['pre_process']['wifi_threshold']
drop_path = tools.get_meeting_people_name(middle_data_path, thres)
for k, v in drop_path.items():
    if len(v[0])==0:
        rmtree(join(middle_data_path, k))

# clean existing god_train
if os.path.exists(dest_parent_dir):
    for label_dir in os.listdir(dest_parent_dir):
        expand_label_dir = join(dest_parent_dir, label_dir)
        rmtree(expand_label_dir)

# rebuild god_train db
meetings_dir = [o for o in os.listdir(middle_data_path) if 'time' not in o]
for meeting in meetings_dir:
    classifier_dir = join(middle_data_path, meeting, 'classifier')
    labels_dir = [o for o in os.listdir(classifier_dir)
                    if os.path.isdir(os.path.join(classifier_dir,o))]
    for label_dir in labels_dir:
        expand_label_dir = join(classifier_dir, label_dir)
        expand_dest_dir = join(dest_parent_dir, label_dir)
        # check whether dest. dir exists
        if not os.path.exists(expand_dest_dir):
            os.makedirs(expand_dest_dir)

        for f in os.listdir(expand_label_dir):
            full_file_name = join(expand_label_dir, f)
            shutil.copy2(full_file_name, expand_dest_dir)


# remove label dir of other(s)
if os.path.exists(dest_parent_dir):
    for label_dir in os.listdir(dest_parent_dir):
        if 'other' in label_dir:
            expand_label_dir = join(dest_parent_dir, label_dir)
            rmtree(expand_label_dir)
