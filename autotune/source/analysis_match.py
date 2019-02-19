
import tools
import os
import yaml
from os.path import join
import json
import numpy as np
import re
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from munkres import Munkres, print_matrix
import pickle
from datetime import datetime, timedelta
from shutil import copyfile, rmtree

copy_pic = False

add_nums = [1,20,2]
paras = [0, 0.05, 0.1, 0.15]

project_dir = os.path.dirname(os.getcwd())
with open(join(project_dir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)

thres = cfg['pre_process']['wifi_threshold']
middle_data_path = cfg['base_conf']['middle_data_path']
meeting_npy_paths = tools.get_meeting_and_path(middle_data_path, r'.+\.npy')
middle_pic_path = tools.get_meeting_and_path(middle_data_path, r'.+\.pk')
middle_people_path = tools.get_meeting_and_path(middle_data_path, r'.+\.png')
all_meeting_people_name = tools.get_meeting_people_name(middle_data_path, thres)
meeting_paths = os.listdir(middle_data_path)


peoples = []

meeting_index = {}
index = 0
meeting_name = []


for k, v in all_meeting_people_name.items():
    peoples.extend(v)
    meeting_index[k] = index
    index += 1
    meeting_name.append(k)
peoples = list(set(peoples))
people_num = len(peoples)
meeting_num = len(all_meeting_people_name)

context_infor = np.zeros([meeting_num, people_num])

for k, v in all_meeting_people_name.items():
    for name in v:
        context_infor[meeting_index[k], peoples.index(name)] = 1

video_info = np.empty([0, 128])
pic_paths = []

for k in middle_pic_path:
    with open(middle_pic_path[k], 'rb') as f:
        iou_pic_path = pickle.load(f)
        pic_paths.extend(iou_pic_path)
    iou_vec = np.load(meeting_npy_paths[k])
    try:
        video_info = np.vstack((video_info, iou_vec))
    except:
        print(k)

for para in paras:
    pic_features = []
    print('start translate')
    print(len(pic_paths))
    for feature, path in zip(video_info, pic_paths):
        parent = tools.get_parent_folder_name(path[0], 3)
        c = parent.split('_')
        li = list(map(lambda x: x * para, context_infor[meeting_index['%s_%s' % (c[0], c[1])]]))
        temp = np.concatenate((feature, li), axis=0)
        pic_features.append(temp)
    print('finish translate')

    for add_num in add_nums:
        if meeting_num > people_num + add_num:
            matrix_size = meeting_num
            print('using meeting number...')
        else:
            matrix_size = people_num + add_num

        pic_people_in_meetings = np.zeros([matrix_size, matrix_size])
        wifi_people_in_meetings = np.zeros([matrix_size, matrix_size])

        for i in range(people_num):
            for name in meeting_name:
                if peoples[i] in all_meeting_people_name[name]:
                    wifi_people_in_meetings[i, meeting_index[name]] = 1


        print('start cluster')
        clusters = AgglomerativeClustering(n_clusters=people_num + add_num, linkage='average').fit(pic_features)
        print('finish cluster')

        if not os.path.exists(join(project_dir, 'global_cluster')):
            os.mkdir(join(project_dir, 'global_cluster'))

        new_path = '%d_%f' % (add_num, para)
        if os.path.exists(join(project_dir, 'global_cluster', new_path)):
            rmtree(join(project_dir, 'global_cluster', new_path))

        os.mkdir(join(project_dir, 'global_cluster', new_path))

        for i in range(people_num + add_num):
            os.mkdir(join(project_dir, 'global_cluster', new_path, str(i)))

        for cluster_num, pic_path in zip(clusters.labels_, pic_paths):
            file_name = tools.get_parent_folder_name(pic_path[0], 1)
            copyfile(pic_path[0], join(project_dir, 'global_cluster', new_path, str(cluster_num), file_name))




