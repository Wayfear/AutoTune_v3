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
from sklearn.preprocessing import normalize


def get_cluster_belong_to_who(paths, true_label):
    li = []
    miss = 0
    for path in paths:
        path = tools.get_parent_folder_name(path[0], 1)
        # temp = path.split('_')
        # temp = '%s_%s_%s' % (temp[0], temp[1], temp[2])
        if path in true_label:
            li.append(true_label[path])
        else:
            miss += 1
            print('miss pic %s' % path)
    t_li = li.copy()
    li = list(set(li))
    max = -1
    if len(t_li) == 0:
        return 0, ''
    for l in li:
        if t_li.count(l)>max:
            ma = t_li.count(l)
            peo = l
    return ma/len(t_li), peo


def dst2reliable(x, min_dst=0.4, max_dst=0.95):
    sig = 1/((max_dst-min_dst)**2)
    for i in range(len(x)):
        if x[i] <= min_dst:
            x[i] = 1
        elif x[i] >= max_dst:
            x[i] = 0
        else:
            x[i] = 1.0 - sig*((x[i]-min_dst)**2)
    return x


def check_scale(x, min=0, max=1):
    for i in range(len(x)):
        if x[i] < min:
            x[i] = 0
        elif x[i] > max:
            x[i] = 1
    return x


load_label = True
err_rate = 0.5
cycle_num = 2

project_dir = os.path.dirname(os.getcwd())
with open(join(project_dir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)

copy_pic = cfg['eval']['copy_pic']
load_label = cfg['eval']['load_label']

thres = cfg['pre_process']['wifi_threshold']
middle_data_path = cfg['base_conf']['middle_data_path']
meeting_npy_paths = tools.get_meeting_and_path(middle_data_path, r'.+\.npy')
middle_pic_path = tools.get_meeting_and_path(middle_data_path, r'.+\.pk')
middle_people_path = tools.get_meeting_and_path(middle_data_path, r'.+\.png')
all_meeting_people_name = tools.get_meeting_people_name(middle_data_path, thres)
meeting_paths = os.listdir(middle_data_path)

if load_label:
    with open(join(project_dir, 'final_result', 'true_label.pk'), 'rb') as f:
        true_label = pickle.load(f)
else:
    true_label = {}
    for meet in meeting_paths:
        name_paths = os.listdir(join(middle_data_path, meet, 'classifier'))
        for name in name_paths:
            if os.path.isfile(join(middle_data_path, meet, 'classifier',name)):
                continue
            for pic in os.listdir(join(middle_data_path, meet, 'classifier', name)):
                true_label[pic] = name


peoples = []

meeting_index = {}
index = 0
meeting_name = []
center_file = '0.050000_53_center.pk'

with open(join(project_dir, 'final_result', center_file), 'rb') as f:
    centers = pickle.load(f)

for k, v in all_meeting_people_name.items():
    peoples.extend(v)
    meeting_index[k] = index
    index += 1
    meeting_name.append(k)
peo_list = peoples.copy()
peoples = list(set(peoples))
eff_peos = []
peo_count = {}
for p in peoples:
    ct = peo_list.count(p)
    peo_count[p] = ct
    if ct>2:
        eff_peos.append(p)

people_num = len(peoples)
meeting_num = len(all_meeting_people_name)

context_infor = np.zeros([meeting_num, people_num])

for k, v in all_meeting_people_name.items():
    for name in v:
        context_infor[meeting_index[k], peoples.index(name)] = 1

video_info = np.empty([0, 512])
pic_paths = []

for k in middle_pic_path:
    with open(middle_pic_path[k], 'rb') as f:
        iou_pic_path = pickle.load(f)
    iou_vec = np.load(meeting_npy_paths[k])
    num = 3*len(all_meeting_people_name[k])

    people_dict = None
    if k in cycle_people:
        with open(cycle_people[k], 'rb') as f:
            people_dict = pickle.load(f)
            people_in_this_meeting = set(people_dict.keys())
    else:
        people_in_this_meeting = set(all_meeting_people_name[k])
    if len(iou_vec) < num:
        num = len(iou_vec)
    if num >= 3:
        clusters = AgglomerativeClustering(n_clusters=num, linkage='average').fit(iou_vec)
    else:
        continue
    cluster_feature = {}
    cluster_path = {}
    for cluster_num, feat, path in zip(clusters.labels_, iou_vec, iou_pic_path):
        if cluster_num not in cluster_feature:
            cluster_feature[cluster_num] = []
            cluster_path[cluster_num] = []
        cluster_feature[cluster_num].append(feat)
        cluster_path[cluster_num].append(path)
    save_matrix = {}
    for num in cluster_feature:
        save_matrix[num] = np.mean(np.array(cluster_feature[num]), axis=0)
    people = ''
    predict_peoples = {}
    for num in save_matrix:
        min_dis = 10000
        for p in eff_peos:
            dis = np.linalg.norm(save_matrix[num] - centers[p])
            if min_dis > dis:
                min_dis = dis
                people = p
        rate, true_pe = get_cluster_belong_to_who(cluster_path[num], true_label)
        print('predict %s, true %s, rate %f, min_dis %f'%(people, true_pe, rate, min_dis))
        if people not in predict_peoples:
            predict_peoples[people] = min_dis
        else:
            if predict_peoples[people] > min_dis:
                predict_peoples[people] = min_dis
    prediect_peo_list = list(predict_peoples.keys())
    predict_value_list = dst2reliable(list(predict_peoples.values()))
    for i in range(len(prediect_peo_list)):
        predict_peoples[prediect_peo_list[i]] = predict_value_list[i]
    new_peops = list(people_in_this_meeting | set(prediect_peo_list))

    pre_vec = np.zeros(len(new_peops)).astype(np.float64)
    if people_dict is None:
        for i in range(len(new_peops)):
            if new_peops[i] in people_in_this_meeting:
                pre_vec[i] = 1
    else:
        for i in range(len(new_peops)):
            if new_peops[i] in people_dict:
                pre_vec[i] = people_dict[new_peops[i]]

    now_vec = np.array(len(new_peops)*[-1]).astype(np.float64)
    for i in range(len(new_peops)):
        if new_peops[i] in predict_peoples:
            now_vec[i] = predict_peoples[new_peops[i]]

    now_vec = check_scale(pre_vec + err_rate*now_vec)
    result = dict(zip(new_peops, now_vec))
    result = {k: v for k, v in result.items() if v != 0}
    path = os.path.split(middle_pic_path[k])
    with open(join(path[0], 'peoples.pk%d' % cycle_num), 'wb') as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
