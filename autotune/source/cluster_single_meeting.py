import tools
import os
import yaml
from os.path import join
import json
import numpy as np
import re
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from munkres import Munkres, print_matrix
from shutil import rmtree, copyfile


project_dir = os.path.dirname(os.getcwd())
with open(join(project_dir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)

multiple_data_path = join(project_dir, cfg['base_conf']['multiple_meeting_data_path'])

cluster_num = cfg['single_meeting']['cluster_num']
folder = cfg['single_meeting']['folder']
folder_path = join(multiple_data_path, folder)
video_info = np.load(join(folder_path, 'mix.npy'))

pic_paths = os.listdir(join(folder_path, 'pic'))
pic_paths.sort(key=lambda x: int(re.split('_|\.|', x)[0]))
print('start cluster')
clusters = AgglomerativeClustering(n_clusters=cluster_num, linkage='average').fit(video_info)
print('finish cluster')

# save clusters
np.save(join(folder_path, 'clusters.npy'), clusters)

classifier_path = join(folder_path, 'classifier')
if os.path.exists(classifier_path):
    rmtree(classifier_path)
os.mkdir(classifier_path)
for i in range(cluster_num):
    os.makedirs(join(classifier_path, str(i)))

index = 0
for d in clusters.labels_:
    copyfile(join(folder_path, 'pic', pic_paths[index]), join(classifier_path, str(d), pic_paths[index]))
    index += 1
