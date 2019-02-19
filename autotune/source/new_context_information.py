import tools
import numpy as np
import os
from shutil import copyfile, rmtree
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, spectral_clustering, dbscan
import re
from os.path import join
import yaml
import json

project_dir = os.path.dirname(os.getcwd())
with open(join(project_dir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)

middle_data_path = join(project_dir, cfg['base_conf']['middle_data_path'])
multiple_data_path = join(project_dir, cfg['base_conf']['multiple_meeting_data_path'])

with open(join(pic_path, 'pics%s.pk' % meeting), 'rb') as f:
    iou_pic_path = pickle.load(f)
iou_vec = np.load(join(pic_path, 'vec%s.npy' % meeting))
iou_vec = iou_vec.tolist()

for folder in cfg['match']['folders']:
    folder_path = join(multiple_data_path, folder)
    with open(join(folder_path, 'wifi_info.json'), 'r') as f:
        wifi_info = json.load(f)

    peoples = []
    index = 0
    meeting_index = {}
    for k, v in wifi_info.items():
        peoples.extend(v)
        meeting_index[k] = index
        index += 1
    meetings = wifi_info.keys()
    peoples = list(set(peoples))
    people_num = len(peoples)
    meeting_num = len(wifi_info)
    meeting_array = np.zeros([meeting_num, people_num])

    for k, v in wifi_info.items():
        row = meeting_index[k]
        for n in v:
            column = peoples.index(n)
            meeting_array[row, column] = 1

    # centers = {}
    # cluster_names = []
    # center_features = []
    # for middle_folder in folder.split('_'):
    #     centers[middle_folder] = {}
    #     center_paths = tools.get_format_file(join(middle_data_path, middle_folder), 2, r'^.+_center\.npy$')
    #     for center_path in center_paths:
    #         temp_center = np.load(center_path)
    #         file_name = tools.get_parent_folder_name(center_path, 1)
    #         cluster_name = file_name.split('_')[0]
    #         centers[middle_folder][cluster_name] = temp_center
    #         cluster_names.append('%s_%s' % (middle_folder, cluster_name))
    #         center_features.append(temp_center)

    for para in cfg['context_information']['hyper_para']:
        # connectivity = np.zeros([len(cluster_names), len(cluster_names)])
        # print(para)
        # context_infor = {}
        # for i in range(len(cluster_names)):
        #     for j in range(i, len(cluster_names)):
        #         c1 = cluster_names[i].split('_')
        #         c2 = cluster_names[j].split('_')
        #         if c1[0] not in context_infor:
        #             context_infor[c1[0]] = {}
        #         context_infor[c1[0]][c2[0]] = np.linalg.norm(meeting_array[meeting_index[c1[0]]] - meeting_array[meeting_index[c2[0]]])
        #         temp = np.linalg.norm(centers[c1[0]][c1[1]] - centers[c2[0]][c2[1]]) + \
        #                para * np.linalg.norm(meeting_array[meeting_index[c1[0]]] - meeting_array[meeting_index[c2[0]]])
        #         connectivity[i, j] = temp
        #         connectivity[j, i] = temp

        # state = 0
        # while True:
        #     result = dbscan(connectivity, metric='precomputed')
        #     state += 1
        #     if len(set(result.labels_)) == people_num:
        #         break
        # result = result.labels_
        # print(connectivity)
        # result = spectral_clustering(n_clusters=people_num, affinity=connectivity)

        video_info = np.load(join(folder_path, 'mix.npy'))
        pic_paths = os.listdir(join(folder_path, 'pic'))
        pic_paths.sort(key=lambda x: int(re.split('_|\.|', x)[0]))
        final_path_result = {}
        final_feature_result = {}
        iou2index = {}
        index = 0
        for path, feature in zip(pic_paths, video_info):
            iou_num = re.split('_|\.|', path)[3]
            if iou_num not in iou2index:
                iou2index[iou_num] = index
                index += 1
            if iou2index[iou_num] in final_path_result:
                final_path_result[iou2index[iou_num]] += [path]
            else:
                final_path_result[iou2index[iou_num]] = [path]

            if iou2index[iou_num] in final_feature_result:
                final_feature_result[iou2index[iou_num]] += [feature]
            else:
                final_feature_result[iou2index[iou_num]] = [feature]

        pic_features = []
        for i in range(index):
            feature =  [np.mean(final_feature_result[i])]
            c = final_path_result[i][0].split('_')
            li = list(map(lambda x: x * para, meeting_array[meeting_index['%s_%s' % (c[1], c[2])]]))
            temp = np.concatenate((feature, li), axis=0)
            pic_features.append(temp)

        result = AgglomerativeClustering(n_clusters=people_num + 1, linkage='average').fit(pic_features)

        with open(join(folder_path, '%f_context_result.txt' % para), 'w') as f:
            for i in range(len(result.labels_)):
                for j in final_path_result[i]:
                    f.writelines('%s: %s\n' % (final_path_result[i][j], result.labels_[i]))

        # pic_features = []
        # for path, feature in zip(pic_paths, video_info):
        #     c = path.split('_')
        #     li = list(map(lambda x: x*para, meeting_array[meeting_index['%s_%s' % (c[1], c[2])]]))
        #
        #     temp = np.concatenate((feature, li), axis=0)
        #     pic_features.append(temp)

        # result = AgglomerativeClustering(n_clusters=people_num + 1, linkage='average').fit(pic_features)
        #
        # with open(join(folder_path, '%f_context_result.txt' % para), 'w') as f:
        #     for v1, v2 in zip(result.labels_, pic_paths):
        #         f.writelines('%s: %s\n' % (v1, v2))
