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

project_dir = os.path.dirname(os.getcwd())
with open(join(project_dir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)

copy_pic = cfg['eval']['copy_pic']
thres = cfg['pre_process']['wifi_threshold']
# middle_data_path = join(project_dir, cfg['base_conf']['middle_data_path'])
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

pic_paths = []

counter = 0
for k in middle_pic_path:
    with open(middle_pic_path[k], 'rb') as f:
        iou_pic_path = pickle.load(f)
        pic_paths.extend(iou_pic_path)
    iou_vec = np.load(meeting_npy_paths[k])
    try:
        if counter == 0:
            video_info = iou_vec
        else:
            video_info = np.vstack((video_info, iou_vec))

        counter += 1
    except:
        print(k)

# db_re = DBSCAN(min_samples=4).fit(video_info)
# index = 0
# for d in db_re.labels_:
#     if d == -1:
#         video_info = np.delete(video_info, (index), axis=0)
#         pic_paths.pop(index)
#         continue
#     index += 1

for para in cfg['context_information']['hyper_para']:
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

    add_num_config = cfg['match']['add_person']
    for add_num in range(add_num_config[0], add_num_config[1], add_num_config[2]):
        # if meeting_num > people_num + add_num:
        #     matrix_size = meeting_num
        # else:
        #     matrix_size = people_num + add_num

        pic_people_in_meetings = np.zeros([people_num+add_num, meeting_num])
        wifi_people_in_meetings = np.zeros([people_num, meeting_num])


        for i in range(people_num):
            for name in meeting_name:
                if peoples[i] in all_meeting_people_name[name]:
                    wifi_people_in_meetings[i, meeting_index[name]] = 1


        print('start cluster')
        clusters = AgglomerativeClustering(n_clusters=people_num+add_num, linkage='average').fit(pic_features)
        print('finish cluster')

        cluster_feature = {}
        for cluster_num, feat in zip(clusters.labels_, video_info):
            if cluster_num not in cluster_feature:
                cluster_feature[cluster_num] = []
            cluster_feature[cluster_num].append(feat)

        stat = {}
        for cluster_num, pic_path in zip(clusters.labels_, pic_paths):
            parent = tools.get_parent_folder_name(pic_path[0], 3)
            tem = parent.split('_')
            if cluster_num not in stat:
                stat[cluster_num] = set([])
            stat[cluster_num].add('%s_%s' % (tem[0], tem[1]))


        res_stat = {}
        for num in stat:
            res_stat[num] = {}
            for n in stat[num]:
                res_stat[num][n] = []
        for cluster_num, pic_path in zip(clusters.labels_, pic_paths):
            parent = tools.get_parent_folder_name(pic_path[0], 3)
            tem = parent.split('_')
            res_stat[cluster_num]['%s_%s' % (tem[0], tem[1])].append(pic_path)

        for peo_no, in_meetings in stat.items():
            for m in in_meetings:
                pic_people_in_meetings[peo_no, meeting_index[m]] = 1

        cost_matrix = np.zeros([people_num+add_num, people_num+add_num])

        for i in range(people_num+add_num):
            for j in range(people_num):
                cost_matrix[i, j] = np.linalg.norm(pic_people_in_meetings[i] - wifi_people_in_meetings[j])

        soft_max = cost_matrix.max()

        for i in range(people_num+add_num):
            print(cost_matrix[i, :])


        m = Munkres()
        indexes = m.compute(cost_matrix)
        print_matrix(cost_matrix, msg='Lowest cost through this matrix:')

        save_matrix = {}
        final_res = {}
        for row, column in indexes:
            if column < people_num and row < people_num+add_num:
                name = peoples[column]
                final_res[name] = res_stat[row]
                save_matrix[name] = np.mean(np.array(cluster_feature[row]), axis=0)


                # soft_save[name] = soft_matrix[row, :]
        # save = {'peoples':peoples, 'soft':soft_save}
        #
        with open(join(project_dir, 'final_result', '%f_%d_center.pk'%(para, add_num)), 'wb') as f:
            pickle.dump(save_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)

        time = cfg['get_meeting_data']['start_time']
        time = datetime.strftime(datetime.now(), '%m-%d-%H-%M-%S')

        with open(join(project_dir, 'final_result', '%s_%f_%d.csv'%(time, para, add_num)), 'w') as f:
            for k, v in final_res.items():
                for iou in v:
                    for paths in v[iou]:
                        # for path in paths:
                        f.write('%s,%s\n'%(paths[0], k))

        if copy_pic:
            os.mkdir(join(project_dir, 'final_result', '%s_%f_%d' % (time, para, add_num)))
            for k, v in final_res.items():
                index = 0
                os.mkdir(join(project_dir, 'final_result', '%s_%f_%d'%(time, para, add_num), k))
                for iou in v:
                    for paths in v[iou]:
                        # for path in paths:
                        s_pic_name = tools.get_parent_folder_name(paths[0], 1)
                        s_meet_name = tools.get_parent_folder_name(paths[0], 3)
                        copyfile(join(middle_data_path, s_meet_name, 'mtcnn', s_pic_name), join(project_dir, 'final_result', '%s_%f_%d'%(time, para, add_num), k, '%s_%d.jpeg'%(k, index)))
                        index += 1
