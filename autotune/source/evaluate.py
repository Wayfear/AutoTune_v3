import tools
import os
import yaml
from os.path import join
from sklearn.metrics import precision_recall_fscore_support
from datetime import datetime
import re

project_dir = os.path.dirname(os.getcwd())
with open(join(project_dir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)

multiple_data_path = join(project_dir, cfg['base_conf']['multiple_meeting_data_path'])
true_data_path = join(project_dir, cfg['base_conf']['true_data_path'])


for folder in cfg['evaluate']['folders']:
    final_rate = {}
    result_paths = tools.get_format_file(join(multiple_data_path, folder), 1, r'.+_final_result\.txt')

    temp_cfg = folder.split('_')
    start_time = datetime.strptime(temp_cfg[0], '%m-%d-%H-%M-%S')
    end_time = datetime.strptime(temp_cfg[1], '%m-%d-%H-%M-%S')

    meetings_folders = os.listdir(true_data_path)
    meetings_folders = list(filter(lambda x: re.match(r'.+?_.+', x), meetings_folders))
    time_folders = []
    for folder_name in meetings_folders:
        times = folder_name.split('_')
        time_folders.append([datetime.strptime(times[0], '%m-%d-%H-%M-%S'),
                             datetime.strptime(times[1], '%m-%d-%H-%M-%S'), folder_name])
        time_folders.sort(key=lambda x: x[0])

    meetings_folders = list(filter(lambda x: x[0] >= start_time and x[1] <= end_time, time_folders))
    meetings_folders = list(map(lambda x: x[2], meetings_folders))
    paths = []
    for meetings_folder in meetings_folders:
        pic_paths = tools.get_format_file(join(true_data_path, meetings_folder), 2, r'.+\.jpeg')
        paths.extend(pic_paths)

    true_data = {}
    for path in paths:
        path_data = path.split('/')
        people_name = path_data[-2]
        pic_name = path_data[-1]
        meeting_name = path_data[-3]
        true_data['%s_%s' % (meeting_name, pic_name)] = people_name

    for result_file in result_paths:
        temp = result_file.split('/')
        temp = temp[-1].split('_')
        if 'all' not in cfg['evaluate']['filter'] and int(temp[0]) not in cfg['evaluate']['filter']:
            continue
        if temp[0] not in final_rate:
            final_rate[temp[0]] = {}
        temp_true_data = true_data.copy()
        predict_result = tools.paser_result_file(result_file)
        keys = list(predict_result.keys())
        sklearn_true_data = []
        sklearn_predict_data = []
        for k in keys:
            sklearn_true_data.append(temp_true_data[k])
            sklearn_predict_data.append(predict_result[k])
        sklearn_result = precision_recall_fscore_support(sklearn_true_data, sklearn_predict_data, average='weighted')

        for pic, name in predict_result.items():
            if temp_true_data[pic] == name:
                temp_true_data[pic] = '1'
            else:
                # print('pic_name %s, pre_name %s, true_name %s' % (pic, name, temp_true_data[pic]))
                temp_true_data[pic] = '0'

        pre_true = 0
        pre_false = 0
        un_pre = 0
        for pic in temp_true_data:
            if temp_true_data[pic] == '1':
                pre_true += 1
            elif temp_true_data[pic] == '0':
                pre_false += 1
            else:
                un_pre += 1
        pt = float(pre_true)/float(pre_true + pre_false)
        up = float(un_pre)/float(pre_true + pre_false + un_pre)
        final_rate[temp[0]][temp[1]] = '%f_%d' % (pt, un_pre)
        print(result_file)
        print(sklearn_result)
        print('predict truth %f, unpredcit %f, total un predict pic %d\n' % (pt, up, un_pre))
label_x = list(final_rate.keys())
label_y = list(final_rate[label_x[0]].keys())
label_x.sort(key=lambda x: int(x))
label_y.sort(key=lambda x: float(x))
with open('analysis.csv', 'w') as f:
    s = ''
    for y in label_y:
        s += ', %s' % y
    f.write(s + '\n')
    for x in label_x:
        f.write(x)
        for y in label_y:
            try:
                f.write(',' + final_rate[x][y])
            except:
                f.write(',')
        f.write('\n')