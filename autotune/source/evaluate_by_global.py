import tools
import os
import yaml
from os.path import join
from sklearn.metrics import precision_recall_fscore_support
from datetime import datetime
import re
import csv
import pickle
import re


project_dir = os.path.dirname(os.getcwd())
with open(join(project_dir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)

middle_data_path = cfg['base_conf']['middle_data_path']
meeting_paths = os.listdir(middle_data_path)
meeting_paths = list(filter(lambda x: x != 'time' and x != 'tmp' and x != 'problem data', meeting_paths))
load_label = cfg['eval']['load_label']

true_label = {}

if load_label:
    with open(join(project_dir, 'final_result', 'true_label.pk'), 'rb') as f:
        true_label = pickle.load(f)
else:
    for meet in meeting_paths:
        name_paths = os.listdir(join(middle_data_path, meet, 'classifier'))
        for name in name_paths:
            if os.path.isfile(join(middle_data_path, meet, 'classifier',name)):
                continue
            for pic in os.listdir(join(middle_data_path, meet, 'classifier', name)):
                true_label[pic] = name

    with open(join(project_dir, 'final_result', 'true_label.pk'), 'wb') as f:
        pickle.dump(true_label, f, protocol=pickle.HIGHEST_PROTOCOL)
peoples = []
for k, v in true_label.items():
    if v!='other':
        peoples.append(v)
peoples = list(set(peoples))


result_data_path = join(project_dir, 'final_result')
paths = os.listdir(result_data_path)
paths = list(filter(lambda x: re.match(r'.+\.csv', x), paths))
print(true_label)

tem_label = {}
for k,v in true_label.items():
    temp = k.split('_')
    tem_label["%s_%s_%s"%(temp[0], temp[1], temp[2])] = v
true_label = tem_label
max_person_sta = {}

for p in peoples:
    max_person_sta[p] = 0

final_report_path = join('../analysis', '%s_report.csv'%(datetime.strftime(datetime.now(), '%m-%d-%H-%M-%S')))
with open(final_report_path, 'w') as final_report:
    final_report.write('file_name,hyper_para,add_person,size,miss_num,accuarcy')
    for p in peoples:
        # print(cfg['mac_name'][p])
        final_report.write(',%s,%s_size'%(p,p))
    final_report.write('\n')
    for path in paths:
        temp_path = os.path.split(path)[-1]
        final_report.write('%s,' % temp_path)
        temp_path = re.split(r'-|_', temp_path)
        final_report.write('%s,%s,'%(temp_path[4], temp_path[5][:-4]))

        true = 0
        miss = 0
        all = 0
        person_sta_true = {}
        person_sta_false = {}
        with open(join(project_dir, 'final_result', path)) as f:
            result = csv.reader(f, delimiter=',')
            for row in result:
                # if row[1] == "NIKi":
                #     continue
                r = row[0].split('/')
                if row[1] not in person_sta_false:
                    person_sta_false[row[1]] = 0
                if row[1] not in person_sta_true:
                    person_sta_true[row[1]] = 0
                temp = r[-1].split('_')
                r[-1] = '%s_%s_%s' % (temp[0], temp[1], temp[2])
                if r[-1] in true_label:
                    all += 1
                    if true_label[r[-1]] == row[1]:
                        true += 1
                        person_sta_true[row[1]] += 1
                    else:
                        # print('label %s, predict %s'%(true_label[r[-1]], row[1]))
                        person_sta_false[row[1]] += 1
                else:
                    miss += 1
                    print('miss pic %s'%r[-1])
            print("true total pic %d, miss pic %d"%(all, miss))
            final_report.write('%d,%d,' % (all, miss))
        acc = true/all
        final_report.write('%f' % acc)
        print('acc %f'%acc)
        for p in peoples:
            if person_sta_true[p] + person_sta_false[p] == 0:
                acc_per = 0
            else:
                acc_per = person_sta_true[p] / (person_sta_true[p] + person_sta_false[p])
            print("%s : %f size: %d"%(p, acc_per, person_sta_true[p]+person_sta_false[p]))
            if acc_per>max_person_sta[p]:
                max_person_sta[p] = acc_per

        for p in peoples:
            if p in person_sta_true:
                if person_sta_true[p] + person_sta_false[p] == 0:
                    acc_per = 0
                else:
                    acc_per = person_sta_true[p] / (person_sta_true[p] + person_sta_false[p])

                final_report.write(',%f,%d' % (person_sta_true[p]/(person_sta_true[p]+person_sta_false[p]),
                                           person_sta_true[p] + person_sta_false[p]))
            else:
                final_report.write(',0,0')

        print('\n')
        final_report.write('\n')
        # name = path[:-4]
        # os.rename(join(project_dir, 'final_result', path), join(project_dir, 'final_result', '%s-%f.csv'%(name, acc)))

    for k, v in max_person_sta.items():
        print("%s max acc rate: %f"%(k, v))

