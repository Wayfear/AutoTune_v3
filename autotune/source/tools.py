
import os
import re
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import itertools
import datetime
from pytz import utc, timezone
from datetime import datetime, timedelta
import json
from matplotlib import cm
from numpy import arange, array, zeros
import pandas as pd
from os.path import join
import multiprocessing
from functools import reduce
from shutil import copyfile


def get_format_file(root_path, piles_num, pattern):
    paths = [root_path]
    for i in range(piles_num):
        paths = list(filter(lambda x: not os.path.isfile(x), paths))
        num = 0
        te = paths.copy()
        for path in te:
            temp = os.listdir(path)
            temp = list(map(lambda x: os.path.join(path, x), temp))
            paths.extend(temp)
            num += 1
        for k in range(num):
            paths.pop(0)
    paths = list(filter(lambda x: os.path.isfile(x) and re.match(pattern, x), paths))
    return paths


def get_parent_folder_name(direction, num=2):
    return direction.split('/')[-num]


def get_meeting_and_path(path, pattern):
    paths = get_format_file(path, 2, pattern)
    result = {}
    for path in paths:
        result[get_parent_folder_name(path, 2)] = path
    return result


def get_meeting_and_path_list(path, pattern):
    paths = get_format_file(path, 2, pattern)
    result = {}
    for path in paths:
        parent = get_parent_folder_name(path, 2)
        if parent not in result:
            result[parent] = []
        result[parent].append(path)
    return result


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def paser_wifi_file(path, threshold, interval):
    result = {}
    statistics = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        bl = 0
        is_start = True
        for line in lines:
            line = line.strip('\n')
            l = line.split('\t')
            if l[0] == 'Mar 22, 2018 22:40:00.389617000':
                t = 1
            if len(l) == 2 or l[2] == '':
                continue
            try:
                strength = int(l[2])
            except:
                nums = l[2].split(',')
                strength = max(nums, key=lambda x: int(x))
                strength = int(strength)
            if strength < threshold:
                continue
            if l[1] not in statistics:
                statistics[l[1]] = {}
            time = l[0][:-3]
            try:
                time = datetime.strptime(time, '%b %d, %Y %H:%M:%S.%f')
            except:
                time = l[0][:-7]
                time = datetime.strptime(time, '%b %d, %Y %H:%M:%S.%f')
            if is_start:
                start_time = time
                is_start = False
                last_time = start_time
            t = (time - last_time).total_seconds()
            if t < interval:
                if bl not in statistics[l[1]]:
                    statistics[l[1]][bl] = 1
                else:
                    statistics[l[1]][bl] += 1
            else:
                bl += int(t / interval)
                last_time += timedelta(seconds=int(t / interval) * interval)
            if l[1] in result:
                result[l[1]].append({time: strength})
            else:
                result[l[1]] = [{time: strength}]
    return result, statistics


def plot_wifi_pic(data, y_classes, pic_name):
    fig, ax = plt.subplots()
    fig.set_size_inches(100, 20)
    ax.imshow(data, interpolation='nearest', cmap=cm.Blues)
    tick_marks_y = arange(len(y_classes))
    plt.yticks(tick_marks_y, y_classes)
    plt.ylabel('MAC Address')
    plt.xlabel('Time')
    plt.savefig(pic_name)


def read_mac_list(path):
    mac_address = pd.read_csv(path, header=None)
    mac_name = {}
    for k, v in zip(mac_address.ix[:, 0], mac_address.ix[:, 1]):
        mac_name[v] = k
    return mac_name


def get_meeting_people_num(path, thres):
    wifi_info_paths = get_format_file(path, 2, r'.+' + re.escape(str(thres)) + r'.+\.png$')
    result = {}
    for wifi in wifi_info_paths:
        name = get_parent_folder_name(wifi, 1)
        meeting = get_parent_folder_name(wifi, 2)
        result[meeting] = len(name.split('_')) - 1
    return result


def get_meeting_people_name(path, thres):
    wifi_info_paths = get_format_file(path, 2, r'.+' + re.escape(str(thres)) + r'.+\.png$')
    result = {}
    for wifi in wifi_info_paths:
        name = get_parent_folder_name(wifi, 1)
        meeting = get_parent_folder_name(wifi, 2)
        result[meeting] = re.split('_|\.|', name)[1:-1]

    return result


def paser_result_file(path):
    result = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            if len(line) < 1:
                continue
            l = line.split(': ')

            # temp = l[1].split('_')
            # result['%s_%s_%s' % (temp[-4], temp[-3], temp[-1])] = l[0]
            result[l[1]] = l[0]
    return result


def simple_paser_result_file(path):
    result = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            if len(line) < 1:
                continue
            l = line.split(': ')
            temp = l[1].split('_')
            if l[0] not in result:
                result[l[0]] = set([])
            result[l[0]].add('%s_%s' % (temp[-3], temp[-2]))
    return result


def parse_wifi_line(line):
    line = line.strip('\n')
    l = line.split('\t')
    if len(l) < 3 or l[2] == '':
        return []
    try:
        strength = int(l[2])
    except:
        nums = l[2].split(',')
        strength = max(nums, key=lambda x: int(x))
        strength = int(strength)
    time = l[0][:-3]
    try:
        time = datetime.strptime(time, '%b %d, %Y %H:%M:%S.%f')
    except:
        time = l[0][:-7]
        time = datetime.strptime(time, '%b %d, %Y %H:%M:%S.%f')
    if len(l[1]) != 17:
        return []
    return [time, l[1], strength]


def split_wifi_file_by_duration(start_time, minutes, path, result_path=None, folder=False):
    duration = timedelta(minutes=minutes)
    if result_path is None:
        result_path = path.split('.')[0]
        if not os.path.exists(result_path):
            os.mkdir(result_path)
    temp_start = start_time
    temp_end = start_time + duration
    file_name = '%s_%s' % (temp_start.strftime("%m-%d-%H-%M-%S"),
                                         temp_end.strftime("%m-%d-%H-%M-%S"))
    if folder:
        if not os.path.exists(join(result_path, file_name)):
            os.mkdir(join(result_path, file_name))
        split_f = open(join(result_path, file_name, '%s.txt' % file_name), 'w')
    else:
        split_f = open(join(result_path, '%s.txt' % file_name), 'w')
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line_result = parse_wifi_line(line)
            if len(line_result) == 0:
                continue
            if line_result[0] > temp_start:
                while True:
                    if line_result[0] <= temp_end:
                        if split_f is None:
                            file_name = '%s_%s' % (temp_start.strftime("%m-%d-%H-%M-%S"),
                                                   temp_end.strftime("%m-%d-%H-%M-%S"))
                            if folder:
                                if not os.path.exists(join(result_path, file_name)):
                                    os.mkdir(join(result_path, file_name))
                                split_f = open(join(result_path, file_name, '%s.txt' % file_name), 'w')
                            else:
                                split_f = open(join(result_path, '%s.txt' % file_name), 'w')
                        split_f.write('%s000\t%s\t%d\n' % (line_result[0].strftime('%b %d, %Y %H:%M:%S.%f'), line_result[1], line_result[2]))
                        break
                    else:

                        split_f.close()
                        split_f = None
                        while line_result[0] >= temp_end:
                            temp_start = temp_end
                            temp_end = temp_start + duration


def get_nearby_time(minutes, solve_time):
    times = solve_time.minute // minutes
    return datetime(year=solve_time.year, month=solve_time.month,
                             day=solve_time.day,hour= solve_time.hour, minute=times * minutes)


def parse_wifi_file_by_duration(minutes, path):
    duration = timedelta(minutes=minutes)
    names = re.split(r'\/|\.', path)[-2]
    names = names.split('_')
    if names[0] == 'desktop':
        names = names[1:]
    if len(names) == 2:
        names = names[0].split('-')
        names.insert(0, '2017')
    print(names)
    start_time = datetime(year=int(names[0]), month=int(names[1]),
                                   day=int(names[2]), hour=int(names[3]), minute=int(names[4]))
    start_time = get_nearby_time(minutes, start_time)
    end_time = start_time + duration
    result = []
    with open(path, 'r') as f:
        lines = f.readlines()
        temp_record = {start_time: {}}
        for line in lines:
            line_result = parse_wifi_line(line)
            if len(line_result) == 0:
                continue
            if line_result[0] > start_time:
                while True:
                    if line_result[0] <= end_time:
                        if line_result[1] not in temp_record[start_time]:
                            temp_record[start_time][line_result[1]] = []
                        temp_record[start_time][line_result[1]].append(int(line_result[2]))
                        break
                    else:
                        result.append(temp_record)
                        start_time = end_time
                        end_time = start_time + duration
                        temp_record = {start_time: {}}
    return result


def filter_wifi_result(wifi_result, threshold):
    for times in wifi_result:
        for key, value in times.items():
            for mac, strengths in value.items():
                value[mac] = list(filter(lambda x: x > threshold, strengths))
            times[key] = dict(filter(lambda x: len(x[1]) > 0, value.items()))
    return wifi_result


def draw_wifi_distribution(filter_result, output_path, mac_name, color=cm.Blues):
    x_labels = []
    y_labels = set([])
    for times_bags in filter_result:
        for key, value in times_bags.items():
            x_labels.append(key)
            for mac in value:
                y_labels.add(mac)
    row = len(y_labels)
    y_labels = list(y_labels)
    x_labels = list(map(lambda x: x.strftime("%H:%M"), x_labels))
    column = len(x_labels)
    arr = np.zeros([row, column])
    for i in range(len(filter_result)):
        for key, value in filter_result[i].items():
            for mac, strengths in value.items():
                arr[y_labels.index(mac), i] = len(strengths) + 1000
    fig, ax = plt.subplots()
    fig.set_size_inches(100, 20)
    ax.imshow(arr, interpolation='nearest', cmap=color)
    y_labels = list(map(lambda x: mac_name[x], y_labels))
    # ax.yaxis.tick_right()
    # ax.yaxis.set_label_position("right")
    # ax.set_yticklabels(y_labels)
    tick_marks_x = arange(-0.5, len(x_labels)-1, 1)
    plt.xticks(tick_marks_x, x_labels)
    tick_marks_y = arange(len(y_labels))

    plt.yticks(tick_marks_y, y_labels)
    plt.ylabel('Name')
    plt.xlabel('Time')
    plt.grid()

    plt.savefig(output_path)
    return y_labels


def get_name_by_list(names, separate='_'):
    name = ''
    for n in names:
        name += (n + separate)
    return name[:-1]


def worker(lines):
    result = []
    for line in lines:
        re = parse_wifi_line(line)
        if len(re) == 3:
            result.append(re)
    return result


def parse_wifi_file_by_duration_multi_process(minutes, path, numthreads=8, numlines=100):
    lines = open(path).readlines()
    num_cpu_avail = multiprocessing.cpu_count()
    numthreads = min(num_cpu_avail, numthreads)
    pool = multiprocessing.Pool(processes=numthreads)
    result_list = pool.map(worker,
        (lines[line:line+numlines] for line in range(0, len(lines), numlines)))
    pool.close()
    pool.join()
    result = []
    for re in result_list:
        result.extend(re)
    result.sort(key=lambda x: x[0])
    if len(result) == 0:
        return []

    duration = timedelta(minutes=minutes)
    start_time = get_nearby_time(minutes, result[0][0])
    end_time = start_time + duration
    final_result = []
    temp_record = {start_time: {}}
    for line_result in result:
        if line_result[0] > start_time:
            while True:
                if line_result[0] <= end_time:
                    if line_result[1] not in temp_record[start_time]:
                        temp_record[start_time][line_result[1]] = []
                    temp_record[start_time][line_result[1]].append(int(line_result[2]))
                    break
                else:
                    final_result.append(temp_record)
                    start_time = end_time
                    end_time = start_time + duration
                    temp_record = {start_time: {}}
    return final_result


def cal_iou(img_size, before, now):
    img = np.zeros([img_size[0], img_size[1]])
    img[before[1]:before[3], before[0]:before[2]] = 1
    img[now[1]:now[3], now[0]:now[2]] += 1
    count = 0
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            if img[i, j] == 2:
                count += 1
    print("count; %d"%count)
    return count/((before[3]-before[1])*(before[2]-before[0])+(now[3]-now[1])*(now[2]-now[0])-count)


def copy_file_list(file_folder, file_list, folder):
    for file in file_list:
        file_name = get_parent_folder_name(file, 1)
        copyfile(join(file_folder, file), join(folder, file_name))


def generate_metadata_file(nrof_classes, meta_data_path):
    center_list = []
    thefile = open(meta_data_path, 'w')
    for idx in range(nrof_classes):
        center_list.append(idx)
        thefile.write("%s\n" % idx)
    thefile.close()