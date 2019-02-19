import source.tools as tools
import os
from os.path import join
import yaml
from datetime import datetime, timedelta

project_dir = os.getcwd()
with open(join(project_dir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)

origin_data_path = join(project_dir, 'video')

origin_data_dir = '/home/chris/Documents/2018/origin_data'


# for path in paths:
#     new_path = path.split('.')
#     # new_path = '%s.MP4' % new_path[0]
#     temp = new_path[0].split('/')
#     temp_name = temp[-1].split('_')
#
#     time = datetime.strptime(temp_name[0], '%m-%d-%H-%M-%S')
#     time += timedelta(hours=7)
#     new_name = time.strftime('%m-%d-%H-%M-%S') + '_' + temp_name[-1] + '.MP4'
#     # new_path = '%s.mp4' % time.strftime('%m-%d-%H-%M-%S')
#     os.rename(path, join(origin_data_path, new_name))
#

for root, dirs, files in os.walk(origin_data_dir):
    for event_dir in dirs:
        paths = tools.get_format_file(join(root, event_dir), 1, r'.+03-31.+font\.MP4')
        for path in paths:
            new_path = path.split('.')
            # new_path = '%s.MP4' % new_path[0]
            temp = new_path[0].split('/')
            temp_name = temp[-1].split('_')

            time = datetime.strptime(temp_name[0], '%m-%d-%H-%M-%S')
            time += timedelta(hours=7)
            new_name = time.strftime('%m-%d-%H-%M-%S') + '_' + temp_name[-1] + '.MP4'
            # new_path = '%s.mp4' % time.strftime('%m-%d-%H-%M-%S')
            os.rename(path, join(origin_data_path, new_name))
            # print(join(origin_data_path, new_name))
