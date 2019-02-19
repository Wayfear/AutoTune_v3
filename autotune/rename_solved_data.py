import source.tools as tools
import os
from os.path import join
import yaml
from datetime import datetime, timedelta
project_dir = os.getcwd()
with open(join(project_dir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)

origin_data_path = join(project_dir, 'video')
paths = tools.get_format_file(origin_data_path, 1, r'.+04-02.+font\.MP4')

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
