from goprocam import GoProCamera
import os
from os.path import join
import json
import re
from datetime import datetime, timedelta
project_dir = os.getcwd()
video_path = join(project_dir, 'video')

# paths = os.listdir(video_path)
#
# for path in paths:
#     video_time = datetime.strptime(path.split('.')[0], '%m-%d-%H-%M-%S')
#     video_time -= timedelta(hours=1)
#     os.rename(join(video_path, path), join(video_path, '%s.MP4' % video_time.strftime('%m-%d-%H-%M-%S')))

with open('video_information.json', 'r') as f:
    file_list = json.load(f)

file_name = list(file_list.keys())
# format time
file_name.sort(key=lambda x: int(re.match(r'\w+?(\d+)\.MP4', x).group(1)))

paths = os.listdir(video_path)
paths = list(filter(lambda x: re.match(r'^GOPR\d+\.MP4$', x), paths))

paths.sort(key=lambda x: int(re.match(r'\w+?(\d+)\.MP4', x).group(1)))

i = 0
for p in paths:
    while True:
        if p == file_name[i]:
            if file_name[i] in file_list:
                file_date = datetime.fromtimestamp(int(file_list[file_name[i]]))
                file_date -= timedelta(hours=1)
                os.rename(join(video_path, p), join(video_path, '%s.MP4' % file_date.strftime('%m-%d-%H-%M-%S')))
            break
        i += 1
