from goprocam import GoProCamera
import os
from os.path import join
import json

project_dir = os.getcwd()
video_path = join(project_dir, 'video')

gpCam = GoProCamera.GoPro()
data = json.loads(gpCam.listMedia())

file_list = {}

for folder in data['media']:
    dic = folder['d']
    for file in folder['fs']:
        file_list[file['n']] = file['mod']

with open('video_information.json', 'w') as f:
    json.dump(file_list, f)
