import tools
import os
import yaml
from os.path import join
import json
import tensorflow as tf
import cv2
import numpy as np
import libs.align.detect_face as detect_face
from libs import facenet
from scipy import misc
from shutil import copyfile, rmtree
import re
from datetime import datetime

project_dir = os.path.dirname(os.getcwd())
with open(join(project_dir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)

middle_data_path = join(project_dir, cfg['base_conf']['middle_data_path'])
multiple_data_path = join(project_dir, cfg['base_conf']['multiple_meeting_data_path'])

start_time = cfg['mix_meeting_data_by_time']['start_time']
start_time = datetime.strptime(start_time, '%m-%d-%H-%M')
end_time = cfg['mix_meeting_data_by_time']['end_time']
end_time = datetime.strptime(end_time, '%m-%d-%H-%M')
meetings = os.listdir(middle_data_path)
meetings = list(filter(lambda x: re.match(r'.+?_.+', x) and x != 'time' and x != 'tmp' and x != 'tmp_video', meetings))
time_folders = []
for folder_name in meetings:
    times = folder_name.split('_')
    time_folders.append([datetime.strptime(times[0], '%m-%d-%H-%M-%S'),
                         datetime.strptime(times[1], '%m-%d-%H-%M-%S'), folder_name])
    time_folders.sort(key=lambda x: x[0])

meetings = list(filter(lambda x: x[0] >= start_time and x[1] <= end_time, time_folders))
meetings = list(map(lambda x: x[2], meetings))

folder_name = '%s_%s' % (start_time.strftime("%m-%d-%H-%M-%S"),
                                         end_time.strftime("%m-%d-%H-%M-%S"))
folder_name = join(multiple_data_path, folder_name)

thres = cfg['pre_process']['wifi_threshold']
all_meeting_people_name = tools.get_meeting_people_name(middle_data_path, thres)
if os.path.exists(folder_name):
    rmtree(folder_name)
os.mkdir(folder_name)
os.mkdir(join(folder_name, 'pic'))
pic_paths = []
for m in meetings:
    pics = tools.get_format_file(join(middle_data_path, str(m)), 3, r'.+\.jpeg')
    pic_paths.extend(pics)
index = 0
for pic in pic_paths:
    temp = pic.split('/')
    name = temp[-1]
    cluster = temp[-2]
    meeting = temp[-4]
    if cluster != '-1':
        copyfile(pic, join(folder_name, 'pic', '%d_%s_%s_%s' % (index, meeting, cluster, name)))
        print(join(folder_name, 'pic', '%d_%s_%s_%s' % (index, meeting, cluster, name)))
        index += 1


meeting_names = {}
for m in meetings:
    meeting_names[str(m)] = all_meeting_people_name[str(m)]

with open(join(folder_name, 'wifi_info.json'), 'w') as f:
    json.dump(meeting_names, f)

with tf.Graph().as_default():
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(cfg['base_conf']['gpu_num'])
    with tf.Session(config=config) as sess:
        facenet.load_model(join(project_dir, cfg['base_conf']['model_path'],
                                cfg['feature_extraction_classifier']['facenet_model_path']))
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

pic_paths = os.listdir(join(folder_name, 'pic'))
pic_paths.sort(key=lambda x: int(re.split('_|\.|', x)[0]))
image_list = []
result = np.empty([0, 128])
piece_num = cfg['feature_extraction_classifier']['piece_num']
for pic in pic_paths:
    im = cv2.imread(join(folder_name, 'pic', pic))
    print(join(folder_name, 'pic', pic))
    prewhitened = facenet.prewhiten(im)
    image_list.append(prewhitened)
    if len(image_list) == piece_num:
        images = np.stack(image_list)
        feed_dict = {images_placeholder: images, phase_train_placeholder: False}
        emb = sess.run(embeddings, feed_dict=feed_dict)
        result = np.vstack((result, emb))
        print(emb.shape)
        image_list.clear()

images = np.stack(image_list)
feed_dict = {images_placeholder: images, phase_train_placeholder: False}
emb = sess.run(embeddings, feed_dict=feed_dict)
result = np.vstack((result, emb))
print(result.shape)
np.save(join(folder_name, 'mix.npy'), result)