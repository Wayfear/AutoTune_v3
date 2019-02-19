import tensorflow as tf
import cv2
# import detect_face
import numpy as np
import source.libs.align.detect_face as detect_face
from scipy import misc
import os
from os.path import join
import yaml
from source.tools import get_format_file, to_rgb
import re


project_dir = os.getcwd()
with open(join(project_dir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)

# data_path = join(project_dir, cfg['capture_pic']['folder'])
data_path = cfg['capture_pic']['abs_folder']
names = get_format_file(data_path, 1, r'.+\.jpg$')

image_size = cfg['capture_pic']['image_size']

print('Creating networks and loading parameters')

with tf.Graph().as_default():
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(cfg['base_conf']['gpu_num'])
    sess = tf.Session(config=config)
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, join(project_dir, cfg['base_conf']['model_path'],
                                                    cfg['pre_process']['mtcnn_model_path']))
path_list = []
index = 0
for pic_dir in names:
    frame = cv2.imread(pic_dir)
    num_rows, num_cols = frame.shape[:2]
    img_size = np.asarray(frame.shape)[0:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if gray.ndim == 2:
        img = to_rgb(gray)
        bounding_boxes, _ = detect_face.detect_face(img, cfg['capture_pic']['mtcnn_mini_size'],
                                pnet, rnet, onet, cfg['capture_pic']['mtcnn_threshold'],
                                cfg['capture_pic']['mtcnn_factor'])
        print(bounding_boxes)

        margin = 44

        split_name = re.split(r'\/|\.', pic_dir)

        for j in range(len(bounding_boxes)):
            det = np.squeeze(bounding_boxes[j, 0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
            cropped = frame[bb[1]:bb[3], bb[0]:bb[2], :]
            cropped_gray = img[bb[1]:bb[3], bb[0]:bb[2], :]
            aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            cv2.imwrite(join(data_path, "%s_%d.jpeg" % (split_name[-2], j)), aligned)
            aligned_gray = misc.imresize(cropped_gray, (image_size, image_size), interp='bilinear')
            cv2.imwrite(join(data_path, "%s_%d_gray.jpeg" % (split_name[-2], j)), aligned_gray)
    print('finish %s'%dir)