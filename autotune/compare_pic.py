import tensorflow as tf
import cv2
import numpy as np
from os.path import join
import yaml
from source.tools import get_format_file, to_rgb
import re
import os
from itertools import combinations
import source.libs.facenet as facenet

project_dir = os.getcwd()
with open(join(project_dir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)

data_path = join(project_dir, cfg['capture_pic']['folder'])

with tf.Graph().as_default():
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(cfg['base_conf']['gpu_num'])
    with tf.Session(config=config) as sess:
        facenet.load_model(join(project_dir, cfg['base_conf']['model_path'],
                                cfg['feature_extraction_classifier']['facenet_model_path']))
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

image_list = []
path_list = cfg['compare_pic']['pic_name']
full_path_list = list(map(lambda x: join(data_path, x), path_list))
result = np.empty([0, 128])
for path in full_path_list:
    im = cv2.imread(path)
    pre_whitened = facenet.prewhiten(im)
    image_list.append(pre_whitened)
images = np.stack(image_list)
feed_dict = {images_placeholder: images, phase_train_placeholder: False}
emb = sess.run(embeddings, feed_dict=feed_dict)
result = np.vstack((result, emb))

res = []
for path, source in zip(path_list, result):
    tmp = {}
    tmp[path] = source
    res.append(tmp)
re = list(combinations(res, 2))

for r in re:
    for a in r[0]:
        n1 = a
        v1 = r[0][a]
    for a in r[1]:
        n2 = a
        v2 = r[1][a]
    print('%s and %s: %f' % (n1, n2, np.linalg.norm(v1 - v2)))