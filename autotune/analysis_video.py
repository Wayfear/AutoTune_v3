import source.tools as tools
import os
import yaml
from os.path import join
import tensorflow as tf
import cv2
import numpy as np
import source.libs.align.detect_face as detect_face
from scipy import misc
from shutil import copyfile, rmtree

project_dir = os.getcwd()
with open(join(project_dir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)

video_source_path = join(project_dir, 'video')

with tf.Graph().as_default():
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(cfg['base_conf']['gpu_num'])
    sess = tf.Session(config=config)
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess,
            join(project_dir, cfg['base_conf']['model_path'], cfg['pre_process']['mtcnn_model_path']))
face_cascade = cv2.CascadeClassifier(join(project_dir, cfg['base_conf']['model_path'],
         cfg['pre_process']['opencv_model_path'], 'haarcascade_frontalface_default.xml'))


for video_path in cfg['analysis_video']['video_name']:
    temp_video = video_path.split('.')[0]
    if os.path.exists(join(video_source_path, temp_video)):
        rmtree(join(video_source_path, temp_video))
    os.mkdir(join(video_source_path, temp_video))
    cap = cv2.VideoCapture(join(video_source_path, video_path))
    frame_num = 0
    index = 0
    false_num = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            false_num += 1
        if false_num > 120:
            break
        frame_num += 1
        if ret:
            num_rows, num_cols = frame.shape[:2]
            img_size = np.asarray(frame.shape)[0:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if gray.ndim == 2:
                img = tools.to_rgb(gray)

                bounding_boxes, _ = detect_face.detect_face(img, cfg['pre_process']['mtcnn_mini_size'],
                                                            pnet, rnet, onet, cfg['pre_process']['mtcnn_threshold'],
                                                            cfg['pre_process']['mtcnn_factor'])
                print(bounding_boxes)
                img_list = []
                if len(bounding_boxes) >= 1:
                    for b in bounding_boxes:
                        cv2.rectangle(gray, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 2)
                    margin = 44
                    for j in range(len(bounding_boxes)):
                        det = np.squeeze(bounding_boxes[j, 0:4])
                        bb = np.zeros(4, dtype=np.int32)
                        bb[0] = np.maximum(det[0] - margin / 2, 0)
                        bb[1] = np.maximum(det[1] - margin / 2, 0)
                        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                        pics = face_cascade.detectMultiScale(cropped, 1.1, 5)
                        if len(pics) != 1:
                            continue
                        aligned = misc.imresize(cropped, (cfg['pre_process']['image_size'],
                                                          cfg['pre_process']['image_size']), interp='bilinear')

                        cv2.imshow('frame', gray)
                        cv2.imwrite(join(video_source_path, temp_video,
                                         '%d-%d.jpeg' % (index, frame_num)), aligned)
                        index += 1
    print('Finish video %s' % video_path)

cap.release()
cv2.destroyAllWindows()
