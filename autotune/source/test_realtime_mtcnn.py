import tools
import os
import yaml
from os.path import join
import tensorflow as tf
import cv2
import numpy as np
from libs.align import detect_face
from scipy import misc
from shutil import copyfile, rmtree
from goprocam import GoProCamera
from time import time
import socket
print("lalla")
project_dir = os.path.dirname(os.getcwd())
with open(join(project_dir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)

face_cascade = cv2.CascadeClassifier(join(project_dir, cfg['base_conf']['model_path'],
         cfg['pre_process']['opencv_model_path'], 'haarcascade_frontalface_default.xml'))
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# with tf.Graph().as_default():
#     config = tf.ConfigProto()
#     config.gpu_options.visible_device_list = str(cfg['base_conf']['gpu_num'])
#     sess = tf.Session(config=config)
#     with sess.as_default():
#         pnet, rnet, onet = detect_face.create_mtcnn(sess,
#             join(project_dir, cfg['base_conf']['model_path'], cfg['pre_process']['mtcnn_model_path']))
#         print("load model")


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
t=time()
gpCam = GoProCamera.GoPro()
gpCam.syncTime()
gpCam.livestream("start")
cap = cv2.VideoCapture("udp://10.5.5.9:8554")
# jump_frame = 0
# start_capture = True
# count = 0
# pre_count = 0
# num_flag = 0
cv2.destroyAllWindows()
i=0
while True:
    # num_flag += 1
    # if num_flag%2 == 0:
    #     continue
    # if start_capture:
    #     nmat, prev_frame = cap.read()
    #     start_capture = False
    #     continue
    # i+=1
    # if i%2==0:
    #     continue
    print("read before")
    nmat, frame = cap.read()
    print(nmat)
    print(frame)
    print("read after")
    # if frame is None:
    #     break
    # print(frame)
    # bounding_boxes, _ = hog.detectMultiScale(frame, winStride=(4, 4),
    #     padding=(8, 8), scale=1.05)
    # bounding_boxes = face_cascade.detectMultiScale(frame, 1.1, 5)
    # # bounding_boxes, _ = detect_face.detect_face(frame, cfg['pre_process']['mtcnn_mini_size'],
    # #                                             pnet, rnet, onet, cfg['pre_process']['mtcnn_threshold'],
    # #                                             cfg['pre_process']['mtcnn_factor'])
    # # # print(bounding_boxes)
    # print("bounding")
    # for b in bounding_boxes:
    #     cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[0])+int(b[2]), int(b[1])+int(b[3])), (255, 0, 0), 2)

    print("show frame")

    # cv2.imshow('image',frame)
    # ch = 0xFF & cv2.waitKey(1)
    # if ch == 27:
    #     break
# cv2.destroyAllWindows()

# for meeting, video_paths in origin_video_paths.items():
#     meeting_middle_data_path = join(middle_data_path, meeting)
#     os.mkdir(meeting_middle_data_path)
#     os.mkdir(join(meeting_middle_data_path, cfg['base_conf']['mtcnn_origin_data_path']))
#
#     thres = cfg['pre_process']['wifi_threshold']
#     for th in range(thres - 10, thres + 20, 10):
#         try:
#             _, sta = tools.paser_wifi_file(origin_wifi_paths[meeting], th, cfg['pre_process']['wifi_time_interval'])
#             max_size = 0
#             for d in sta:
#                 s = list(sta[d].keys())
#                 if len(s) == 0:
#                     continue
#                 if max(s) > max_size:
#                     max_size = max(s)
#
#             for d in sta:
#                 sta[d] = dict(filter(lambda x: x[1] > 2, sta[d].items()))
#
#             y_lable = []
#             file_name = ''
#             sta = dict(filter(lambda x: len(x[1]) > 0, sta.items()))
#             arr = np.zeros(shape=[len(sta), max_size + 1])
#             index = 0
#             for d in sta:
#                 for m in sta[d]:
#                     arr[index, int(m)] = 200
#                 index += 1
#             for li in list(sta.keys()):
#                 y_lable.append(cfg['mac_name'][li])
#                 file_name += (cfg['mac_name'][li] + '_')
#             file_name = file_name[:-1]
#             tools.plot_wifi_pic(arr, y_lable,
#                                 join(meeting_middle_data_path, '%d_%s.png' % (th, file_name)))
#             print('Finish wifi data file %s' % ('%d_%s.png' % (th, file_name)))
#         except:
#             print('Can not find wifi data file!')
#
#     index = 0
#     for video_path in video_paths:
#         cap = cv2.VideoCapture(video_path)
#         frame_num = 0
#         false_num = 0
#         while (cap.isOpened()):
#             ret, frame = cap.read()
#             if frame is None:
#                 false_num += 1
#             if false_num > 120:
#                 break
#             frame_num += 1
#             if ret:
#                 num_rows, num_cols = frame.shape[:2]
#                 img_size = np.asarray(frame.shape)[0:2]
#                 gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#                 if gray.ndim == 2:
#                     img = tools.to_rgb(gray)
#
#                     bounding_boxes, _ = detect_face.detect_face(img, cfg['pre_process']['mtcnn_mini_size'],
#                                                                 pnet, rnet, onet, cfg['pre_process']['mtcnn_threshold'],
#                                                                 cfg['pre_process']['mtcnn_factor'])
#                     print(bounding_boxes)
#                     img_list = []
#                     if len(bounding_boxes) >= 1:
#                         # for b in bounding_boxes:
#                         #     cv2.rectangle(gray, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 2)
#                         margin = 44
#                         for j in range(len(bounding_boxes)):
#                             det = np.squeeze(bounding_boxes[j, 0:4])
#                             bb = np.zeros(4, dtype=np.int32)
#                             bb[0] = np.maximum(det[0] - margin / 2, 0)
#                             bb[1] = np.maximum(det[1] - margin / 2, 0)
#                             bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
#                             bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
#                             cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
#                             pics = face_cascade.detectMultiScale(cropped, 1.1, 5)
#                             if len(pics) != 1:
#                                 continue
#                             aligned = misc.imresize(cropped, (cfg['pre_process']['image_size'],
#                                                               cfg['pre_process']['image_size']), interp='bilinear')
#                             if cfg['pre_process']['show_pic']:
#                                 cv2.imshow('frame', aligned)
#                             cv2.imwrite(join(meeting_middle_data_path, cfg['base_conf']['mtcnn_origin_data_path'],
#                                              '%d-%d.jpeg' % (index, frame_num)), aligned)
#                             index += 1
#         print('Finish video %s' % video_path)
#
# cap.release()
# cv2.destroyAllWindows()
