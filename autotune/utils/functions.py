from goprocam import GoProCamera
from goprocam import constants
import cv2
from time import time
import socket
import numpy as np
import os
from os.path import join
import yaml
import json
from datetime import datetime, timedelta
import subprocess
import re


def live_stream_func(shoot_duration):
    project_dir = os.getcwd()
    with open(join(project_dir, 'config.yaml'), 'r') as f:
        cfg = yaml.load(f)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    gpCam = GoProCamera.GoPro()
    gpCam.power_on()
    gpCam.syncTime()
    gpCam.livestream("start")
    cap = cv2.VideoCapture("udp://10.5.5.9:8554")
    jump_frame = 0
    t=time()
    start = time()

    while True:
        start_capture = True
        count = 0
        pre_count = 0
        num_flag = 0
        while True:
            num_flag += 1
            if num_flag%2 == 0:
                continue

            if start_capture:
                nmat, prev_frame = cap.read()
                start_capture = False
                continue

            nmat, frame = cap.read()
            cv2.imshow('image',frame)
            if jump_frame != 0:
                jump_frame -= 1
                continue

            pre_count = count
            frame_diff = cv2.absdiff(frame, prev_frame)
            gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
            thrs = 32
            ret, motion_mask = cv2.threshold(gray_diff, thrs, 1, cv2.THRESH_BINARY)
            flag = np.sum(motion_mask)
            prev_frame = frame.copy()
            print(flag)

            if flag > 60:
                count += 1
            if count == pre_count:
                count = 0

            if count > 3:
                print(str(time()) + ' take_video!')
                gpCam.shoot_video(cfg['live_stream']['video_time'])
                jump_frame = 45

                break

            # cv2.imshow("GoPro OpenCV", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if time() - t >= 2.5:
                sock.sendto("_GPHD_:0:0:2:0.000000\n".encode(), ("10.5.5.9", 8554))
                t=time()

            # scheduled to exit
            if time() > start + shoot_duration:
                print("Shooting done, prepare downloading videos!")
                gpCam.livestream("stop")
                cv2.destroyAllWindows()
                return

        cv2.destroyAllWindows()


def download_video_func():
    project_dir = os.getcwd()
    video_path = join(project_dir, 'video')

    gpCam = GoProCamera.GoPro()
    data = json.loads(gpCam.listMedia())

    file_list = []

    for folder in data['media']:
        dic = folder['d']
        for file in folder['fs']:
            file_list.append({'dictionary': dic, 'name': file['n'], 'time': file['mod']})

    # get the wireless SSID
    tmp = subprocess.check_output('iwconfig', stderr=subprocess.STDOUT)
    tmp_ls = tmp.decode("ascii").replace("\r", "").split("\n")
    wifi_info = tmp_ls[4]
    start_symbol = '"'
    end_symbol = '" '
    wifi = re.search('%s(.*)%s' % (start_symbol, end_symbol), wifi_info).group(1)
    print('connected wifi is {}'.format(wifi))

    # format time
    for file in reversed(file_list):
        gpCam.downloadMedia(file['dictionary'], file['name'])
        name = datetime.fromtimestamp(int(file['time']))
        name = name - timedelta(hours=1)
        os.rename(file['name'], join(video_path, '%s_%s.MP4' %(name.strftime('%m-%d-%H-%M-%S'), wifi)))
        gpCam.deleteFile(file['dictionary'], file['name'])

    print('All videos downloaded!')
    return