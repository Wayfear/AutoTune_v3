import schedule
import time
import datetime
import sys
import math
from os.path import join
import os
import yaml
from utils.functions import live_stream_func, download_video_func
import subprocess

project_dir = os.getcwd()
with open(join(project_dir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)

def start_shoot():
	# connect to camera
	print('Try to connect to camera..., current time is {}'.format(datetime.datetime.now()))
	bashCommand = "nmcli d wifi connect goprofont password 12345678 iface wlan0"
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()
	time.sleep(40)

	print('Start shooting now...')
	shoot_seconds = math.ceil(cfg['auto_stream']['shoot_hours'] * 3600)
	live_stream_func(shoot_seconds)
	time.sleep(30)

	print('Downloading videos now...')
	download_video_func()

	print('Connect back to wifi ...')
	bashCommand = "nmcli d wifi connect mobicare_vicon password 55555555 iface wlan0"
	process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
	output, error = process.communicate()


start_time = cfg['auto_stream']['start_time']
schedule.every().day.at(start_time).do(start_shoot)
# schedule.every(1).minutes.do(start_shoot)

while True:
    schedule.run_pending()
    time.sleep(1)