"""
usage: detect the mac addresses of non-mobile devices, e.g., wifi APs, desktops
Sample usage:

"""

from source.tools import parse_wifi_file_by_duration_multi_process
import yaml
import os
from os.path import join
import collections

project_dir = os.getcwd()
with open(join(project_dir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)

data_dir = join(project_dir, 'wifi_data')
data_path = join(data_dir, cfg['device_filtering']['data_name'])

# convent RSSI readings to list of timeslots
timeslot_ls = parse_wifi_file_by_duration_multi_process(30, data_path)

all_mac_ls = []
# iterate to examine
for timeslot in timeslot_ls:
    key_ls = list(timeslot.keys())
    devices = timeslot[key_ls[0]]
    tmp_all_mac_ls = list(devices.keys())
    all_mac_ls += tmp_all_mac_ls

collections.Counter(all_mac_ls)