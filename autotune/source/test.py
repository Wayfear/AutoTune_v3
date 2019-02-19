import tools
import yaml
import os
from os.path import join
import toolz


project_dir = os.path.dirname(os.getcwd())
with open(join(project_dir, 'config.yaml'), 'r') as f:
    cfg = yaml.load(f)

multiple_data_path = join(project_dir, cfg['base_conf']['multiple_meeting_data_path'])

for folder in cfg['match']['folders']:
    folder_path = join(multiple_data_path, folder)
    result_paths = tools.get_format_file(folder_path, 1, r'.+_context_result\.txt')
    for path in result_paths:
        print(path)
        result = tools.simple_paser_result_file(path)
        print(result)