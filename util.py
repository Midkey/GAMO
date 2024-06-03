import shutil
import os


def backup_file(backup_list, backup_path):
    save_path = os.path.join(backup_path, 'scripts')
    os.makedirs(save_path, exist_ok=True)
    for file in backup_list:
        shutil.copy(file, os.path.join(save_path, os.path.basename(file)))
