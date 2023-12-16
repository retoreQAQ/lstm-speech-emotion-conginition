import os
import random
from collections import defaultdict

# 设置数据目录和输出目录
data_folder = '../database/CREMA-D/AudioWAV/'  # 替换为您语音文件的实际路径
output_folders = ['./feat-for-all', './feat-for-type']  # 替换为您想要保存 wav.scp 文件的路径
wav_path = '/home/you/workspace/son/database/CREMA-D/AudioWAV/'
pwd = os.getcwd()

# 获取所有文件名
filenames = os.listdir(data_folder)
filenames = [f for f in filenames if f.endswith('.wav')]
filenames_withou_xx = [f for f in filenames if 'XX' not in f]

def divide(file_list):
    emotion_files = defaultdict(list)
    for filename in file_list:
        emotion_label = filename.split('.')[0].split('_')[3]
        emotion_files[emotion_label].append(filename)

    # 划分数据集
    train_files = []
    valid_files = []
    test_files = []

    for emotion_label, files in emotion_files.items():
        random.shuffle(files)
        valid_size = len(files) // 10
        train_files.extend(files[:-2*valid_size])
        valid_files.extend(files[-2*valid_size:-valid_size])
        test_files.extend(files[-valid_size:])
    
    random.shuffle(train_files)
    random.shuffle(valid_files)
    random.shuffle(test_files)
    
    return train_files, valid_files, test_files

# 生成 wav.scp 文件
def create_scp_file(file_list, dataset_type, output_folder):
    with open(os.path.join(output_folder, dataset_type, 'wav.scp'), 'w') as scp_file:
        for filename in file_list:
            prefix = filename.split('.')[0]  # 获取前缀，移除.wav
            path = f'{wav_path}{filename}'  # 拼接完整路径
            scp_file.write(f'{prefix} {path}\n')

train_files, valid_files, test_files = divide(filenames)
create_scp_file(train_files, 'train', output_folders[1])
create_scp_file(valid_files, 'val', output_folders[1])
create_scp_file(test_files, 'test', output_folders[1])
train_files, valid_files, test_files = divide(filenames_withou_xx)
create_scp_file(train_files, 'train', output_folders[0])
create_scp_file(valid_files, 'val', output_folders[0])
create_scp_file(test_files, 'test', output_folders[0])

print("Wav.scp files have been created.")
