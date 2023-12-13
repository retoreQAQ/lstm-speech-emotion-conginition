import os
import sys
import random
import time
import torch
import torch.nn as nn
import kaldiio
import json
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class MyDataset(Dataset):
    def __init__(self, ark_path, if_norm, num_time_steps):
        super(Dataset, self).__init__()
        # 在外面做好路径对应标签的文件，在这里用load_txt处理得到data数据集，定义好len和getitem，就ok了，可以传给dataloader处理
        self.ark_path = ark_path
        self.if_norm = if_norm
        self.num_time_steps = num_time_steps
        self.feat_label_length_list = self.get_feat(self.ark_path)
            
    def __len__(self):
        # 数据集数量。决定迭代多少次能够遍历一遍数据集，也就是一个epoch大小
        return len(self.feat_label_length_list)
    
    def __getitem__(self, idx):
        # idx：[0, length(self.feat_label_length_list)]
        # feature:tensor label:float length:int
        sample = self.feat_label_length_list[idx]
        return sample
    
    def get_feat(self, ark_path):
        feat_label_length_list = []
       # 读取ark特征文件，返回一个字典，键为utt-id，值为特征矩阵（NumPy数组）
        feats_dict = kaldiio.load_ark(ark_path)
        # 将字典中的特征数据转换为label，feature列表
        # feats_list = [[torch.tensor(feat), name.split('-')[1], name.split('-')[2], name.split('-')[3]] for name, feat in feats_dict]
        # if self.mel is not None:
        #     # 参数：采样率：跟kaldi提取spec时的设置保持一致；fft与提取时frame-length设置有关；mel滤波器数量可以调整
        #     mel_filter = librosa.filters.mel(sr=self.mel["sr"], n_fft=self.mel["n_fft"], n_mels=self.mel["n_mels"])
        #     feats_list = [[torch.tensor(np.dot(feat, mel_filter.T)), uttid] for uttid, feat in feats_dict]
        feats_list = [[torch.tensor(feat), uttid] for uttid, feat in feats_dict]
        # [
        #     [tensor(n_frame, 39), uttid(str)],
        #     ...
        # ]
        for feat_label in feats_list:
            length = feat_label[0].shape[0]
            # 归一化，kaldi提取的话不用开
            if self.if_norm:
                feat_mean = torch.mean(feat_label[0], dim=0, keepdim=True)
                feat_std = torch.std(feat_label[0], dim=0, keepdim=True)
                feat_label[0] = (feat_label[0] - feat_mean) / feat_std
            # pre填充
            # feat_label[0] = torch.nn.functional.pad(feat_label[0], (0, 0, num_time_steps - length, 0))
            # post填充
            feat_label[0] = torch.nn.functional.pad(feat_label[0], (0, 0, 0, self.num_time_steps - length))
            feat_label.append(length)
            feat_label_length_list.append(feat_label)
        return feat_label_length_list
    
class MyLSTM(nn.Module):
    def __init__(self, n_feature, hidden_size, lstm_num_layers, lstm_dropout, l2_lambda, if_sigmoid, if_bi):
        super(MyLSTM, self).__init__()
        self.bn = nn.BatchNorm1d(n_feature).cuda()
        self.lstm = nn.LSTM(input_size=n_feature, hidden_size=hidden_size, num_layers=lstm_num_layers, batch_first=True, dropout=lstm_dropout, bidirectional=if_bi).cuda()
        self.linear_type = nn.Linear(in_features=hidden_size, out_features=6).cuda()
        self.linear_level = nn.Linear(in_features=hidden_size, out_features=4).cuda()
        self.linear_type_bi = nn.Linear(in_features=hidden_size*2, out_features=6).cuda()
        self.linear_level_bi = nn.Linear(in_features=hidden_size*2, out_features=4).cuda()
        self.l2_lambda = l2_lambda
        self.if_sigmoid = if_sigmoid
        self.if_bi = if_bi
    
    def forward(self, inputs, lengths):
        inputs = inputs.to(torch.float).cuda()
        inputs = inputs.permute(0, 2, 1)
        inputs = self.bn(inputs)
        inputs = inputs.permute(0, 2, 1)
        feats = nn.utils.rnn.pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)
        # the shape of h_last: (num_layers * num_directions, batch_size, hidden_size)
        outputs, (h_last, c_last) = self.lstm(feats)
        # when single direction, h_last[-1] is output of the last layer.
        # but when bidirection is true, maybe need to use h_last[-1] and h_last[-2] which from different directions.
        # combine them and change the layer with shape of input is (hidden_size * 2)
        h_last_combined = torch.cat((h_last[-1], h_last[-2]), dim=1)
        if self.if_bi:
            emo_type = self.linear_type_bi(h_last_combined)
            emo_level = self.linear_level_bi(h_last_combined)
        else:
            emo_type = self.linear_type(h_last[-1])
            emo_level = self.linear_level(h_last[-1])
        emo_type = torch.mean(output, dim=-1)
        
        if self.if_sigmoid:
            output = torch.sigmoid(output) * 5
        return output

    def l2_regularization(self):
        l2_reg = torch.tensor(0.0).cuda()
        for name, param in self.named_parameters():
            if 'weight' in name:
                l2_reg += torch.norm(param, p=2)
        return self.l2_lambda * l2_reg

class Logger(object):
    def __init__(self, filename='log.txt', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # 确保刷新输出到文件
        self.terminal.flush()
        self.log.flush()

def model_eval(scores, labels):
    # 协方差,用来衡量两个变量之间的整体误差
    values_matrix = torch.stack([scores, labels])
    # # 计算两个张量之间的协方差矩阵。该矩阵为对称矩阵，形状为(x, x),x是上方拼接了几个张量，也就是几行。
    # cov_matrix = torch.cov(values_matrix)
    # # cov_matrix[i ,j]代表第i个张量和第j个张量的协方差，等同于cov_matrix[j, i]。
    # cov = cov_matrix[0, 1]
    # 皮尔森系数，衡量两个变量之间线性相关程度
    corr_matrix = torch.corrcoef(values_matrix)
    pearson = corr_matrix[0, 1]
    mse = torch.nn.functional.mse_loss(scores, labels)
    mae = torch.mean(torch.abs(scores - labels))
    print(f"mse: {mse:.2f}\nmae: {mae:.2f}\npearson: {pearson:.2f}")
    return mse, mae, pearson

def run(model, data_loader, optimizer, loss_fun, label_name, train=True):
    if train:
        model.train()
    else:
        model.eval()
    label_class = {'mis': 1, 'smooth': 2, 'total': 3}
    loss_all = 0.0
    scores_all = torch.tensor([]).cuda()
    labels_all = torch.tensor([]).cuda()
    for data in data_loader:
        # feats shape为(batch_size, num_time_steps, n_feature),labels shape为(batch_size)
        feats, uttids, lengths = data
        # 这里对ark文件的uttid处理，格式为 uttname-label1-label2-label3, 得到目标标签
        labels = [float(uttid.split('-')[label_class[label_name]]) for uttid in uttids]
        if train:
            # 将上一步得到的梯度清零
            optimizer.zero_grad()
        # 跑模型
        scores = model(feats.cuda(), lengths)
        # scores shape为(batch_size, 1),但是labels shape为(batch_size)，squeeze去维
        # 如果最后一批刚好剩下一个，也就是形状为（1），就还原
        scores = torch.squeeze(scores)
        if scores.shape == ():
            scores = scores.reshape(1)
        labels = torch.tensor(labels, dtype=torch.float32).cuda()
        # 求损失
        loss = loss_fun(scores, labels) + model.l2_regularization()
        if train:
            # 梯度下降
            loss.backward()
            optimizer.step()
        loss_all += loss
        scores_all = torch.cat((scores_all, scores), dim=0)
        labels_all = torch.cat((labels_all, labels), dim=0)
    return loss_all, scores_all, labels_all

def line_chart(save_path, data, data_label, x_label, y_label, title, size=(10 ,5)):
    plt.figure(figsize=size)
    plt.plot(data, label=data_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    # 添加图例
    plt.legend()
    # 网格线
    plt.grid(True)
    plt.savefig(f'{save_path}/{title}.png')
    
def scatter_points(save_path, labels, predictions, mse, pearson, name, scatter_random, size=(10 ,10)):
    if scatter_random:
        for i in range(len(labels)):
            labels[i] = labels[i] + 0.2 * random.uniform(-1, 1)
        for i in range(len(predictions)):
            predictions[i] = predictions[i] + 0.2 * random.uniform(-1, 1)
    labels = labels.detach().cpu().numpy()
    predictions = predictions.detach().cpu().numpy()
    plt.figure(figsize=size)
    # 画散点图
    plt.scatter(labels, predictions, s=10)
    # 画一条直线（例如，y=x）
    # plt.plot([min(labels), max(labels)], [min(labels), max(labels)])
    plt.plot([0, 5], [0, 5])
    # 添加标签和标题
    plt.title(f'pcc = {pearson}\nmse = {mse}')
    plt.xlabel('Label value')
    plt.ylabel('Predicted value')
    plt.savefig(f'{save_path}/scatter_{name}.png')
    plt.clf()
    
def adjust_lr(optimizer, step_num, pearson):
    # if step_num < n_warm_up:
    #     learning_rate = lr * (step_num / n_warm_up)
    # else:
    #     learning_rate = lr * ((n_epoch - step_num) / (n_epoch - n_warm_up))
    learning_rate = 0.00005
    if step_num >= 50:
        learning_rate = 0.00001
    if step_num >= 100:
        learning_rate = 0.000003
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    

def main(args, parameter_dict):
    torch.cuda.empty_cache()
    
    # 默认为mfcc设置
    # 时间步长，序列长度，算数据集得到
    num_time_steps = parameter_dict["num_time_steps"]
    # 特征数量
    n_feature = parameter_dict["mfcc"]["n_feature"]
    # 隐藏层大小
    hidden_size = parameter_dict["mfcc"]["hidden_size"]
    # 批大小
    batch_size_train = parameter_dict["mfcc"]["batch_size_train"]
    batch_size_val = parameter_dict["mfcc"]["batch_size_val"]
    # 跑几遍数据集
    n_epoch = parameter_dict["n_epoch"]
    n_warm_up = int(n_epoch / 10)
    # 学习率
    lr = parameter_dict["lr"]
    # lstm层数
    lstm_num_layers = parameter_dict["lstm_num_layers"]
    # L2范数系数
    l2_lambda = parameter_dict["l2_lambda"]
    # 是否使用sigmoid
    if_sigmoid = parameter_dict["if_sigmoid"]
    lstm_dropout = parameter_dict["lstm_dropout"]
    loss_name = parameter_dict["loss_name"]
    optimizer_name = parameter_dict["optimizer_name"]
    # mfcc提取时已归一化
    if_norm = parameter_dict["mfcc"]["if_norm"]
    # 特征类型
    feature = parameter_dict["feature"]
    # 是否使散点云雾化
    scatter_random = parameter_dict["scatter_random"]
    if_early_stopping = parameter_dict["if_early_stopping"]
    es_threshold = parameter_dict["es_threshold"]
    # 分数类型
    label_name = 'total'
    if args:
        if args[0] == '0':
            label_name = 'mis'
        elif args[0] == '1':
            label_name = 'smooth'
        elif args[0] == '2':
            label_name = 'total'
    
    gpu_name = torch.cuda.get_device_name(0)
    if gpu_name == "NVIDIA GeForce MX350":
        device = 'son'
        num_workers = 0
        os.chdir("D:/workspace/yd/")
    else:
        device = 'you'
        num_workers = 16
        os.chdir("/home/you/workspace/yd_dataset")
    
    ark_path_train = './feat/mfcc_train_39.ark'
    ark_path_val = './feat/mfcc_val_39.ark'
    
    if feature == 'mfcc_13':
        n_feature = parameter_dict["mfcc_13"]["n_feature"]
        hidden_size = parameter_dict["mfcc_13"]["hidden_size"]
        batch_size_train = parameter_dict["mfcc_13"]["batch_size_train"]
        batch_size_val = parameter_dict["mfcc_13"]["batch_size_val"]
        if_norm = parameter_dict["mfcc_13"]["if_norm"]
        ark_path_train = './feat/mfcc_train_13_norm.ark'
        ark_path_val = './feat/mfcc_val_13_norm.ark'
    elif feature == 'spec':
        n_feature = parameter_dict["spec"]["n_feature"]
        hidden_size = parameter_dict["spec"]["hidden_size"]
        batch_size_train = parameter_dict["spec"]["batch_size_train"]
        batch_size_val = parameter_dict["spec"]["batch_size_val"]
        if_norm = parameter_dict["spec"]["if_norm"]
        ark_path_train = './feat/spec_train.ark'
        ark_path_val = './feat/spec_val.ark'
    elif feature == 'fbank':
        n_feature = parameter_dict["fbank"]["n_feature"]
        hidden_size = parameter_dict["fbank"]["hidden_size"]
        batch_size_train = parameter_dict["fbank"]["batch_size_train"]
        batch_size_val = parameter_dict["fbank"]["batch_size_val"]
        if_norm = parameter_dict["fbank"]["if_norm"]
        ark_path_train = './feat/fbank_train.ark'
        ark_path_val = './feat/fbank_val.ark'
    elif feature == 'plp':
        n_feature = parameter_dict["plp"]["n_feature"]
        hidden_size = parameter_dict["plp"]["hidden_size"]
        batch_size_train = parameter_dict["plp"]["batch_size_train"]
        batch_size_val = parameter_dict["plp"]["batch_size_val"]
        if_norm = parameter_dict["plp"]["if_norm"]
        ark_path_train = './feat/plp_train_39.ark'
        ark_path_val = './feat/plp_val_39.ark'
    
    
    # 生成数据集
    dataset_train = MyDataset(ark_path_train, if_norm, num_time_steps)
    dataset_val = MyDataset(ark_path_val, if_norm, num_time_steps)
        
    # 生成data_loader
    # 这里生成的是一个迭代器,返回值batch在第一个，需要设置lstm类参数batch_first
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size_train, shuffle=True, num_workers=num_workers, drop_last=False)
    data_loader_val = DataLoader(dataset_val, batch_size=batch_size_val, shuffle=False, num_workers=num_workers, drop_last=False)
        
    # 实例化模型
    mylstm = MyLSTM(n_feature, hidden_size, lstm_num_layers, lstm_dropout, l2_lambda, if_sigmoid, if_bi=True).cuda()
        
    # 定义损失函数
    # 对批次损失值默认求平均，可以改成sum求和
    loss_fun = nn.MSELoss(reduction='mean')
    if loss_name == 'cross':
        loss_fun = nn.CrossEntropyLoss().cuda()
        
    # 定义优化器
    optimizer = torch.optim.Adam(mylstm.parameters(), lr=lr)
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(mylstm.parameters(), lr=lr)

    # 开始训练
    loss_train_list = []
    loss_val_list = []
    pearson = 0
    
    for epoch in range(n_epoch):
        # 动态调整学习率
        adjust_lr(optimizer, epoch, pearson)
        # 训练
        print(f'epoch: {epoch+1}\nThe performance in the dataset_train:')
        loss_train, scores_train_all, labels_train_all = run(mylstm, data_loader_train, optimizer, loss_fun, label_name)
        loss_train_list.append(loss_train.item())
        print(f'The loss of train: {loss_train:.2f}')
        model_eval(scores_train_all, labels_train_all)
        # 验证
        print('The performance in the dataset_val:')
        loss_val, scores_val_all, labels_val_all = run(mylstm, data_loader_val, optimizer, loss_fun, label_name, False)
        loss_val_list.append(loss_val.item())
        print(f"The loss of val: {loss_val:.2f}")
        mse, mae, pearson = model_eval(scores_val_all, labels_val_all)
        if if_early_stopping:
            if pearson>=es_threshold:
                print('early stopping')
                break
        print('\n')
    
    # 保存模型
    torch.save(mylstm.state_dict(), f'{log_folder}/model.pth')
    
    line_chart(log_folder, loss_train_list, 'loss', 'epoch', 'loss', 'loss_train')
    line_chart(log_folder, loss_val_list, 'loss', 'epoch', 'loss', 'loss_val')
    
    labels_val_all_int = torch.round(labels_val_all)
    scores_val_all_int = torch.round(scores_val_all)
    
    scatter_points(log_folder, labels_val_all, scores_val_all, mse, pearson, scatter_random, 'float')
    mse, mae, pearson_p = model_eval(scores_val_all_int, labels_val_all)
    scatter_points(log_folder, labels_val_all, scores_val_all_int, mse, pearson_p, scatter_random, 'predictions_int')
    mse, mae, pearson_l = model_eval(scores_val_all, labels_val_all_int)
    scatter_points(log_folder, labels_val_all_int, scores_val_all, mse, pearson_l, scatter_random, 'labels_int')
    mse, mae, pearson_b = model_eval(scores_val_all_int, labels_val_all_int)
    scatter_points(log_folder, labels_val_all_int, scores_val_all_int, mse, pearson_b, scatter_random, 'both_int')
    
    print('\n\n\nThe hyperparameter list:\n')
    print('epoch: ' + str(n_epoch))
    print('hidden_size: ' + str(hidden_size))
    print('batch_size: ' + str(batch_size_train))
    print('batch_size_val: ' + str(batch_size_val))
    print('learning rate: ' + str(lr))
    print('l2_lambda: ' + str(l2_lambda))
    print('loss_function: ' + str(loss_name))
    print('optimizer: ' + optimizer_name)
    print('lstm_dropout: ' + str(lstm_dropout))
    print('if_norm: ' + str(if_norm))
    print('lstm_num_layers: ' + str(lstm_num_layers))
    print('if_sigmoid: ' + str(if_sigmoid))
    print('feature_type: ' + str(feature))
    
    return pearson
        
    
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("No GPU found")
        exit(1)
    
    # 加载超参数文件
    with open('parameter.json', 'r') as f:
        parameter_group = json.load(f)
        
    if not os.path.exists('log'):
        # 创建 log 文件夹
        os.makedirs('log')
    pwd = os.getcwd()
    os.chdir('./log')
    
    if type(parameter_group) is list:
        for parameter_dict in parameter_group:
            
            # 获取当前时间
            current_time = time.strftime('%Y-%m-%d_%H_%M', time.localtime())
            # 创建文件夹
            log_folder_name = f'log_{current_time}'
            os.makedirs(log_folder_name, exist_ok=True)
            # 切换到文件夹内
            log_folder = f'{pwd}/log/{log_folder_name}'
            os.chdir(log_folder)
            # 保存原始的sys.stdout
            original_stdout = sys.stdout
            # 打开一个文件来将print的内容写入
            sys.stdout = Logger(stream=original_stdout)
            
            pcc = main(sys.argv[1:], parameter_dict)
            
            # 恢复原始的sys.stdout
            sys.stdout = original_stdout
            os.chdir(f'{pwd}/log')
            new_logger_folder = f'{log_folder}_{pcc:.3f}'
            os.rename(log_folder, new_logger_folder)
    else:
        print(f'parameter_group.json error:\n{parameter_group}')
        exit(1)