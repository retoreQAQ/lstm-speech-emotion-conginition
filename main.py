import os
import sys
import time
import torch
import torch.nn as nn
import kaldiio
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns

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
        feats_list = [[torch.tensor(feat), uttid] for uttid, feat in feats_dict]
        # shape of feat:(n_frame/seq_length/num_time_steps, n_feature)
        for feat_label in feats_list:
            length = feat_label[0].shape[0]
            # 归一化，kaldi提取的话不用开
            if self.if_norm:
                feat_mean = torch.mean(feat_label[0], dim=0, keepdim=True)
                feat_std = torch.std(feat_label[0], dim=0, keepdim=True)
                feat_label[0] = (feat_label[0] - feat_mean) / feat_std
            # pre padding
            # feat_label[0] = torch.nn.functional.pad(feat_label[0], (0, 0, num_time_steps - length, 0))
            # post padding
            feat_label[0] = torch.nn.functional.pad(feat_label[0], (0, 0, 0, self.num_time_steps - length))
            feat_label.append(length)
            feat_label_length_list.append(feat_label)
        return feat_label_length_list
    
class MyLSTM(nn.Module):
    def __init__(self, n_feature, hidden_size, lstm_num_layers, lstm_dropout, l2_lambda, if_bi, only_type):
        super(MyLSTM, self).__init__()
        self.bn = nn.BatchNorm1d(n_feature)
        self.lstm = nn.LSTM(input_size=n_feature, hidden_size=hidden_size, num_layers=lstm_num_layers, batch_first=True, dropout=lstm_dropout, bidirectional=if_bi)
        # out_features should be the number of label types
        self.if_bi = if_bi
        if self.if_bi is True:
            hidden_size = hidden_size * 2
        self.only_type = only_type
        out_features_n = 6 if only_type is True else 5
        self.linear_type = nn.Linear(in_features=hidden_size, out_features=out_features_n)
        self.linear_level = nn.Linear(in_features=hidden_size, out_features=3)
        self.l2_lambda = l2_lambda
    
    def forward(self, inputs, lengths):
        # input shape: (batch_size, num_time_steps, n_feature)
        inputs = inputs.permute(0, 2, 1)
        inputs = self.bn(inputs)
        inputs = inputs.permute(0, 2, 1)
        feats = nn.utils.rnn.pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)
        # the shape of h_last: (num_layers * num_directions, batch_size, hidden_size)
        outputs, (h_last, c_last) = self.lstm(feats)
        # when single direction, h_last[-1] is output of the last layer.
        # but when bidirection is true, maybe need to use h_last[-1] and h_last[-2] which from different directions.
        # combine them and change the layer with shape of input is (hidden_size * 2)
        h = h_last[-1] if self.if_bi is False else torch.cat((h_last[-1], h_last[-2]), dim=1)
        emo_type = self.linear_type(h)
        emo_level = self.linear_level(h)

        return emo_type, emo_level

    def l2_regularization(self):
        l2_reg = torch.tensor(0.0).cuda()
        for name, param in self.named_parameters():
            if 'weight' in name:
                l2_reg += torch.norm(param, p=2)
        return self.l2_lambda * l2_reg
    
class DilatedConvolution(nn.Module):
    def __init__(self, n_feature, batch_size, num_time_steps, l2_lambda):
        super(DilatedConvolution, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_feature, out_channels=64, kernel_size=3, dilation=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, dilation=2)
        self.bn2 = nn.BatchNorm1d(128)
        
        # 动态计算全连接层的输入维度
        input_shape = (batch_size, n_feature, num_time_steps)  # 假设的输入尺寸
        conv_output_size = self._get_conv_output(input_shape)
        
        # for type
        self.fc1 = nn.Linear(conv_output_size, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 5)
        # for level
        self.fc3 = nn.Linear(conv_output_size, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc4 = nn.Linear(64, 3)
        
        self.l2_lambda = l2_lambda
        
    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.rand(shape) # (32, 23, 499)
            output = self.pool(self.relu(self.bn1(self.conv1(input)))) # (32, 64, 248)
            output = self.pool(self.relu(self.bn2(self.conv2(output))))  # # (32, 128, 122)
            return torch.flatten(output, 1).size(1)
    
    def l2_regularization(self):
        l2_reg = torch.tensor(0.0).cuda()
        for name, param in self.named_parameters():
            if 'weight' in name:
                l2_reg += torch.norm(param, p=2)
        return self.l2_lambda * l2_reg
    
    def forward(self, inputs, lengths):
        # input shape of conv should be like (batch_size, n_feature, num_time_steps), which is diferent from lstm so permute it.
        x = inputs.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        # Flatten the output for the fully connected layer
        x = torch.flatten(x, 1)
        x_type = self.fc1(x)
        x_type = self.dropout(x_type)
        x_type = self.fc2(x_type)
        x_level = self.fc3(x)
        x_level = self.dropout(x_level)
        x_level = self.fc4(x_level)
        return x_type, x_level

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

def evaluate_classification(save_path, y_true, y_pred, class_labels, task):
    '''
    evaluate model
    注意:如果混淆矩阵热力图只显示第一行数值, 请将matplotlib版本降为3.7.2
    '''
    # if y_true and y_pred are PyTorch Tensor, and maybe on the GPU, use this to convert them:
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    # 计算基本指标
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0, labels=np.unique(y_true))

    # 创建混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap="BuPu", cbar=True)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'{save_path}/Confusion_Matrix_of_{task}.png')
    plt.clf()
    
    # Create a DataFrame for the metrics
    metrics_data = {
        "Accuracy": [accuracy],
        "Precision": [precision],
        "Recall": [recall],
        "F1 Score": [f1]
    }
    metrics_df = pd.DataFrame(metrics_data)
    # Plotting the DataFrame as a table and saving as an image
    fig, ax = plt.subplots(figsize=(6, 1))  # Adjust the size as needed
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, loc='center')
    plt.savefig(f"{save_path}/Metrics_Table_of_{task}.png")
    plt.clf()
    
def line_chart(save_path, data, data_label, x_label, y_label, title, size=(10 ,5)):
    '''
    draw loss curve.
    '''
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

def encode_type_label(label):
    '''
    Anger (ANG)
    Disgust (DIS)
    Fear (FEA)
    Happy/Joy (HAP)
    Neutral (NEU)
    Sad (SAD)
    '''
    if label == 'ANG':
        label_int = 0
    if label == 'DIS':
        label_int = 1
    if label == 'FEA':
        label_int = 2
    if label == 'HAP':
        label_int = 3
    if label == 'SAD':
        label_int = 4
    if label == 'NEU':
        label_int = 5
    return label_int

def encode_level_label(label):
    '''
    Low (LO)
    Medium (MD)
    High (HI)
    Unspecified (XX)
    '''
    if label == 'XX':
        label_int = 3
    if label == 'LO':
        label_int = 0
    if label == 'MD':
        label_int = 1
    if label == 'HI':
        label_int = 2
    return label_int
    
def run(model, data_loader, optimizer, loss_fun, type_only, train=True):
    if train:
        model.train()
    else:
        model.eval()
    # init
    loss_all, loss_type_all, loss_level_all, loss_level = 0.0, 0.0, 0.0, 0.0
    types_int_all = torch.tensor([]).cuda()
    levels_int_all = torch.tensor([]).cuda()
    labels_type_all = torch.tensor([]).cuda()
    labels_level_all = torch.tensor([]).cuda()
    
    for data in data_loader:
        # feats shape为(batch_size, num_time_steps, n_feature),labels shape is (batch_size)
        feats, uttids, lengths = data
        # because labels are included in the uttid from ark files so that process it correctly
        labels_type_str = [uttid.split('_')[2] for uttid in uttids]
        labels_level_str = [uttid.split('_')[3] for uttid in uttids]
        # labels_type and labels_level is int
        labels_type = [encode_type_label(label_str) for label_str in labels_type_str]
        labels_type = torch.tensor(labels_type).cuda()
        labels_level = [encode_level_label(label_str) for label_str in labels_level_str]
        labels_level = torch.tensor(labels_level).cuda()
        if train:
            # zero the gradients of last step
            optimizer.zero_grad()
            
        # types and levels is for the cross-loss witch accept the original output of linear layer
        # int is for comparing predictions and ground-truth to evaluate model
        feats = feats.to(torch.float).cuda()
        types, levels = model(feats.cuda(), lengths)
        
        # use softmax layer to predict the type and level
        
        types_int = torch.argmax(nn.functional.softmax(types, dim=-1), dim=-1)
        levels_int = torch.argmax(nn.functional.softmax(levels, dim=-1), dim=-1) if type_only is False else None
        
        # compute loss
        # nitice: if use CrossEntropyLoss, then just use outputs of linear layer instead of softmax.
        loss_type = loss_fun(types, labels_type) + model.l2_regularization()
        loss_level = loss_fun(levels, labels_level) + model.l2_regularization() if type_only is False else 0.0
        # set a method to compute the total loss, sum or add weights.
        loss = loss_type + loss_level
        
        if train:
            loss.backward()
            optimizer.step()
        
        # for evaluation and showing
        loss_all += loss
        loss_type_all += loss_type
        loss_level_all += loss_level
        types_int_all = torch.cat((types_int_all, types_int), dim=0)
        levels_int_all = torch.cat((levels_int_all, levels_int), dim=0) if type_only is False else None
        labels_type_all = torch.cat((labels_type_all, labels_type), dim=0)
        labels_level_all = torch.cat((labels_level_all, labels_level), dim=0)
    return loss_type_all, loss_level_all, loss_all, types_int_all, labels_type_all, levels_int_all, labels_level_all
    
def adjust_lr(optimizer, step_num, argu=None):
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
        
def wrong_parameter(x):
    print(f'wrong parameter: {x}')
    exit(1)

    

def main(args, parameter_dict):
    torch.cuda.empty_cache()
    
    emo_type_text = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Neutral']
    emo_level_text = ['Low', 'Medium', 'High']
    
    # num_time_steps/length of sequence/number of features in one data.
    num_time_steps = parameter_dict["num_time_steps"]
    # predict only emotion type or type and level
    if_only_type = parameter_dict["only_type"]
    # if use bi-lstm
    if_bi = parameter_dict["if_bi"]
    # set which model to use
    model_type = parameter_dict["model_type"]
    # number of epoch
    n_epoch = parameter_dict["n_epoch"]
    # number of warm-up epoch
    n_warm_up = int(n_epoch / 10)
    # learning rate
    lr = parameter_dict["lr"]
    # number of lstm layers
    lstm_num_layers = parameter_dict["lstm_num_layers"]
    # L2 parameter
    l2_lambda = parameter_dict["l2_lambda"]
    lstm_dropout = parameter_dict["lstm_dropout"]
    # loss function
    loss_name = parameter_dict["loss_name"]
    # optimizer
    optimizer_name = parameter_dict["optimizer_name"]
    # type of feature
    feature = parameter_dict["feature"]
    # early stopping and threshold
    if_early_stopping = parameter_dict["if_early_stopping"]
    es_threshold = parameter_dict["es_threshold"]
    # 
    feature_folder_dict = parameter_dict["feature_folder"]
    if if_only_type is True:
        feature_folder = feature_folder_dict["only_type"]
    else:
        feature_folder = feature_folder_dict["all"]
    
    gpu_name = torch.cuda.get_device_name(0)
    if gpu_name == "NVIDIA GeForce MX350":
        device = 'son'
        num_workers = 0
    else:
        device = 'you'
        num_workers = 16
        
    if feature in parameter_dict:
        feature_params = parameter_dict[feature]
        n_feature = feature_params["n_feature"]
        hidden_size = feature_params["hidden_size"]
        batch_size_train = feature_params["batch_size_train"]
        batch_size_val = feature_params["batch_size_val"]
        if_norm = feature_params["if_norm"]
        # name feature file like "{feature}_train.ark"
        ark_path_train = f'{pwd}/{feature_folder}/train/{feature}.ark'
        ark_path_val = f'{pwd}/{feature_folder}/val/{feature}.ark'            
    else:
        raise ValueError(f"Unsupported feature type: {feature}")
    
    
    # 生成数据集
    dataset_train = MyDataset(ark_path_train, if_norm, num_time_steps)
    dataset_val = MyDataset(ark_path_val, if_norm, num_time_steps)
        
    # 生成data_loader
    # 这里生成的是一个迭代器,返回值batch在第一个，需要设置lstm类参数batch_first
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size_train, shuffle=True, num_workers=num_workers, drop_last=False)
    data_loader_val = DataLoader(dataset_val, batch_size=batch_size_val, shuffle=False, num_workers=num_workers, drop_last=False)
        
    # 实例化模型
    if model_type == 'lstm':
        model = MyLSTM(n_feature, hidden_size, lstm_num_layers, lstm_dropout, l2_lambda, if_bi, if_only_type).cuda()
    elif model_type == 'conv':
        model = DilatedConvolution(n_feature, batch_size_train, num_time_steps, l2_lambda).cuda()
    else:
        wrong_parameter(model_type)
        
    # 定义损失函数
    # 对批次损失值默认求平均，可以改成sum求和
    if loss_name == 'mse':
        loss_fun = nn.MSELoss(reduction='mean').cuda()
    elif loss_name == 'cross':
        loss_fun = nn.CrossEntropyLoss().cuda()
    else:
        wrong_parameter(loss_name)
        
    # 定义优化器
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        wrong_parameter(optimizer_name)

    # 开始训练
    loss_type_train_list = []
    loss_level_train_list = []
    loss_train_list = []
    loss_val_list = []
    loss_type_val_list= []
    loss_level_val_list = []
    
    for epoch in range(n_epoch):
        # adjust learning rate
        adjust_lr(optimizer, epoch)
        # train
        print(f'epoch: {epoch+1}\nThe performance in the dataset_train:')
        loss_type_train, loss_level_train, loss_train, types_int_all, labels_type_all, levels_int_all, labels_level_all = run(model, data_loader_train, optimizer, loss_fun, if_only_type)
        # draw loss curve
        loss_train_list.append(loss_train.item())
        loss_type_train_list.append(loss_type_train.item())
        loss_level_train_list.append(loss_level_train.item())
        print(f'The loss of train: {loss_train:.2f}')
        # val
        loss_type_val, loss_level_val, loss_val, types_int_all, labels_type_all, levels_int_all, labels_level_all = run(model, data_loader_val, optimizer, loss_fun, if_only_type)
        loss_val_list.append(loss_val.item())
        loss_type_val_list.append(loss_type_val.item())
        loss_level_val_list.append(loss_level_val.item())
        print(f'The loss of val: {loss_val:.2f}\n')
        # set the early stopping criterion
        if if_early_stopping:
            pass
    
    # save model
    torch.save(model.state_dict(), os.path.join(log_folder, 'model.pth'))
    
    # visualize the evaluation
    if if_only_type is False:
        emo_type_text = emo_type_text[:-1]
        evaluate_classification(log_folder, labels_level_all, levels_int_all, emo_level_text, 'level')
        line_chart(log_folder, loss_level_train_list, 'loss', 'epoch', 'loss', 'loss_level_train')
        line_chart(log_folder, loss_level_val_list, 'loss', 'epoch', 'loss', 'loss_level_val')
        line_chart(log_folder, loss_type_train_list, 'loss', 'epoch', 'loss', 'loss_type_train')
        line_chart(log_folder, loss_type_val_list, 'loss', 'epoch', 'loss', 'loss_type_val')
    evaluate_classification(log_folder, labels_type_all, types_int_all, emo_type_text, 'type')
    line_chart(log_folder, loss_train_list, 'loss', 'epoch', 'loss', 'loss_train')
    line_chart(log_folder, loss_val_list, 'loss', 'epoch', 'loss', 'loss_val')
    
    # list hyperparameters
    print(f'\n\n\nThe hyperparameter list:\n{parameter_dict}')
    return None
        
    
if __name__ == "__main__":
    # detect GPU
    if not torch.cuda.is_available():
        print("No GPU found")
        exit(1)
    
    # load hyperparameters file
    with open('parameter.json', 'r') as f:
        parameter_group = json.load(f)
        
    # make log folder
    if not os.path.exists('log'):
        os.makedirs('log')
    pwd = os.getcwd()
    os.chdir('./log')
    
    # make sure your hyperparameters file is a list of hyperparameter dicts
    if type(parameter_group) is list:
        for parameter_dict in parameter_group:
            # 获取当前日期和时间
            current_date = time.strftime('%Y-%m-%d', time.localtime())
            current_time = time.strftime('%H_%M', time.localtime())
            # 创建日期文件夹
            date_folder_name = f'logs_{current_date}'
            os.makedirs(date_folder_name, exist_ok=True)
            # 在日期文件夹内创建时间文件夹
            log_folder = os.path.join(pwd, 'log', date_folder_name, f'log_{current_time}')
            os.makedirs(log_folder, exist_ok=True)
            # 切换到文件夹内
            os.chdir(log_folder)
            # 保存原始的sys.stdout
            original_stdout = sys.stdout
            # 打开一个文件来将print的内容写入
            sys.stdout = Logger(stream=original_stdout)
            
            main(sys.argv[1:], parameter_dict)
            
            # 恢复原始的sys.stdout
            sys.stdout = original_stdout
            # os.chdir(f'{pwd}/log')
            # new_logger_folder = f'{log_folder}_{pcc:.3f}'
            # os.rename(log_folder, new_logger_folder)
    else:
        print(f'parameter.json is not a list!')
        exit(1)