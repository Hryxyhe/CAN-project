import os
import pdb

import cv2
import yaml
import math
import torch
import numpy as np
from difflib import SequenceMatcher


def load_config(yaml_path):
    try:
        with open(yaml_path, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    except:
        print('尝试UTF-8编码....')
        with open(yaml_path, 'r', encoding='UTF-8') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    if not params['experiment']:
        print('实验名不能为空!')
        exit(-1)
    if not params['train_image_path']:
        print('训练图片路径不能为空！')
        exit(-1)
    if not params['train_label_path']:
        print('训练label路径不能为空！')
        exit(-1)
    if not params['word_path']:
        print('word dict路径不能为空！')
        exit(-1)
    if 'train_parts' not in params:
        params['train_parts'] = 1
    if 'valid_parts' not in params:
        params['valid_parts'] = 1
    if 'valid_start' not in params:
        params['valid_start'] = 0
    if 'word_conv_kernel' not in params['attention']:
        params['attention']['word_conv_kernel'] = 1
    return params


def update_lr(optimizer, current_epoch, current_step, steps, epochs, initial_lr):
    if current_epoch < 1:
        new_lr = initial_lr / steps * (current_step + 1)
    elif 1 <= current_epoch <= 200:
        new_lr = 0.5 * (1 + math.cos((current_step + 1 + (current_epoch - 1) * steps) * math.pi / (200 * steps))) * initial_lr
    else:
        new_lr = 0.5 * (1 + math.cos((current_step + 1 + (current_epoch - 1) * steps) * math.pi / (epochs * steps))) * initial_lr   
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def save_checkpoint(model, model_name,optimizer,word_score, ExpRate_score, epoch, optimizer_save=False, path='checkpoints', multi_gpu=False, local_rank=0):
    filename = f'{os.path.join(path, model_name)}/ExpRate-{ExpRate_score:.6f}_{epoch}.pth'#_WordRate-{word_score:.4f}
    if optimizer_save:
        state = {
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
    else:
        state = {
            'model': model.module.state_dict()
        }
    torch.save(state, filename)
    print(f'Save checkpoint: {filename}\n')
    return filename


def load_checkpoint(model, optimizer, path, rank=0):
    state = torch.load(path, map_location='cpu')

    if 'model' in state:
        model_state_dict = state['model']

        # 如果是分布式训练，需要处理带有 "module." 前缀的键
        if rank == 0 and any(key.startswith('module.') for key in model_state_dict):
            new_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
            model.load_state_dict(new_state_dict, strict=True)
        else:
            model.load_state_dict(model_state_dict, strict=True)
    else:
        raise ValueError("Model state_dict not found in checkpoint")

    if optimizer is not None and 'optimizer' in state:
        optimizer.load_state_dict(state['optimizer'])
    elif optimizer is not None:
        print(f'Warning: No optimizer state_dict in the checkpoint')

    return model, optimizer
# def load_checkpoint(model, optimizer, path, rank=0):
#     state = torch.load(path, map_location='cpu')
#     if 'model' in state:
#         # 如果是分布式训练，需要处理带有 "module." 前缀的键
#         if rank == 0 and 'module.' in next(iter(state['model'])):
#             new_state_dict = {}
#             for k, v in state['model'].items():
#                 name = k[7:]  # 去掉 "module." 前缀
#                 new_state_dict[name] = v
#
#             state['model'] = new_state_dict
#
#         model.load_state_dict(state['model'],strict=True)
#     else:
#         raise ValueError("Model state_dict not found in checkpoint")
#
#     if optimizer is not None and 'optimizer' in state:
#         optimizer.load_state_dict(state['optimizer'])
#     elif optimizer is not None:
#         print(f'Warning: No optimizer state_dict in the checkpoint')
#
#     return model, optimizer



class Meter:
    def __init__(self, alpha=0.9):
        self.nums = []
        self.exp_mean = 0
        self.alpha = alpha

    @property
    def mean(self):
        return np.mean(self.nums)

    def add(self, num):
        if len(self.nums) == 0:
            self.exp_mean = num
        self.nums.append(num)
        self.exp_mean = self.alpha * self.exp_mean + (1 - self.alpha) * num


def cal_score(word_probs, word_label, mask):
    line_right = 0
    if word_probs is not None:
        _, word_pred = word_probs.max(2)
    word_scores = [SequenceMatcher(None, s1[:int(np.sum(s3))], s2[:int(np.sum(s3))], autojunk=False).ratio() * (len(s1[:int(np.sum(s3))]) + len(s2[:int(np.sum(s3))])) / len(s1[:int(np.sum(s3))]) / 2
              for s1, s2, s3 in zip(word_label.cpu().detach().numpy(), word_pred.cpu().detach().numpy(), mask.cpu().detach().numpy())]
    
    batch_size = len(word_scores)
    for i in range(batch_size):
        if word_scores[i] == 1:
            line_right += 1

    ExpRate = line_right / batch_size
    word_scores = np.mean(word_scores)
    return word_scores, ExpRate


def draw_attention_map(image, attention):
    h, w = image.shape
    attention = cv2.resize(attention, (w, h))
    attention_heatmap = ((attention - np.min(attention)) / (np.max(attention) - np.min(attention))*255).astype(np.uint8)
    attention_heatmap = cv2.applyColorMap(attention_heatmap, cv2.COLORMAP_JET)
    image_new = np.stack((image, image, image), axis=-1).astype(np.uint8)
    attention_map = cv2.addWeighted(attention_heatmap, 0.4, image_new, 0.6, 0.)
    return attention_map


def draw_counting_map(image, counting_attention):
    h, w = image.shape
    counting_attention = torch.clamp(counting_attention, 0.0, 1.0).numpy()
    counting_attention = cv2.resize(counting_attention, (w, h))
    counting_attention_heatmap = (counting_attention * 255).astype(np.uint8)
    counting_attention_heatmap = cv2.applyColorMap(counting_attention_heatmap, cv2.COLORMAP_JET)
    image_new = np.stack((image, image, image), axis=-1).astype(np.uint8)
    counting_map = cv2.addWeighted(counting_attention_heatmap, 0.4, image_new, 0.6, 0.)
    return counting_map


def cal_distance(word1, word2):
    m = len(word1)
    n = len(word2)
    if m*n == 0:
        return m+n
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            a = dp[i-1][j] + 1
            b = dp[i][j-1] + 1
            c = dp[i-1][j-1]
            if word1[i-1] != word2[j-1]:
                c += 1
            dp[i][j] = min(a, b, c)
    return dp[m][n]


def compute_edit_distance(prediction, label):
    prediction = prediction.strip().split(' ')
    label = label.strip().split(' ')
    distance = cal_distance(prediction, label)
    return distance

def log_results( epoch,eval_loss, eval_exprate,log_file):
    """
    日志记录函数，用于记录每一批次训练和验证的信息

    :param epoch: 当前epoch的序号
    :param eval_loss: 验证损失
    :param eval_word_score: 验证词级准确率
    :param eval_exprate: 验证准确率
    """
    log_str = f'Epoch: [{epoch+1}]'

    # 记录验证信息
    log_str += f',Eval Loss: {eval_loss:.6f}, Eval ExpRate: {eval_exprate:.6f}'
    if log_file:
        with open(log_file, 'a') as f:
            f.write(log_str + '\n')
