import os
import time
import argparse
import random
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
import torch.distributed as dist
import torch.utils.data.distributed
import numpy as np
from tensorboardX import SummaryWriter

from utils import load_config, save_checkpoint, load_checkpoint,log_results
from dataset import get_crohme_dataset
from models.can import CAN
from dist_training import train, eval

parser = argparse.ArgumentParser(description='model training')
parser.add_argument('--dataset', default='CROHME', type=str, help='数据集名称')
parser.add_argument('--check', action='store_true', help='测试代码选项')
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)

def main():
    dist.init_process_group("nccl", init_method='env://')

    if not args.dataset:
        print('请提供数据集名称')
        exit(-1)

    if args.dataset == 'CROHME':
        config_file = 'config.yaml'
    elif args.dataset == 'half_CROHME':
        config_file = 'half_crohme_train_data_config.yaml'

    """加载config文件"""
    params = load_config(config_file)

    """设置随机种子"""
    seed = params['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params['device'] = device

    if args.dataset == 'CROHME' or args.dataset == 'half_CROHME':
        train_loader, eval_loader = get_crohme_dataset(params, use_aug=False, is_dist=True)

    model = CAN(params)
    now = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    # model.name = f'{params["experiment"]}_{now}_decoder-{params["decoder"]["net"]}'
    model_name = f'{params["experiment"]}_{now}_decoder-{params["decoder"]["net"]}'
    print(model_name)
    #print(model.name)

    if args.check:
        writer = None
    else:
        writer = SummaryWriter(f'{params["log_dir"]}/{model_name}')

    optimizer = getattr(torch.optim, params['optimizer'])(model.parameters(), lr=float(params['lr']),
                                                        eps=float(params['eps']), weight_decay=float(params['weight_decay']))
    if params['using_amp']:
        scaler = GradScaler()
    else: scaler = None

    if params['finetune']:
        print('加载预训练模型权重')
        print(f'预训练权重路径: {params["checkpoint"]}')
        load_checkpoint(model, optimizer, params['checkpoint'])
        # 将模型中的标准批归一化（BatchNorm）层转换为同步批归一化（SyncBatchNorm）层
        # 具体来说，当调用了torch.nn.SyncBatchNorm.convert_sync_batchnorm()函数时，
        # 它会遍历模型的所有子模块（包括您的backbone，如ResNet），检查是否存在BatchNorm层，并将其替换为相应的SyncBatchNorm层
    model = model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # output_device：这是输出设备的ID，用于收集梯度并执行梯度平均操作。在这个代码片段中，args.local_rank指定的设备将用于这个目的
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    if not args.check:
        if not os.path.exists(os.path.join(params['checkpoint_dir'], model_name)):
            os.makedirs(os.path.join(params['checkpoint_dir'], model_name), exist_ok=True)
        os.system(f'cp {config_file} {os.path.join(params["checkpoint_dir"], model_name, model_name)}.yaml')
    """在CROHME上训练"""
    if args.dataset == 'CROHME' or args.dataset == 'half_CROHME':
        min_score, init_epoch = 0, 0
        log_file = 'checkpoints/log.txt'
        dist.barrier()

        for epoch in range(init_epoch, params['epochs']):
            train_loader.sampler.set_epoch(epoch)

            train_loss, train_word_score, train_exprate = train(params, model, optimizer, epoch, train_loader, writer=writer, scaler=scaler)
            # 在每个epoch结束后，进行分布式训练的进程进行同步，以确保每个进程都完成了当前epoch的训练，然后才进行下一个操作
            dist.barrier()
            # 当前进程是主进程（dist.get_rank() == 0），则调用eval()函数来进行模型验证


            if epoch >= params['valid_start']:
                        eval_loss, eval_word_score, eval_exprate = eval(params, model, epoch, eval_loader, writer=writer, scaler=scaler)
                        # 使用 all_reduce 对每个卡上的评估结果进行归约，求和
                        eval_loss_sum = torch.tensor(eval_loss).to(device)
                        eval_word_score_sum = torch.tensor(eval_word_score).to(device)
                        eval_exprate_sum = torch.tensor(eval_exprate).to(device)
                        dist.all_reduce(eval_loss_sum)
                        dist.all_reduce(eval_word_score_sum)
                        dist.all_reduce(eval_exprate_sum)

                        # 取平均值
                        num_eval_samples = len(eval_loader.dataset)
                        eval_loss_avg = eval_loss_sum / dist.get_world_size()
                        eval_word_score_avg = eval_word_score_sum / dist.get_world_size()
                        eval_exprate_avg = eval_exprate_sum /params['len_HME100K']

                        print(f'Epoch: {epoch+1} loss: {eval_loss_avg:.4f} word score: {eval_word_score_avg:.4f} ExpRate: {eval_exprate_avg:.4f}')
                        if eval_exprate_avg > min_score and dist.get_rank() == 0 and not args.check:
                            min_score = eval_exprate_avg
                            save_checkpoint(model, model_name, optimizer, eval_word_score_avg, eval_exprate_avg, epoch+1,
                                            optimizer_save=params['optimizer_save'], path=params['checkpoint_dir'])
                            log_results(epoch, eval_loss_avg, eval_exprate_avg, log_file)


            if dist.get_rank() == 0:
                save_checkpoint(model, model_name, optimizer, 0, 0, 'latest',
                                optimizer_save=params['optimizer_save'], path=params['checkpoint_dir'])

if __name__ == "__main__":
    main()