import argparse
import os
import statistics
import glob
import torch
from datetime import datetime
from torch.utils.data import DataLoader, DistributedSampler, Subset
import torch.distributed as dist
from tensorboardX import SummaryWriter
# import wandb # if use wandb, import this.
#import swanlab as wandb # if you use swanlab, import this as wandb for compatibility
# if you dont use either, comment the above two lines and any wandb lines in the code and loss file.
import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import multi_gpu_utils
from icecream import ic
import sys
import warnings

warnings.filterwarnings('ignore')

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env opencood/tools/train_ddp.py --hypes_yaml ${CONFIG_FILE} [--model_dir  ${CHECKPOINT_FOLDER}

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    parser.add_argument("--half", action='store_true',
                        help="whether train with half precision")
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--visualize_feature', action='store_true',
                        help="whether visualize feature maps")
    parser.add_argument('--perception_loss', action='store_true',
                        help="whether use perception loss")                    
    opt = parser.parse_args()
    return opt

"""
def init_wandb(hypes, current_time):
    if dist.get_rank() == 0:

        run_name = hypes['name'] + current_time.strftime("_%Y_%m_%d_%H_%M_%S")
        wandb.init(project="GenComm", name=run_name, config=hypes)
"""
        

def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    multi_gpu_utils.init_distributed_mode(opt)
    hypes['model']['args']['visualize_feature'] = opt.visualize_feature
    hypes['model']['args']['perception_loss'] = opt.perception_loss
    #current_time = datetime.now()
    #init_wandb(hypes, current_time)
    current_time = datetime.now()
    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False)
    
    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset)
        sampler_val = DistributedSampler(opencood_validate_dataset, shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes['train_params']['batch_size'], drop_last=True)
        
        if 'verify_mode' in hypes and hypes['verify_mode']:
            tiny_opencood_train_dataset = Subset(opencood_train_dataset, range(1300,2400))
            tiny_opencood_validate_dataset = Subset(opencood_validate_dataset, range(300))
            print("Verify mode, only use part samples")
            train_loader = DataLoader(tiny_opencood_train_dataset,
                                    batch_sampler=batch_sampler_train,
                                    num_workers=8,
                                    collate_fn=opencood_train_dataset.collate_batch_train)
            val_loader = DataLoader(tiny_opencood_validate_dataset,
                                    sampler=sampler_val,
                                    num_workers=8,
                                    collate_fn=opencood_train_dataset.collate_batch_train,
                                    drop_last=False)
        else:
            train_loader = DataLoader(opencood_train_dataset,
                                    batch_sampler=batch_sampler_train,
                                    num_workers=8,
                                    collate_fn=opencood_train_dataset.collate_batch_train)
            val_loader = DataLoader(opencood_validate_dataset,
                                    sampler=sampler_val,
                                    num_workers=8,
                                    collate_fn=opencood_train_dataset.collate_batch_train,
                                    drop_last=False)
    else:
        if 'verify_mode' in hypes and hypes['verify_mode']:
            tiny_opencood_train_dataset = Subset(opencood_train_dataset, range(1300,2400))
            tiny_opencood_validate_dataset = Subset(opencood_validate_dataset, range(300))
            print("Verify mode, only use part samples")
            train_loader = DataLoader(tiny_opencood_train_dataset,
                                    batch_size=hypes['train_params'][
                                        'batch_size'],
                                    num_workers=8,
                                    collate_fn=opencood_train_dataset.collate_batch_train,
                                    shuffle=True,
                                    pin_memory=True,
                                    drop_last=True)
            val_loader = DataLoader(tiny_opencood_validate_dataset,
                                    batch_size=hypes['train_params']['batch_size'],
                                    num_workers=8,
                                    collate_fn=opencood_train_dataset.collate_batch_train,
                                    shuffle=True,
                                    pin_memory=True,
                                    drop_last=True)
        else:
            train_loader = DataLoader(opencood_train_dataset,
                                    batch_size=hypes['train_params'][
                                        'batch_size'],
                                    num_workers=8,
                                    collate_fn=opencood_train_dataset.collate_batch_train,
                                    shuffle=True,
                                    pin_memory=True,
                                    drop_last=True)
            val_loader = DataLoader(opencood_validate_dataset,
                                    batch_size=hypes['train_params']['batch_size'],
                                    num_workers=8,
                                    collate_fn=opencood_train_dataset.collate_batch_train,
                                    shuffle=True,
                                    pin_memory=True,
                                    drop_last=True)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('Top-level modules:')
    for name, module in model.named_children():  # 只迭代顶层
        print(f"- {name}: {module.__class__.__name__}")
    # record lowest validation loss checkpoint.
    lowest_val_loss = 1e5
    lowest_val_epoch = -1

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
        
    # ddp setting
    model_without_ddp = model

     # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer)

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, lowest_val_loss, lowest_val_epoch, model = train_utils.load_full_checkpoint(
            saved_path, model, optimizer, scheduler
        )
        print(f"resume from {init_epoch} epoch.")
    else:
        init_epoch = 0
        saved_path = train_utils.setup_train(hypes, current_time)

    if opt.distributed:
        model = \
            torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=True,
                                                      ) # True
        model_without_ddp = model.module

    
    
    

    # record training
    writer = SummaryWriter(saved_path)
    
    train_utils.optim_to_device(optimizer, device)

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    print('Training start')
    epoches = hypes['train_params']['epoches']
    supervise_single_flag = False if not hasattr(opencood_train_dataset, "supervise_single") else opencood_train_dataset.supervise_single
    log_interval = hypes['train_params'].get("log_interval", 0)
    # used to help schedule learning rate

    iter = 0
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if opt.distributed:
            sampler_train.set_epoch(epoch)

        # the model will be evaluation mode during validation
        model.train()
        try:  # heter_model stage2
            model_without_ddp.model_train_init()
        except:
            print("No model_train_init function")

        train_ave_loss = []
        for i, batch_data in enumerate(train_loader):
            iter += 1
            # 判断本地 batch 是否有效
            is_valid = batch_data is not None and batch_data['ego']['object_bbx_mask'].sum() > 0

            # 分布式同步：只要有一个 rank 无效，所有 rank 都 skip
            if opt.distributed:
                is_valid_tensor = torch.tensor([int(is_valid)], device=device)
                dist.all_reduce(is_valid_tensor, op=dist.ReduceOp.MIN)
                is_valid = bool(is_valid_tensor.item())

            if not is_valid:
                continue

            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)
            batch_data['ego']['epoch'] = epoch
            batch_data['ego']['dataset'] = opencood_validate_dataset
            batch_data['ego']['label_dict']['single'] = batch_data['ego']['label_dict_single']
            if not opt.half:
                ouput_dict = model(batch_data['ego'])
                final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
            else:
                with torch.cuda.amp.autocast():
                    ouput_dict = model(batch_data['ego'])
                    final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])

            train_ave_loss.append(final_loss.item())

            # optional step logging
            if log_interval > 0 and dist.get_rank() == 0 and (i % log_interval == 0):
                criterion.logging(epoch, i, len(train_loader), writer, iter=iter)

            if supervise_single_flag:
                if not opt.half:
                    final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single") * hypes['train_params'].get("single_weight", 1)
                else:
                    with torch.cuda.amp.autocast():
                        final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single") * hypes['train_params'].get("single_weight", 1)
                if log_interval > 0 and dist.get_rank() == 0 and (i % log_interval == 0):
                    criterion.logging(epoch, i, len(train_loader), writer, suffix="_single", iter=iter)

            if not opt.half:
                # Anti-Deadlock sync for DDP:
                # Synchronize requires_grad across all ranks. If ANY rank produced a detached loss,
                # we skip the backward pass on ALL ranks to prevent DDP deadlock or hook errors.
                local_requires_grad = final_loss.requires_grad
                if opt.distributed:
                    rg_tensor = torch.tensor([int(local_requires_grad)], device=device)
                    dist.all_reduce(rg_tensor, op=dist.ReduceOp.MIN)
                    global_requires_grad = bool(rg_tensor.item())
                else:
                    global_requires_grad = local_requires_grad

                if not global_requires_grad:
                    continue
                
                final_loss.backward()
                optimizer.step()
            else:
                # Anti-Deadlock sync for DDP in AMP:
                local_requires_grad = final_loss.requires_grad
                if opt.distributed:
                    rg_tensor = torch.tensor([int(local_requires_grad)], device=device)
                    dist.all_reduce(rg_tensor, op=dist.ReduceOp.MIN)
                    global_requires_grad = bool(rg_tensor.item())
                else:
                    global_requires_grad = local_requires_grad

                if not global_requires_grad:
                    continue

                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()
        #if dist.get_rank() == 0:
            #train_ave_loss = statistics.mean(train_ave_loss)
            #wandb.log({"epoch": epoch, "train_loss": train_ave_loss,}, step = iter)


        # torch.cuda.empty_cache() # it will destroy memory buffer
        if epoch % hypes['train_params']['save_freq'] == 0:
            if dist.get_rank() == 0: # 仅限 Rank 0
                ckpt = {
                    "epoch": epoch + 1,
                    "model": strip_module_prefix(model.state_dict()),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "lowest_val_loss": lowest_val_loss,
                    "lowest_val_epoch": lowest_val_epoch,
                }
                torch.save(ckpt, os.path.join(saved_path, f"net_epoch{epoch+1}.pth"))
            
        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    # 判断本地 batch 是否有效
                    is_valid = batch_data is not None
                    if opt.distributed:
                        is_valid_tensor = torch.tensor([int(is_valid)], device=device)
                        dist.all_reduce(is_valid_tensor, op=dist.ReduceOp.MIN)
                        is_valid = bool(is_valid_tensor.item())
                    if not is_valid:
                        continue
                    model.zero_grad()
                    optimizer.zero_grad()
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data['ego']['epoch'] = epoch
                    batch_data['ego']['dataset'] = opencood_validate_dataset
                    batch_data['ego']['label_dict']['single'] = batch_data['ego']['label_dict_single']
                    batch_data['ego']['eval_with_recon_loss'] = True
                    ouput_dict = model_without_ddp(batch_data['ego'])

                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                    valid_ave_loss.append(final_loss.item())
            # 1. 计算当前进程的局部平均 Loss
            local_loss = statistics.mean(valid_ave_loss)
            local_loss_tensor = torch.tensor(local_loss).to(device)
            # 2. 同步所有 Rank 的 Loss (取平均值)
            if opt.distributed:
                dist.all_reduce(local_loss_tensor, op=dist.ReduceOp.SUM)
                global_val_loss = local_loss_tensor.item() / dist.get_world_size()
            else:
                global_val_loss = local_loss
            
            # 3. 仅在 Rank 0 进行比较和文件操作
            if dist.get_rank() == 0:
                print('At epoch %d, the global validation loss is %f' % (epoch, global_val_loss))
                writer.add_scalar('Validate_Loss', global_val_loss, epoch)

                if global_val_loss < lowest_val_loss:
                    # 删除旧的 best 文件
                    if lowest_val_epoch != -1:
                        old_best_path = os.path.join(saved_path, f"net_epoch_bestval_at{lowest_val_epoch}.pth")
                        if os.path.exists(old_best_path):
                            os.remove(old_best_path)
                    
                    # 更新记录并保存新文件
                    lowest_val_loss = global_val_loss
                    lowest_val_epoch = epoch + 1
                    
                    ckpt = {
                        "epoch": epoch + 1,
                        "model": strip_module_prefix(model.state_dict()),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "lowest_val_loss": lowest_val_loss,
                        "lowest_val_epoch": lowest_val_epoch,
                    }
                    save_name = os.path.join(saved_path, f"net_epoch_bestval_at{lowest_val_epoch}.pth")
                    torch.save(ckpt, save_name)

        #scheduler.step(epoch)
        scheduler.step()
        
        opencood_train_dataset.reinitialize()

    print('Training Finished, checkpoints saved to %s' % saved_path)

    #显式销毁进程组，防止 NCCL 超时
    if opt.distributed:
            dist.destroy_process_group()


def strip_module_prefix(state_dict):
    """去掉module.前缀"""
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v
            for k, v in state_dict.items()}


if __name__ == '__main__':
    main()