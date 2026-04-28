import argparse
import os
import statistics
import torch
from torch.utils.data import DataLoader, DistributedSampler, Subset
import torch.distributed as dist
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils_stamp as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import multi_gpu_utils

import warnings
warnings.filterwarnings('ignore')

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True, help="data generation yaml file needed ")
    parser.add_argument("--model_dir", default="", help="Continued training path")
    parser.add_argument("--fusion_method", "-f", default="intermediate", help="passed to inference.")
    parser.add_argument('--flop_count', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    opt = parser.parse_args()
    return opt

def strip_module_prefix(state_dict):
    """Remove module. prefix for saved state dict"""
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v
            for k, v in state_dict.items()}

def _count_params(module, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def _print_trainable_param_stats(model):
    print('Top-level modules:')
    for name, module in model.named_children():
        trainable_params = _count_params(module, trainable_only=True)
        total_params = _count_params(module, trainable_only=False)
        print(
            f"- {name}: {module.__class__.__name__} | "
            f"trainable={trainable_params:,} / total={total_params:,}"
        )

    total_trainable = _count_params(model, trainable_only=True)
    total_params = _count_params(model, trainable_only=False)
    print(
        f"Trainable params (all modules): {total_trainable:,} "
        f"({total_trainable / 1e6:.3f} M) / Total params: {total_params:,}"
    )

def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    multi_gpu_utils.init_distributed_mode(opt)

    print("Dataset Building")
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes, visualize=False, train=False)

    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset)
        sampler_val = DistributedSampler(opencood_validate_dataset, shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes["train_params"]["batch_size"], drop_last=True
        )

        train_loader = DataLoader(
            opencood_train_dataset,
            batch_sampler=batch_sampler_train,
            num_workers=8,
            collate_fn=opencood_train_dataset.collate_batch_train,
        )
        val_loader = DataLoader(
            opencood_validate_dataset,
            sampler=sampler_val,
            num_workers=8,
            collate_fn=opencood_train_dataset.collate_batch_train,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            opencood_train_dataset,
            batch_size=hypes["train_params"]["batch_size"],
            num_workers=8,
            collate_fn=opencood_train_dataset.collate_batch_train,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            opencood_validate_dataset,
            batch_size=hypes["train_params"]["batch_size"],
            num_workers=8,
            collate_fn=opencood_train_dataset.collate_batch_train,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    print("Creating Model")
    model = train_utils.create_model(hypes)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _print_trainable_param_stats(model)
    # record lowest validation loss checkpoint.
    lowest_val_loss = 1e5
    lowest_val_epoch = -1

    # define the loss
    criterion_dict = train_utils.create_losses_heter(hypes)
    criterion_adapter = train_utils.create_adapter_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        # Assuming you load original model
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
        lowest_val_epoch = init_epoch
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, init_epoch=init_epoch)
        print(f"resume from {init_epoch} epoch.")
    else:
        raise NotImplementedError("model_dir must be provided for training adapter")

    if torch.cuda.is_available():
        model.to(device)

    # ddp setting
    model_without_ddp = model

    if opt.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[opt.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    # record training
    writer = SummaryWriter(saved_path)
    train_utils.optim_to_device(optimizer, device)

    print("Training start")
    epoches = hypes["train_params"]["epoches"]
    log_interval = hypes['train_params'].get("log_interval", 0)    
    for epoch in [0] if opt.flop_count else range(init_epoch, max(epoches, init_epoch)):
        if opt.distributed:
            sampler_train.set_epoch(epoch)

        for param_group in optimizer.param_groups:
            print("learning rate %f" % param_group["lr"])
            
        model.train()
        try:
            model_without_ddp.model_train_init()
        except:
            print("No model_train_init function")

        for i, batch_data in enumerate(train_loader):
            is_valid = batch_data is not None and batch_data["ego"]["object_bbx_mask"].sum() > 0
            if opt.distributed:
                is_valid_tensor = torch.tensor([int(is_valid)], device=device)
                dist.all_reduce(is_valid_tensor, op=dist.ReduceOp.MIN)
                is_valid = bool(is_valid_tensor.item())

            if not is_valid:
                continue

            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)
            batch_data["ego"]["epoch"] = epoch
            
            output_dict, output_feat = model(batch_data["ego"])
            loss_adapter = 0
            if output_feat is not None:
                FM, FP2M, FM2P2M, FP, FM2P = output_feat
                loss_adapter = criterion_adapter(FM, FP2M, FM2P2M, FP, FM2P)
                if log_interval>0 and dist.get_rank() == 0 and (i % log_interval == 0):
                    criterion_adapter.logging(epoch, i, len(train_loader), writer)
                
            final_loss_dict = dict()
            if output_dict is not None:
                for modality_name in criterion_dict.keys():
                    if modality_name == "m0":
                        final_loss_dict[modality_name] = criterion_dict[modality_name](
                            output_dict[modality_name], batch_data["ego"]["label_dict_protocol"]
                        )
                        if log_interval>0 and dist.get_rank() == 0 and (i % log_interval == 0):
                            criterion_dict[modality_name].logging(epoch, i, len(train_loader), writer)
                    else:
                        final_loss_dict[modality_name] = criterion_dict[modality_name](
                            output_dict[modality_name], batch_data["ego"]["label_dict"]
                        )
                        if log_interval>0 and dist.get_rank() == 0 and (i % log_interval == 0):
                            criterion_dict[modality_name].logging(epoch, i, len(train_loader), writer)
            
            final_loss = sum(final_loss_dict.values()) + loss_adapter

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

        if epoch % hypes["train_params"]["save_freq"] == 0:
            if not opt.distributed or dist.get_rank() == 0:
                torch.save(strip_module_prefix(model.state_dict()), os.path.join(saved_path, "net_epoch%d.pth" % (epoch + 1)))

        if epoch % hypes["train_params"]["eval_freq"] == 0:
            valid_ave_loss = []

            model.eval()
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    is_valid = batch_data is not None
                    if opt.distributed:
                        is_valid_tensor = torch.tensor([int(is_valid)], device=device)
                        dist.all_reduce(is_valid_tensor, op=dist.ReduceOp.MIN)
                        is_valid = bool(is_valid_tensor.item())
                    if not is_valid:
                        continue

                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data["ego"]["epoch"] = epoch
                    
                    output_dict, output_feat = model(batch_data["ego"])
                    loss_adapter = 0
                    if output_feat is not None:
                        FM, FP2M, FM2P2M, FP, FM2P = output_feat
                        loss_adapter = criterion_adapter(FM, FP2M, FM2P2M, FP, FM2P)
                        
                    final_loss_dict = dict()
                    if output_dict is not None:
                        for modality_name in criterion_dict.keys():
                            if modality_name == "m0":
                                final_loss_dict[modality_name] = criterion_dict[modality_name](
                                    output_dict[modality_name], batch_data["ego"]["label_dict_protocol"]
                                )
                            else:
                                final_loss_dict[modality_name] = criterion_dict[modality_name](
                                    output_dict[modality_name], batch_data["ego"]["label_dict"]
                                )
                    final_loss = sum(final_loss_dict.values()) + loss_adapter
                    if not torch.isnan(final_loss) and not torch.isinf(final_loss):
                        valid_ave_loss.append(final_loss.item())

            local_loss = statistics.mean(valid_ave_loss) if len(valid_ave_loss) > 0 else 0
            if opt.distributed:
                local_loss_tensor = torch.tensor(local_loss).to(device)
                dist.all_reduce(local_loss_tensor, op=dist.ReduceOp.SUM)
                global_val_loss = local_loss_tensor.item() / dist.get_world_size()
            else:
                global_val_loss = local_loss

            if not opt.distributed or dist.get_rank() == 0:
                print("At epoch %d, the validation loss is %f" % (epoch, global_val_loss))
                writer.add_scalar("Validate_Loss", global_val_loss, epoch)

                if global_val_loss < lowest_val_loss:
                    if lowest_val_epoch != -1:
                        old_best_path = os.path.join(saved_path, "net_epoch_bestval_at%d.pth" % (lowest_val_epoch))
                        if os.path.exists(old_best_path):
                            os.remove(old_best_path)
                            
                    lowest_val_loss = global_val_loss
                    lowest_val_epoch = epoch + 1
                    save_name = os.path.join(saved_path, "net_epoch_bestval_at%d.pth" % (lowest_val_epoch))
                    torch.save(strip_module_prefix(model.state_dict()), save_name)

        scheduler.step()

        opencood_train_dataset.reinitialize()

    print("Training Finished, checkpoints saved to %s" % saved_path)

    if opt.distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()