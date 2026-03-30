# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import glob
import importlib
import yaml
import os
import re
from datetime import datetime
import shutil
import torch
import torch.optim as optim


###added loss&load are used for stamp

def load_original_model(saved_path, model, protocol=True):
    """
    Load protocol and ego model

    Parameters
    __________
    saved_path : str
       model saved path
    model : opencood object
        The model instance.

    Returns
    -------
    model : opencood object
        The model instance loaded pretrained params.
    """
    
    
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)
    ego_path = os.path.join(saved_path, 'ego.pth')
    
    if protocol:
        protocol_path = os.path.join(saved_path, 'protocol.pth')
        assert os.path.exists(protocol_path), '{} not found'.format(protocol_path)
        print("load protocal checkpoint from %s" % protocol_path)
        loaded_protocol_state_dict = torch.load(protocol_path, map_location='cpu')
        check_missing_key(model.state_dict(), loaded_protocol_state_dict)
        model.load_state_dict(loaded_protocol_state_dict, strict=False)
        print()
    
    assert os.path.exists(ego_path), '{} not found'.format(ego_path)
    print("load ego checkpoint from %s" % ego_path)
    loaded_ego_state_dict = torch.load(ego_path, map_location='cpu')
    check_missing_key(model.state_dict(), loaded_ego_state_dict)
    model.load_state_dict(loaded_ego_state_dict, strict=False)
    
    return model

def create_losses_heter(hypes):
    """
    Create the loss function based on the given loss name.

    Parameters
    ----------
    hypes : dict
        Configuration params for training.
    Returns
    -------
    criterion : opencood.object
        The loss function.
    """
    criterion_dict = dict()
    for modality_name in hypes['loss'].keys():
        
        loss_func_name = hypes['loss'][modality_name]['core_method']
        loss_func_config = hypes['loss'][modality_name]['args']

        loss_filename = "opencood.loss." + loss_func_name
        loss_lib = importlib.import_module(loss_filename)
        loss_func = None
        target_loss_name = loss_func_name.replace('_', '')

        for name, lfunc in loss_lib.__dict__.items():
            if name.lower() == target_loss_name.lower():
                loss_func = lfunc

        if loss_func is None:
            print('loss function not found in loss folder. Please make sure you '
                'have a python file named %s and has a class '
                'called %s ignoring upper/lower case' % (loss_filename,
                                                        target_loss_name))
            exit(0)

        criterion_dict[modality_name] = loss_func(loss_func_config)
    return criterion_dict

def create_adapter_loss(hypes):
    """
    Create the loss function based on the given loss name.

    Parameters
    ----------
    hypes : dict
        Configuration params for training.
    Returns
    -------
    criterion : opencood.object
        The loss function.
    """
    loss_func_name = hypes['loss_adapter']['core_method']
    loss_func_config = hypes['loss_adapter']['args']

    loss_filename = "opencood.loss." + loss_func_name
    loss_lib = importlib.import_module(loss_filename)
    loss_func = None
    target_loss_name = loss_func_name.replace('_', '')

    for name, lfunc in loss_lib.__dict__.items():
        if name.lower() == target_loss_name.lower():
            loss_func = lfunc

    if loss_func is None:
        print('loss function not found in loss folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (loss_filename,
                                                       target_loss_name))
        exit(0)

    criterion = loss_func(loss_func_config)
    return criterion

def backup_script(full_path, folders_to_save=["models", "data_utils", "utils", "loss"]):
    target_folder = os.path.join(full_path, 'scripts')
    if not os.path.exists(target_folder):
        if not os.path.exists(target_folder):
            os.mkdir(target_folder)
    
    current_path = os.path.dirname(__file__)  # __file__ refer to this file, then the dirname is "?/tools"

    for folder_name in folders_to_save:
        ttarget_folder = os.path.join(target_folder, folder_name)
        source_folder = os.path.join(current_path, f'../{folder_name}')
        shutil.copytree(source_folder, ttarget_folder)

def check_missing_key(model_state_dict, ckpt_state_dict):
    checkpoint_keys = set(ckpt_state_dict.keys())
    model_keys = set(model_state_dict.keys())

    missing_keys = model_keys - checkpoint_keys
    extra_keys = checkpoint_keys - model_keys

    missing_key_modules = set([keyname.split('.')[0] for keyname in missing_keys])
    extra_key_modules = set([keyname.split('.')[0] for keyname in extra_keys])

    print("------ Loading Checkpoint ------")
    if len(missing_key_modules) == 0 and len(extra_key_modules) ==0:
        return

    print("Missing keys from ckpt:")
    print(*missing_key_modules,sep='\n',end='\n\n')
    # print(*missing_keys,sep='\n',end='\n\n')

    print("Extra keys from ckpt:")
    print(*extra_key_modules,sep='\n',end='\n\n')
    print(*extra_keys,sep='\n',end='\n\n')

    print("You can go to tools/train_utils.py to print the full missing key name!")
    print("--------------------------------")

def optim_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                if k == "step":
                    state[k] = v.cpu()      # step 留在 CPU
                else:
                    state[k] = v.to(device) # exp_avg 等搬到 GPU

def load_full_checkpoint(saved_path, model, optimizer, scheduler):
    assert os.path.exists(saved_path), f"{saved_path} not found"

    def find_last(save_dir):
        file_list = glob.glob(os.path.join(save_dir, "net_epoch*.pth"))
        if not file_list:
            return None
        epochs_exist = []
        for f in file_list:
            base = os.path.basename(f)
            m = re.findall(r"epoch(\d+)", base)
            if m:
                epochs_exist.append(int(m[0]))
        return max(epochs_exist) if epochs_exist else None
    
    ckpt_path = None

    last_epoch = find_last(saved_path)
    if last_epoch is not None:  # 优先最后一次常规 ckpt
        ckpt_path = os.path.join(saved_path, f"net_epoch{last_epoch}.pth")
        print(f"resuming last checkpoint epoch {last_epoch}")
    else:  # 没有常规 ckpt，再尝试 best
        best_list = glob.glob(os.path.join(saved_path, "net_epoch_bestval_at*.pth"))
        if best_list:
            ckpt_path = best_list[0]
            print(f"resuming best checkpoint {ckpt_path}")

    if ckpt_path is None:
        return 0, 1e5, -1, model  # 没找到 ckpt

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model" not in ckpt.keys(): # 兼容之前只保存了 state_dict 的旧版本 ckpt
        loaded_state_dict = torch.load(os.path.join(saved_path,
                         'net_epoch%d.pth' % last_epoch), map_location='cpu')
        check_missing_key(model.state_dict(), loaded_state_dict)
        model.load_state_dict(loaded_state_dict, strict=False)
        return last_epoch, 1e5, -1, model

    check_missing_key(model.state_dict(), ckpt["model"])
    model.load_state_dict(ckpt["model"], strict=False)
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    init_epoch = ckpt.get("epoch", 0)
    lowest_val_loss = ckpt.get("lowest_val_loss", 1e5)
    lowest_val_epoch = ckpt.get("lowest_val_epoch", -1)
    return init_epoch, lowest_val_loss, lowest_val_epoch, model

def load_saved_model(saved_path, model):
    """
    Load saved model if exiseted

    Parameters
    __________
    saved_path : str
       model saved path
    model : opencood object
        The model instance.

    Returns
    -------
    model : opencood object
        The model instance loaded pretrained params.
    """
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)

    def findLastCheckpoint(save_dir):
        file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*epoch(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch_ = max(epochs_exist)
        else:
            initial_epoch_ = 0
        return initial_epoch_

    file_list = glob.glob(os.path.join(saved_path, 'net_epoch_bestval_at*.pth'))
    if file_list:
        assert len(file_list) == 1
        print("resuming best validation model at epoch %d" % \
                eval(file_list[0].split("/")[-1].rstrip(".pth").lstrip("net_epoch_bestval_at")))
        loaded_state_dict = torch.load(file_list[0] , map_location='cpu')
        if "model" in loaded_state_dict.keys(): # 兼容之前保存了整个 ckpt 的版本
            loaded_state_dict = loaded_state_dict["model"]
            loaded_state_dict = {k.replace("module.", "", 1): v for k, v in loaded_state_dict.items()}
        check_missing_key(model.state_dict(), loaded_state_dict)
        model.load_state_dict(loaded_state_dict, strict=False)
        return eval(file_list[0].split("/")[-1].rstrip(".pth").lstrip("net_epoch_bestval_at")), model

    initial_epoch = findLastCheckpoint(saved_path)
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        loaded_state_dict = torch.load(os.path.join(saved_path,
                         'net_epoch%d.pth' % initial_epoch), map_location='cpu')
        if "model" in loaded_state_dict.keys(): # 兼容之前保存了整个 ckpt 的版本
            loaded_state_dict = loaded_state_dict["model"]
            loaded_state_dict = {k.replace("module.", "", 1): v for k, v in loaded_state_dict.items()}
        check_missing_key(model.state_dict(), loaded_state_dict)
        model.load_state_dict(loaded_state_dict, strict=False)

    return initial_epoch, model


def setup_train(hypes, current_time):
    """
    Create folder for saved model based on current timestep and model name

    Parameters
    ----------
    hypes: dict
        Config yaml dictionary for training:
    """
    model_name = hypes['name']


    folder_name = current_time.strftime("_%Y_%m_%d_%H_%M_%S")
    folder_name = model_name + folder_name

    current_path = os.path.dirname(__file__)
    current_path = os.path.join(current_path, '../logs')

    full_path = os.path.join(current_path, folder_name)

    if not os.path.exists(full_path):
        if not os.path.exists(full_path):
            try:
                os.makedirs(full_path)
                backup_script(full_path)
            except FileExistsError:
                pass
        save_name = os.path.join(full_path, 'config.yaml')
        with open(save_name, 'w') as outfile:
            yaml.dump(hypes, outfile)

        

    return full_path


def create_model(hypes):
    """
    Import the module "models/[model_name].py

    Parameters
    __________
    hypes : dict
        Dictionary containing parameters.

    Returns
    -------
    model : opencood,object
        Model object.
    """
    backbone_name = hypes['model']['core_method']
    backbone_config = hypes['model']['args']

    model_filename = "opencood.models." + backbone_name
    model_lib = importlib.import_module(model_filename)
    model = None
    target_model_name = backbone_name.replace('_', '')

    if backbone_name == "heter_model_baseline_w_heterlora" and hasattr(model_lib, "build_heter_lora_model"):
        return model_lib.build_heter_lora_model(backbone_config)


    for name, cls in model_lib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print('backbone not found in models folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (model_filename,
                                                       target_model_name))
        exit(0)
    instance = model(backbone_config)
    return instance


def create_loss(hypes):
    """
    Create the loss function based on the given loss name.

    Parameters
    ----------
    hypes : dict
        Configuration params for training.
    Returns
    -------
    criterion : opencood.object
        The loss function.
    """
    loss_func_name = hypes['loss']['core_method']
    loss_func_config = hypes['loss']['args']

    loss_filename = "opencood.loss." + loss_func_name
    loss_lib = importlib.import_module(loss_filename)
    loss_func = None
    target_loss_name = loss_func_name.replace('_', '')

    for name, lfunc in loss_lib.__dict__.items():
        if name.lower() == target_loss_name.lower():
            loss_func = lfunc

    if loss_func is None:
        print('loss function not found in loss folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (loss_filename,
                                                       target_loss_name))
        exit(0)

    criterion = loss_func(loss_func_config)
    return criterion


def setup_optimizer(hypes, model):
    """
    Create optimizer corresponding to the yaml file

    Parameters
    ----------
    hypes : dict
        The training configurations.
    model : opencood model
        The pytorch model
    """
    method_dict = hypes['optimizer']
    optimizer_method = getattr(optim, method_dict['core_method'], None)
    if not optimizer_method:
        raise ValueError('{} is not supported'.format(method_dict['name']))
    if 'args' in method_dict:
        return optimizer_method(model.parameters(),
                                lr=method_dict['lr'],
                                **method_dict['args'])
    else:
        return optimizer_method(model.parameters(),
                                lr=method_dict['lr'])


def setup_lr_schedular(hypes, optimizer, init_epoch=None):
    """
    Set up the learning rate schedular.

    Parameters
    ----------
    hypes : dict
        The training configurations.

    optimizer : torch.optimizer
    """
    lr_schedule_config = hypes['lr_scheduler']
    last_epoch = init_epoch if init_epoch is not None else 0
    

    if lr_schedule_config['core_method'] == 'step':
        from torch.optim.lr_scheduler import StepLR
        step_size = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif lr_schedule_config['core_method'] == 'multistep':
        from torch.optim.lr_scheduler import MultiStepLR
        milestones = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = MultiStepLR(optimizer,
                                milestones=milestones,
                                gamma=gamma)

    else:
        from torch.optim.lr_scheduler import ExponentialLR
        gamma = lr_schedule_config['gamma']
        scheduler = ExponentialLR(optimizer, gamma)

    for _ in range(last_epoch):
        scheduler.step()

    return scheduler

def to_device(inputs, device):
    if isinstance(inputs, list):
        return [to_device(x, device) for x in inputs]
    elif isinstance(inputs, dict):
        return {k: to_device(v, device) for k, v in inputs.items()}
    else:
        if isinstance(inputs, int) or isinstance(inputs, float) \
                or isinstance(inputs, str) or not hasattr(inputs, 'to'):
            return inputs
    
        return inputs.to(device, non_blocking=True)


