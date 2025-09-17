import os
import itertools
import yaml
import logging
logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

from collections import namedtuple

import pandas as pd
import torch
from torch import nn

from src.models.patchTST import PatchTST
from src.models.MCformer import MCformer
from src.learner import Learner, transfer_weights
from src.callback.tracking import *
from src.callback.patch_mask import *
from src.callback.transforms import *
from src.metrics import *
from datautils import *


def load_base_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config):
    with open(config['output']['save_path'] + config['output']['model_name'] + '_config.yml', 'w') as f:
        yaml.dump(config, f)


def set_hyperparameters(config,
                        n_layers,
                        learning_rate=0,
                        n_features=16,
                        dset_pretrain='force_finetune',
                        context_points=460,
                        target_points=46,
                        batch_size=64,
                        num_workers=0,
                        scaler='standard',
                        features='M',
                        patch_len=12,
                        stride=12,
                        revin=0,
                        model_backbone='PatchTST',
                        n_heads=16,
                        d_model=128,
                        d_ff=512,
                        dropout=0.2,
                        head_dropout=0.2,
                        mask_ratio=0.4,
                        n_epochs_pretrain=1,
                        n_epochs_finetune=1,
                        pretrained_model_id=1,
                        model_type='based_model',
                        stack_len=3,
                        lr=1e-4,
                        loss_function='MSELoss'):
    config['hyperparameters'] = {
        'learning_rate': learning_rate,
        'n_layers': n_layers,
        'n_features': n_features,
        'dset': dset_pretrain,
        'use_time_features': False,
        'context_points': context_points,
        'target_points': target_points,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'scaler': scaler,
        'features': features,
        'patch_len': patch_len,
        'stride': stride,
        'revin': revin,
        'model_backbone': model_backbone,
        'n_heads': n_heads,
        'd_model': d_model,
        'd_ff': d_ff,
        'dropout': dropout,
        'head_dropout': head_dropout,
        'mask_ratio': mask_ratio,
        'n_epochs_pretrain': n_epochs_pretrain,
        'n_epochs_finetune': n_epochs_finetune,
        'pretrained_model_id': pretrained_model_id,
        'model_type': model_type,
        'initial_lr': lr,
        'loss_function': loss_function
    }
    num_patch = (max(config['hyperparameters']['context_points'], config['hyperparameters']['patch_len'])-config['hyperparameters']['patch_len']) // config['hyperparameters']['stride'] + 1    
    config['hyperparameters']['num_patch'] = num_patch
    if model_backbone == 'MCformer':
        config['hyperparameters']['stack_len'] = stack_len


def set_model_name(config):
    config['output']['model_name'] = f'{config["hyperparameters"]["model_backbone"]}'\
                                   + f'_layers{config["hyperparameters"]["n_layers"]}'\
                                   + f'_revin{config["hyperparameters"]["revin"]}'\
                                   + f'_loss{config["hyperparameters"]["loss_function"]}'\
                                   + f'_patch{config["hyperparameters"]["patch_len"]}'\
                                   + f'_stride{config["hyperparameters"]["stride"]}'\


def get_model(config, head_type):
    # get model
    if config['hyperparameters']['model_backbone'] == 'PatchTST':
        model = PatchTST(n_vars=config['hyperparameters']['n_features'],
                    target_dim=config['hyperparameters']['target_points'],
                    patch_len=config['hyperparameters']['patch_len'],
                    stride=config['hyperparameters']['stride'],
                    num_patch=config['hyperparameters']['num_patch'],
                    n_layers=config['hyperparameters']['n_layers'],
                    n_heads=config['hyperparameters']['n_heads'],
                    d_model=config['hyperparameters']['d_model'],
                    shared_embedding=True,
                    d_ff=config['hyperparameters']['d_ff'],                        
                    dropout=config['hyperparameters']['dropout'],
                    head_dropout=config['hyperparameters']['head_dropout'],
                    act='gelu',
                    head_type=head_type,
                    res_attention=False
                    )  
    
    elif config['hyperparameters']['model_backbone'] == 'MCformer':
        model = MCformer(n_vars=config['hyperparameters']['n_features'],
                    target_dim=config['hyperparameters']['target_points'],
                    patch_len=config['hyperparameters']['patch_len'],
                    stride=config['hyperparameters']['stride'],
                    num_patch=config['hyperparameters']['num_patch'],
                    n_layers=config['hyperparameters']['n_layers'],
                    n_heads=config['hyperparameters']['n_heads'],
                    d_model=config['hyperparameters']['d_model'],
                    shared_embedding=True,
                    d_ff=config['hyperparameters']['d_ff'],                        
                    dropout=config['hyperparameters']['dropout'],
                    head_dropout=config['hyperparameters']['head_dropout'],
                    act='gelu',
                    head_type=head_type,
                    res_attention=False
                    )  
    else:
        raise ValueError(f'Model backbone {config["hyperparameters"]["model_backbone"]} not supported')
    # print out the model size
    logger.info(f'number of model params {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    return model


def find_lr(config, head_type):
    # get dataloader
    dls = get_dls_namedtuple(config)    
    model = get_model(config, head_type=head_type)
    # get loss
    loss_func = get_loss_func(config)
    # get callbacks
    cbs = [RevInCB(dls.vars, denorm=False)] if config['hyperparameters']['revin'] else []
    if head_type == 'pretrain':
        cbs += [PatchMaskCB(patch_len=config['hyperparameters']['patch_len'], stride=config['hyperparameters']['stride'], mask_ratio=config['hyperparameters']['mask_ratio'])]
    elif head_type in ['prediction', 'force_prediction', 'force_prediction_revin', 'force_regression', 'regression', 'classification']:
        cbs += [PatchCB(patch_len=config['hyperparameters']['patch_len'], stride=config['hyperparameters']['stride'])]
        pretrain_model_name = 'pretrain_' + config['output']['model_name'] + '_epochs-pretrain' + str(config['hyperparameters']['n_epochs_pretrain']) + '.pth'
        model = transfer_weights(config['output']['save_path'] + pretrain_model_name, model)
        
    # define learner
    learn = Learner(dls, model, 
                        loss_func, 
                        lr=config['hyperparameters']['initial_lr'], 
                        cbs=cbs,
                        )                        

    # fit the data to the model
    logger.info('finding learning rate')
    suggested_lr = learn.lr_finder()
    logger.info(f'suggested_lr {suggested_lr}')
    return suggested_lr


def get_loss_func(config):
    if config['hyperparameters']['loss_function'] == 'MSELoss':
        return torch.nn.MSELoss(reduction='mean')
    elif config['hyperparameters']['loss_function'] == 'L1Loss':
        return torch.nn.L1Loss()
    else:
        raise ValueError(f'Loss function {config["hyperparameters"]["loss_function"]} not supported')


def get_dls_namedtuple(config):
    # Combine hyperparameters and data configs
    all_config = {**config['hyperparameters'], **config['data']}
    return get_dls(namedtuple('Params', all_config.keys())(*all_config.values()))


def pretrain_model(config):
    # find learning rate
    suggested_lr = find_lr(config, head_type='pretrain')
    config['hyperparameters']['pretrain_lr'] = suggested_lr
    # get dataloader
    dls = get_dls_namedtuple(config)
    # get model     
    model = get_model(config, head_type='pretrain')
    # get loss
    loss_func = get_loss_func(config)
    # get callbacks
    fname = 'pretrain_' + config['output']['model_name'] + '_epochs-pretrain' + str(config['hyperparameters']['n_epochs_pretrain'])
    cbs = [RevInCB(dls.vars, denorm=False)] if config['hyperparameters']['revin'] else []
    cbs += [
         PatchMaskCB(patch_len=config['hyperparameters']['patch_len'], stride=config['hyperparameters']['stride'], mask_ratio=config['hyperparameters']['mask_ratio']),
         SaveModelCB(monitor='valid_loss', fname=fname,                       
                        path=config['output']['save_path'])
        ]
    # define learner
    learn = Learner(dls, model, 
                        loss_func, 
                        lr=config['hyperparameters']['pretrain_lr'], 
                        cbs=cbs,
                        #metrics=[mse]
                        )                        
    # fit the data to the model
    learn.fit_one_cycle(n_epochs=config['hyperparameters']['n_epochs_pretrain'], lr_max=config['hyperparameters']['pretrain_lr'])
    save_recorders(config, learn, training_type='pretrain')
    

def save_recorders(config, learn, training_type='pretrain'):
    train_loss = learn.recorder['train_loss']
    valid_loss = learn.recorder['valid_loss']
    df = pd.DataFrame(data={'train_loss': train_loss, 'valid_loss': valid_loss})
    df.to_csv(config['output']['save_path'] + '_' + training_type + '_' + config['output']['model_name'] + '_losses.csv', float_format='%.6f', index=False)


def finetune_model(config):
    # find learning rate
    suggested_lr = find_lr(config, head_type=config['hyperparameters']['head_type'])
    config['hyperparameters']['finetune_lr'] = suggested_lr
    # get dataloader
    dls = get_dls_namedtuple(config)
    # get model 
    model = get_model(config, head_type=config['hyperparameters']['head_type'])
    # transfer weight
    pretrain_model_name = 'pretrain_' + config['output']['model_name'] + '_epochs-pretrain' + str(config['hyperparameters']['n_epochs_pretrain']) + '.pth'
    model = transfer_weights(config['output']['save_path'] + pretrain_model_name, model)
    # get loss
    loss_func = get_loss_func(config)
    # get callbacks
    fname = 'finetune_' + config['output']['model_name'] + '_epochs-finetune' + str(config['hyperparameters']['n_epochs_finetune'])
    cbs = [RevInCB(dls.vars, denorm=True)] if config['hyperparameters']['revin'] else []
    cbs.append(PatchCB(patch_len=config['hyperparameters']['patch_len'], stride=config['hyperparameters']['stride']))
    if config['hyperparameters']['revin']: cbs.append(RevInRegressionHeadCB(config['hyperparameters']['n_features'], config['hyperparameters']['d_model'], config['hyperparameters']['num_patch'], config['hyperparameters']['target_points'], config['hyperparameters']['head_dropout']))
    cbs.append(SaveModelCB(monitor='valid_loss', fname=fname, path=config['output']['save_path']))
    # define learner
    learn = Learner(dls, model, 
                        loss_func, 
                        lr=config['hyperparameters']['finetune_lr'], 
                        cbs=cbs,
                        metrics=[mae]
                        )                            
    # fit the data to the model
    learn.fine_tune(n_epochs=config['hyperparameters']['n_epochs_finetune'], base_lr=config['hyperparameters']['finetune_lr'], freeze_epochs=1)
    save_recorders(config, learn, training_type='finetune')


def main():
    ##### load base config
    config = load_base_config('configs/config_long_PatchTST_base.yml')

    N_LAYERS = [1, 2, 3]
    model_backbone = ['PatchTST'] #, 'MCformer']
    revin = [1] #[0, 1]
    loss_function = ['MSELoss']

    for n_layers, model_backbone, revin, loss_function in itertools.product(N_LAYERS, model_backbone, revin, loss_function):
        logger.info(f'Running {n_layers} layers and {model_backbone} model')
        ##### set hyperparameters for run
        set_hyperparameters(config, n_layers=n_layers, model_backbone=model_backbone, revin=revin, loss_function=loss_function)
        set_model_name(config)
        save_config(config)
        
        ##### pretrain model
        logger.info(f'Pretraining {n_layers} layers and {model_backbone} model')
        pretrain_model(config)
        logger.info('pretraining completed')
        save_config(config)

        ##### finetune model
        logger.info(f'Finetuning {n_layers} layers and {model_backbone} model')
        if revin == 1:
            config['hyperparameters']['head_type'] = 'force_prediction_revin'
        else:
            config['hyperparameters']['head_type'] = 'force_prediction'
        finetune_model(config)
        save_config(config)
        logger.info('finetuning completed')


if __name__ == '__main__':
    main()
