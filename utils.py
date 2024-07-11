import os
import yaml
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn as nn
import shutil
import logging
import time
import matplotlib.pyplot as plt

def dict_to_namespace(config_dict):
    if isinstance(config_dict, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in config_dict.items()})
    elif isinstance(config_dict, list):
        return [dict_to_namespace(i) for i in config_dict]
    else:
        return config_dict

def load_config(file_path='config.yml'):
    with open(file_path, 'r') as stream:
        try:
            config_data = yaml.safe_load(stream)
            return dict_to_namespace(config_data)
        except yaml.YAMLError as exc:
            print(f"Error loading YAML file: {exc}")
            return None
        
class Selector:
    def __init__(self, opt) -> None:
        self.loss = opt.loss.type
        self.optimizer = opt.optimizer
        
    def loss(self, *args):
        if self.loss == 'mse':
            loss_function = nn.MSELoss(*args)
        if self.loss == 'cross-entropy':
            loss_function = nn.CrossEntropyLoss(*args)

        return loss_function
    
    def optimizer(self, *args):
        if self.optimizer == 'adam':
            optim = torch.optim.Adam(*args)
        if self.optimizer == 'adagrad':
            optim = torch.optim.Adagrad(*args)

        return optim

def get_weights_file_path(weight_folder, model_basename, data_name, epoch: str):
    model_filename = f"{model_basename}-{data_name}-{epoch}.pt"
    path = os.path.join(os.getcwd(), weight_folder, model_basename, data_name, model_filename)
    directory_path = os.path.join(os.getcwd(), weight_folder, model_basename, data_name)
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path, exist_ok=True)
    return path

def latest_weights_file_path(weight_folder, model_basename, data_name):
    path = os.path.join(os.getcwd(), weight_folder, model_basename, data_name)    
    model_filename = f"{model_basename}-{data_name}*"
    weights_files = list(Path(path).glob(model_filename))
    # print(weights_files, path)
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])

def mask(time, agent):
    sz = time*agent
    mask = torch.zeros(sz, sz)
    for step in range(time):
        start = agent * step
        stop = start + agent
        # The players can look at the other players.
        mask[start:stop, :stop] = 1

    # mask = mask.masked_fill(mask == 0, float("-inf"))
    # mask = mask.masked_fill(mask == 1, float(0.0))
    return mask==0 # shape: (time*n_particles, time*n_particles)

def dynamic_mask(time, agent):
    sz = time*agent
    mask = torch.triu(torch.ones(sz,sz), diagonal=1)
    # mask = mask.masked_fill(mask==1, float('inf'))
    return mask==1 # shape: (time*n_particles, time*n_particles)

def cross_attention_mask(n_encoder_seq, n_decoder_seq, n_particles):
    assert n_encoder_seq>n_decoder_seq, "The seq length for the encoder should be larger than decoder seq."
    mask = torch.zeros(n_decoder_seq*n_particles, n_encoder_seq*n_particles)
    lookup = n_encoder_seq-n_decoder_seq
    j = 0
    for i in range(n_decoder_seq):
            rstart, rend = i*n_particles, (i+1)*n_particles
            cstart, cend = j*n_particles, (j+lookup)*n_particles
            mask[rstart:rend, cstart:cend] = 1
            j+=1
    return mask==0 # shape: (n_decoder_seq*n_particles, n_encoder_seq*n_particles) 

def copy_file(src_file, dest_path):
    try:
        # Ensure the source file exists
        if not os.path.isfile(src_file):
            raise FileNotFoundError(f"Source file '{src_file}' not found.")
        
        # Determine if the destination path is a directory or a file path
        if os.path.isdir(dest_path):
            # If it's a directory, construct the full destination file path
            dest_file = os.path.join(dest_path, os.path.basename(src_file))
        else:
            # If it's a file path, use it as is
            dest_file = dest_path
        
        # Ensure the destination directory exists
        dest_dir = os.path.dirname(dest_file)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        
        # Copy the file to the destination path
        shutil.copy2(src_file, dest_file)
        # print(f'File "{src_file}" copied to "{dest_file}"')
    except Exception as e:
        print(f"An error occurred while copying: {e}")

def logging_setup(path):
    logging.basicConfig(level=logging.INFO, 
                        filename=os.path.join(path, 'log.log'), 
                        filemode='a', 
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    return logging

def get_logger(id, path, name='log'):
    logger = logging.getLogger(f'expt-{id}')
    handler = logging.FileHandler(os.path.join(path, f'{name}.log'))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

time_consumed = lambda x: round((time.time()-x)/60, 3)

class Watch:
    def __init__(self):
        self.st = time.time()

    def start(self):
        self.st = time.time()
        return 

    def end(self):
        return f'Done: {time_consumed(self.st)} mins.'

def plot_trajectory(data, n_particles, location, path, multi_step_location=None, hf_time=None):
        plt.clf()
        data = data.cpu()
        color = ['r','b','g','y','c']
        ch = data[0,0,:,0]
        for i in range(n_particles):
                # ch = data[0,0,:,0]
                marker = '+' if ch[i]==1 else '_'
                plt.scatter(data.squeeze(0)[0,i,1], data.squeeze(0)[0,i,2], color='black', marker=marker)
                # plt.scatter(location[0,i,0], location[0,i,1], color='black')
                plt.plot(data.squeeze(0)[:,i,1], data.squeeze(0)[:,i,2], label=f'particle {i}', color=color[i], linewidth=0.5)
                plt.plot(location[:,i,0], location[:,i,1], '--', color=color[i])
                if multi_step_location is not None:
                    plt.plot(multi_step_location[:,i,0], multi_step_location[:,i,1], color=color[i], alpha=0.4, linewidth=0.8)
                    # plt.scatter(multi_step_location[0,i,0], multi_step_location[0,i,1], color=color[i], marker='.', alpha=0.3)
                    assert hf_time is not None, "hf_time not avialable."
                    plt.scatter(data.squeeze(0)[hf_time,i,1], data.squeeze(0)[hf_time,i,2], color=color[i], marker='*')
                    plt.scatter(multi_step_location[:,i,0], multi_step_location[:,i,1], color=color[i], alpha=0.4, marker='.')
                plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.savefig(path)
        # plt.show()

