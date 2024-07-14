import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from model import Encoder
import logging
from utils import *
import argparse
import random
from data.charge.charge_dataset import ChargedParticleData
from torchsummary import summary
import io
from contextlib import redirect_stdout

# set the seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed) # If using GPU
# torch.cuda.manual_seed_all(seed) # If using multi-GPU
# torch.autograd.set_detect_anomaly(True)

def parser_config():
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--data', type=str, default='charge', help='Name of the dataset being used.')
        parser.add_argument('--model', type=str, default='transformer', help='Name of the model being used.')
        parser.add_argument('--config-file', type=str, default='config.yml', help='Configuration file.')
        parser.add_argument('--gpu', type=int, default=0, help='GPU id to use (default: 0)')
        # parser.add_argument('--test-mode', type=bool, action='store_true', help='Run the train file in code test mode.')
        parser.add_argument('--test-mode', action='store_true', help='Run the train file in code test mode.')
        args = parser.parse_args()
        return args

def test_step(model, data, train_opt, n_time_steps):
        # data = batch['data'] # shape: (batch, time, n_particles, n_features)
        batch = data.size(0) 
        n_particles = data.size(2)
        n_features = data.size(-1)
        time = data.size(1)
        charge = data[:,:,:,0]
        input = data # shape: (batch, time, n_particles, n_features)

        out = torch.empty(batch, n_time_steps, n_particles, n_features)
        for t in range(n_time_steps):
                output = model(input)
                output = output.reshape(-1, time, n_particles, n_features-1) # shape: (batch, time, n_particles, n_features-1)
                output = torch.concat([charge.unsqueeze(-1), output], dim=-1)
                input = torch.concat([input[:, 1:, :, :], output[:, -1, :, :].unsqueeze(1)], dim=1) # shape: (batch, time, n_particles, n_features)

                out[:, t, :, :] = output[:, -1, :, :]
                
        return out # shape: (batch, n_time_steps, n_particles, n_features)                        

def train_model(model, device, train_opt, train_dataloader, valid_dataloader, init_epoch, optim, criteria, logger, result_dir):
        # model training                
        train_epoch_loss, valid_epoch_loss, valid_epochs = [],[],[]
        logger.info('')
        logger.info('========== Starting model training. ==========')
        wt.start()
        for epoch in tqdm(range(init_epoch, train_opt.epochs), desc='Epoch'):
                train_batch_loss = []
                torch.cuda.empty_cache()
                model.train()
                i = 0
                for batch in tqdm(train_dataloader, desc=f'epoch-{epoch}:: train'):
                        data = batch['data'].to(device).float() # shape: (batch, time, n_particles, n_features)
                        if torch.isnan(data).any():
                                logger.error("Oops, NaN detected in data")
                                exit(0)
                        # n_particles = data.size(2)
                        n_features = data.size(-1)
                        # time = data.size(1)

                        # data 
                        label = data[:, 1:, :, 1:].reshape(-1, n_features-1).clone() # shape: (batch*{time-1}*n_particles, n_features-1)
                        input = data[:, :-1, :, :].clone() # shape: (batch, {time-1}, n_particles, n_features)
                        
                        # model prediction
                        output = model(input)

                        # back propagate
                        train_loss = criteria(output, label.contiguous())
                        train_loss.backward()

                        # parameter update
                        optim.step()
                        optim.zero_grad(set_to_none=True)

                        train_batch_loss.append(train_loss.item())
                        logger.debug(f'epoch: {epoch+1}-batch: {i+1} :: loss: {train_loss.item()}')
                        #break
                train_loss_val = torch.mean(torch.tensor(train_batch_loss))
                train_epoch_loss.append(train_loss_val)

                # save the model instance
                if train_opt.save_model and (epoch+1)%train_opt.save_step==0:
                        file = os.path.join(result_dir, 'weights', f'epoch-{epoch+1}.pt')
                        torch.save({
                                'epoch': epoch,
                                'model': model.state_dict(),
                                'optimizer': optim.state_dict(),
                                'model_opt': model_opt, 
                                'train_opt': train_opt,
                                'loss': train_loss_val
                        }, f = file)

                # validation step
                if not train_opt.skip_validation and epoch%train_opt.val_step==0:
                        model.eval()
                        val_batch_loss = []
                        valid_epochs.append(epoch+1)
                        with torch.no_grad():
                                for batch in tqdm(valid_dataloader, desc=f'epoch-{epoch}:: validation'):
                                        data = batch['data'].to(device).float()
                                        label = data[:, 1:, :, 1:].reshape(-1, n_features-1).clone() # shape: (batch*{time-1}*n_particles, n_features-1)
                                        input = data[:, :-1, :, :].clone() # shape: (batch, {time-1}, n_particles, n_features)

                                        output = model(input) # shape: (batch, time, n_particles, n_features-1)
                                        # loss computation
                                        val_loss = criteria(output, label.contiguous())
                                        val_batch_loss.append(val_loss.item())
                                        #break
                                val_loss_val = torch.mean(torch.tensor(val_batch_loss))
                                valid_epoch_loss.append(val_loss_val)
                        logger.info(f'epoch:{epoch}:: train_loss:{train_loss_val} | val_loss:{val_loss_val}')
                else:
                        logger.info(f'epoch:{epoch}:: train_loss:{train_loss_val}')
                
        # training summary
        plt.plot(range(init_epoch+1, train_opt.epochs+1), train_epoch_loss, label='training')
        plt.plot(valid_epochs, valid_epoch_loss, label='validation')
        plt.title('Loss Plot')
        plt.xlabel('epochs')
        plt.ylabel('loss value')
        plt.legend()
        plt.savefig(os.path.join(result_dir, 'loss.png'))
        plt.clf()
        # summary.info(f'best train loss: {min(train_epoch_loss)} @ epoch: {train_epoch_loss.index(min(train_epoch_loss))+1}')
        # summary.info(f'best validation loss: {min(valid_epoch_loss)} @ epoch: {valid_epochs[valid_epoch_loss.index(min(valid_epoch_loss))]}')
        with open(os.path.join(result_dir, 'summary.txt'), 'a') as f:
                f.write('\n\n---------------- Training Result ----------------\n')
                f.write(f'training epochs: {train_opt.epochs-init_epoch} on {torch.cuda.get_device_name(device.index)}')
                f.write(f'\nbest train loss: {min(train_epoch_loss)} @ epoch: {train_epoch_loss.index(min(train_epoch_loss))+1} \n')
                f.write(f'best validation loss: {min(valid_epoch_loss)} @ epoch: {valid_epochs[valid_epoch_loss.index(min(valid_epoch_loss))]} \n')

        logger.info(wt.end())

def device_config(args, logger):
        if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                logger.info(f'Number of GPU avialable: {num_gpus}')
                if args.gpu >= num_gpus or args.gpu < 0:
                        logger.error(f"GPU ID {args.gpu} is not valid. Available GPU IDs are 0 to {num_gpus - 1}.")
                        sys.exit(1)
                else:
                        device = torch.device(f'cuda:{args.gpu}')
                        device_name = torch.cuda.get_device_name(device.index) if device.type == 'cuda' else 'CPU'
                        logger.info(f'Used device: {device_name}')
        else:
                if args.gpu != 0:
                        logger.warning(f"CUDA is not available. Falling back to CPU.")
                device = torch.device('cpu')
        return device

if __name__=='__main__':
        args = parser_config()
        # logging
        now = datetime.now()
        tm = now.strftime('%Y-%m-%d %H:%M')

        results = os.path.join(os.getcwd(), 'results')
        if args.test_mode:
                results = os.path.join(os.getcwd(), '.bin')
                if os.path.exists(results):
                        shutil.rmtree(results)
        os.makedirs(results, exist_ok=True)
        entries = os.listdir(results)
        # entries.sort()
        expts = int(entries[-1].split('|')[0].strip().split('-')[1].strip()) if len(entries)>0 else 0
        # expts = len([_ for _ in entries if os.path.isdir(os.path.join(results, _))])

        # result directory configurations
        result_dir = os.path.join(results, f'expt-{expts+1}| {tm}')
        os.makedirs(result_dir, exist_ok=True)
        copy_file(os.path.join(os.getcwd(), args.config_file), os.path.join(result_dir, 'config.yml'))
        logging = logging_setup(results)
        logger = get_logger(expts+1, result_dir)
        # summary = get_logger(f'summary:{expts+1}', result_dir, name='summary')
        wt = Watch()
        logger.info('')
        logger.info('')
        logger.info(f'++++++++++++++++++++++++++ Experiment: {expts+1} ++++++++++++++++++++++++++')
        logger.info('')
        try:
                # configurations
                logger.info('Loading config file.')
                wt.start()
                config = load_config(args.config_file)
                train_opt = config.train
                train_opt.num_workers = os.cpu_count()-1
                model_opt = vars(config.model).get(args.model)
                data_opt = vars(config.data).get(args.data)
                logger.info(wt.end())
        except:
                logger.error('Oops, error in selected argument for the file check with config file.')
                exit(0)
        os.makedirs(os.path.join(result_dir, train_opt.weight_folder), exist_ok=True)

        # hardware setup..
        logger.info('')
        logger.info('Loading device.')
        device = device_config(args, logger)

        try:
                # loading dataset
                logger.info('')
                logger.info('Loading dataset.')
                wt.start()
                directory = os.path.join(config.data.directory, data_opt.name, f'charged{data_opt.n_particles}')

                train_data = ChargedParticleData('train', directory, data_opt.n_particles, data_opt.name_format)
                valid_data = ChargedParticleData('valid', directory, data_opt.n_particles, data_opt.name_format)
                test_data = ChargedParticleData('test', directory, data_opt.n_particles, data_opt.name_format)

                train_dataloader = DataLoader(train_data, batch_size=train_opt.batch_size, shuffle=True, num_workers=train_opt.num_workers)
                valid_dataloader = DataLoader(train_data, batch_size=train_opt.batch_size, shuffle=False)
                test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)
                logger.info(wt.end())
        except FileNotFoundError:
                logger.error('Oops, dataset code/data-files not found.')
                exit(0)

        # load model
        logger.info('')
        logger.info('Loading model.')
        wt.start()
        model = Encoder(model_opt, train_data.n_features, train_data.n_particles, train_opt.use_agent_id).to(device)
        with io.StringIO() as buf, redirect_stdout(buf):
                summary(model, input_size=(20,5,5))
                model_summary = buf.getvalue()
        with open(os.path.join(result_dir, 'summary.txt'), 'a') as f:
                f.write('\n---------------- Model Summary ----------------\n\n')
                f.write('Input shape: (batch=32, time=20, n_particles=5, n_features=5)\n')
                f.write(model_summary)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"No. of parameters: {n_params}")
        logger.info(wt.end())

        # optimizer & loss function
        optim = torch.optim.Adam(model.parameters(), lr=train_opt.lr)
        criteria = nn.MSELoss().to(device)
        if args.test_mode:
                train_opt.epochs = 1
                model_opt.n_encoder_layer = 1

        init_epoch = 0
        print(f'\nStarting Experiment {expts+1}.\n------------------------------------------')
        train_model(model, device, train_opt, train_dataloader, valid_dataloader, init_epoch, optim, criteria, logger, result_dir)
        
        logger.info('')
        logger.info('Starting test procedures.')       
        wt.start() 
        print()
        with torch.no_grad():
                for i in tqdm(range(5), desc='test step'):
                        data = random.choice(list(test_dataloader))['data'].to(device).float() # shape: (1, time, n_particles, n_features)
                        assert data.size(0)==1, "Batch size of test dataloader should be 1."
                        time = data.size(1)
                        input = data[:, :-1, :, :].clone() # shape: (batch, {time-1}, n_particles, n_features)
                        output = model(input) # shape: (-1, n_features-1)
                        output = output.view(input.size(1), input.size(2), input.size(-1)-1).cpu()
                        location = output[:,:,:2] # shape: (time, n_particles, 2)
                        os.makedirs(os.path.join(result_dir, 'plots'), exist_ok=True)
                        path = os.path.join(result_dir, 'plots', f'test-{i+1}.png')

                        hf_time = int(time*data_opt.encoder_ratio)
                        n_time_steps = data.size(1)-hf_time
                        multi_step_input = data[:, :hf_time, :, :].clone() # shape: (batch, s1, n_particles, n_features)
                        multi_step_output = test_step(model, multi_step_input, train_opt, n_time_steps).cpu() # shape: (batch, n_time_steps, n_particles, n_features)
                        multi_step_location = multi_step_output.squeeze(0)[:,:,1:3] # shape: (time, n_particles, 2)

                        plot_trajectory(data, train_data.n_particles, location, path, multi_step_location, hf_time)
        logger.info(wt.end())
        print(f'\n Experiment {expts+1} complete!!')