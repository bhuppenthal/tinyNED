from functools import partial
import tempfile
from pathlib import Path

from model import EntDataset, NED

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import ray
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
from ray.air import session

def make_argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str,
                        default= "./saved_models/trained_NED.pt",
                        help="name of model file")
    parser.add_argument('--data_dir', type=str,
                        default="/home/tomcat/entrainment/fisher_processed_files/fisher_h5_files",
                        help='location of h5 files')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--cuda_name', action='store_true', default=1,
                        help='choose cuda device to use')

    return parser

def load_data(data_dir):
    fdset_train = EntDataset(data_dir + '/train_Fisher_nonorm.h5')
    fdset_val = EntDataset(data_dir + '/val_Fisher_nonorm.h5')
    fdset_test = EntDataset(data_dir + '/test_Fisher_nonorm.h5')

    return fdset_train, fdset_val, fdset_test

def train(config, data_dir, cuda_name):
    net = NED()

    device = 'cuda:'+str(cuda_name)
    net.to(device)

    # criterion = F.smooth_l1_loss()
    optimizer = optim.Adam(net.parameters(), lr=config['lr'])

    checkpoint = get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / 'data.pkl'
            with open(data_path, 'rb') as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state['epoch']
            net.load_state_dict(checkpoint_state['net_state_dict'])
            optimizer.load_state_dict(checkpoint_state['optimizer_state_dict'])
    else:
        start_epoch = 0

    train_set, val_set, _ = load_data(data_dir)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=True, num_workers=8)

    # training loop
    for epoch in range(start_epoch, 10):
        net.train()
        epoch_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            reconstructions = net(x)
            # loss = criterion(reconstructions, y)
            loss = F.smooth_l1_loss(reconstructions, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.data
        print(f'===> Epoch {epoch} : Average loss {epoch_loss/len(train_loader.dataset):.4f}')

        # validation loss
        net.eval()
        val_loss = 0
        for idx, (x, y) in enumerate(val_loader):
            x = x.to(device)
            y = y.to(device)

            reconstructions = net(x)
            # loss = criterion(reconstructions,y)
            loss = F.smooth_l1_loss(reconstructions, y)
            val_loss += loss.data
        val_loss /= len(val_loader.dataset)
        print(f'===> Validation set loss {val_loss:.4f}')

        checkpoint_data = {
            'epoch': epoch,
            'net_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / 'data.pkl'
            with open(data_path, 'wb') as fp:
                pickle.dump(checkpoint_data, fp)

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            session.report({'loss': val_loss.cpu().item()}, checkpoint=checkpoint)
    print("finished training!")

def test_accuracy(net, data_dir):
    net.eval()
    net.to("cuda:0") # just in case

    hf = h5py.File(data_dir + 'test_Fisher_nonorm.h5', 'r')
    X = np.array(hf['dataset'])

    test_loss = 0
    fake_test_loss = 0
    Loss = []
    Fake_loss = []

    for idx, data in enumerate(X):
        x = data[:228]
        y = data[228:-1]

        idx_same_speaker = list(np.where(X[:,-1] == data[-1]))[0]
        ll = random.choice(list(idx_same_speaker - set([idx])))
        y_fake = X[ll, 228:-1]

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        y_fake = torch.from_numpy(y_fake)

        x = x.to(device)
        y = y.to(device)
        y_fake = y_fake.to(device)

        z_x = net.embedding(x)
        z_y = net.embedding(y)
        z_y_fake = net.embedding(y_fake)

        loss_real = F.smooth_l1_loss(z_x, z_y, size_average=False).data
        loss_fake = F.smooth_l1_loss(z_x, z_y_fake, size_average=False).data

        test_loss += loss_real
        fake_test_loss += loss_fake
        Loss.append(loss_real)
        Fake_loss.append(loss_fake)
    
    accuracy = float(np.sum(Loss < Fake_loss)) / Loss.shape[0]
    return accuracy

if __name__ == '__main__':
    '''
    Drawing out a set of configurations for training
    '''
    parser = make_argument_parser()
    args = parser.parse_args()

    '''
    Raytune will dray a uniform log ditributions between two values, and returns 
    the value with the best validation loss.
    '''
    config = {
        'lr': tune.loguniform(1e-4, 1e-1)
    }

    scheduler = ASHAScheduler(
        metric='loss',
        mode='min',
        max_t=10,
        grace_period=1,
        reduction_factor=2
    )

    trainable_with_gpu = tune.with_resources(
        partial(train, data_dir=args.data_dir),
        {'cpu': 8, 'gpu': 1}
    )

    tuner = tune.Tuner(
        trainable_with_gpu,
        param_space=config,
        tune_config=tune.TuneConfig(
            num_samples=10,
            scheduler=scheduler,
        )
    )

    results = tuner.fit()
    
    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")

    best_trained_model = NED()
    device='cuda:'+str(args.cuda_name)
    best_trained_model.to(device)

    best_checkpoint = result.get_best_checkpoint(trial=best_trial, metric="loss", mode="min")
    with best_checkpoint.as_directory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "rb") as fp:
            best_checkpoint_data = pickle.load(fp)

        best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])
        test_acc = test_accuracy(best_trained_model, data_dir)
        print("Best trial test set accuracy: {}".format(test_acc))

