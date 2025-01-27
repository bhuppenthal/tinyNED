from torch import nn
from torch.utils.data import Dataset
import h5py

class EntDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.feat_dim = 228

        hf = h5py.File(self.data_path, 'r')
        self.len = hf['dataset'].shape[0]
        hf.close()

    def __getitem__(self, idx):
        hf = h5py.File(self.data_path, 'r')
        X = hf['dataset'][idx, 0:self.feat_dim]
        Y = hf['dataset'][idx, self.feat_dim:2*self.feat_dim]
        hf.close()
        return (X, Y)

    def __len__(self):
        return self.len

class NED(nn.Module):
    def __init__(self, feat_dim=228, int_dim=128, latent_dim=30):
        super(NED, self).__init__()

        self.feat_dim = feat_dim

        self.fc1 = nn.Linear(feat_dim, int_dim)
        self.bn1 = nn.BatchNorm1d(int_dim)
        self.fc2 = nn.Linear(int_dim, latent_dim)
        self.bn2 = nn.BatchNorm1d(latent_dim)

        self.fc3 = nn.Linear(latent_dim, int_dim)
        self.bn3 = nn.BatchNorm1d(int_dim)
        self.fc4 = nn.Linear(int_dim, feat_dim)
        self.bn4 = nn.BatchNorm1d(feat_dim)

        self.relu = nn.ReLU()

    def encode(self, x):
        h1 = self.fc1(x)
        h1 = self.relu(self.bn1(h1))

        h2 = self.fc2(h1)
        z = self.bn2(h2)

        return z

    def decode(self, z):
        h3 = self.fc3(z)
        h3 = self.relu(self.bn3(h3))

        h4 = self.fc4(h3)
        recon = self.bn4(h4)

        return recon

    def forward(self, x):
        z = self.encode(x.view(-1, self.feat_dim))
        return self.decode(z)

    def embedding(self, x):
        return self.encode(x.view(-1, self.feat_dim))
