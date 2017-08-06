import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 64
Z_dim = 100
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 128
cnt = 0
lr = 1e-3

class CVAE(nn.Module):
	def __init__(self):
		super(CVAE, self).__init__()
		
		self.fc1 = nn.Linear(794, 128)
		self.fc21 = nn.Linear(128, 100)
		self.fc22 = nn.Linear(128, 100)
		self.fc3 = nn.Linear(110, 128)
		self.fc4 = nn.Linear(128, 784)
	
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
	
		self.Z_dim = 100
		self.mb_size = 64

	def Q(self, X, c):
		inputs = torch.cat([X, c], 1)
		h1 = self.relu(self.fc1(inputs))
		return self.fc21(h1), self.fc22(h1)

	def sample_z(self, mu, log_var):
		eps = Variable(torch.randn(self.mb_size, self.Z_dim))
		return mu + torch.exp(log_var / 2) * eps

	def P(self, z, c):
		inputs = torch.cat([z, c], 1)
		h3 = self.relu(self.fc3(inputs))
		return self.sigmoid(self.fc4(h3))

	def forward(self, X, c):
		z_mu, z_logvar = self.Q(X, c)
		z = self.sample_z(z_mu, z_logvar)
		return self.P(z, c), z_mu, z_logvar

# =============================== TRAINING ====================================

model = CVAE()

recon_loss = nn.BCELoss()
recon_loss.size_average = False #Confused on this part. 

def loss_function(recon_x, x, mu, logvar, mb_size):
	marginal_liklihood = recon_loss(recon_x, x) / mb_size
	KLLoss = torch.mean(0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar, 1))
	return marginal_liklihood + KLLoss

solver = optim.Adam(model.parameters(), lr=lr)
model.train()

for it in range(100000):
    solver.zero_grad()
    #data
    X, c = mnist.train.next_batch(mb_size)

    X = Variable(torch.from_numpy(X))
    c = Variable(torch.from_numpy(c.astype('float32')))
    
    # Forward
    recon_data, mu, logVar = model(X, c)

    # Loss
    loss = loss_function(recon_data, X, mu, logVar, mb_size)

    # Backward
    loss.backward()
 
    # Update
    solver.step()

    # Print and plot every now and then
    if it % 1000 == 0:
        print('Iter-{}; Loss: {:.4}'.format(it, loss.data[0]))

        c = np.zeros(shape=[mb_size, y_dim], dtype='float32')
        c[:, np.random.randint(0, 10)] = 1.
	print(c)
        c = Variable(torch.from_numpy(c))
        z = Variable(torch.randn(mb_size, Z_dim))
        samples = model.P(z, c).data.numpy()[:16]

        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

        if not os.path.exists('outCVAE/'):
            os.makedirs('outCVAE/')

        plt.savefig('outCVAE/{}.png'.format(str(cnt).zfill(3)), bbox_inches='tight')
        cnt += 1
        plt.close(fig)
