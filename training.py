import torch
import os
import time
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary
import pickle

### my modules
import model
from data_utils import MP3Dataset


### Set parameters:
save_model = True
load_model = False
model_name = "model_state"


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

train_dir = "./train_data/"
train_paths = [f for f in os.listdir(train_dir) if f.endswith(".mp3")]
train_paths = [os.path.join(train_dir, x) for x in train_paths]

val_dir = "./val_data/"
val_paths = [f for f in os.listdir(val_dir) if f.endswith(".mp3")]
val_paths = [os.path.join(val_dir, x) for x in val_paths]


resample_rate = 8000
train_dataset = MP3Dataset(train_paths,resample_rate)
val_dataset = MP3Dataset(val_paths,resample_rate)
train = DataLoader(train_dataset, batch_size=64, shuffle = True)
val = DataLoader(val_dataset, batch_size = 32, shuffle = True)

input_size = (2,resample_rate*15) # MNIST images are 28x28 = 784
hidden_size = 400
latent_size = 20

if load_model:
    with open(f'{model_name}.pkl', 'wb') as model_state:
        vae = pickle.load(model_state)
else:
    vae = model.VAE(input_size, hidden_size, latent_size)

vae.to(device)

# Define the optimizer
optimizer = optim.Adam(vae.parameters(), lr=2e-5) # 1e-5 seems good
VAEloss = model.loss_function()


#input_size = resample_rate*15 # 15s per sample

summary(vae,input_size=[64,*input_size])


## Training Loop
print(device)
vae.train()

print(next(vae.parameters()).device)
vae.to(device)
print(next(vae.parameters()).device)

train_loss = []
recon_loss = []
KLD_loss = []

val_loss = []
val_recon_loss = []
val_KLD_loss = []

n_epochs = 1000
start_time = time.time()

try:
    for epoch in range(n_epochs):
        print(f"epoch: {epoch}")
        for batch in train:
            batch.to(device)
            decoded, mu, logvar = vae(batch)
            
            losses = VAEloss(decoded,batch,mu,logvar)
            #losses = VAEloss(decoded.to(device), batch.to(device), mu.to(device), logvar.to(device))

            loss = losses[0]
            
            train_loss.append(loss.detach().to('cpu'))
            recon_loss.append(losses[1])
            KLD_loss.append(losses[2])
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print("loss: ", loss.item())
        
        if epoch % 5 == 0:
            for batch in val:
                vae.eval()
                
                batch.to(device)
                decoded, mu, logvar = vae(batch)
                
                losses = VAEloss(decoded,batch,mu,logvar)
                loss = losses[0]
                
                val_loss.append(loss.detach().to('cpu'))
                val_recon_loss.append(losses[1])
                val_KLD_loss.append(losses[2])
                
                vae.train()
            
            
            print(f"### Val loss: {val_loss[int(epoch/5)]}")    
            print(f"### Time since start: {time.time() - start_time}\t Time per epoch: {(time.time()-start_time)/(epoch +1)}")
finally:
    if save_model:
        with open(f'{model_name}.pkl', 'wb') as model_state:
            pickle.dump(vae, model_state)