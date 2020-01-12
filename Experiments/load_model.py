# %%
import torch 
from model import Net
MODEL_PATH = 'model_cifar10.pth'
OPTIM_PATH = 'optim_cifar10.pt'
FULL_PATH = 'mdoel_full_cifar10.pth'

# Saving model parameters
# torch.save(net.state_dict(),MODEL_PATH)
# torch.save(optimizer.state_dict(),OPTIM_PATH)
# torch.save(net.state_dict(),FULL_PATH )



# # Save entire model 
# torch.save(net, PATH)
#net = Net()
# Load model 
param_dict = torch.load(MODEL_PATH)
loaded_model = Net.load_state_dict(param_dict)
#loaded_optim = Net.load_state_dict(torch.load(OPTIM_PATH))

# %%
for param in loaded_model.state_dict():
    print(param,model.state_dict()[param].size())

for param in loaded_optim.state_dict():
    print(param,loaded_optim.state_dict()[param].size())
    