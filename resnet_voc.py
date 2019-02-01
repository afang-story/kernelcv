import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
from resnet_helpers import train_model, set_parameter_requires_grad
from voc_helpers.ptvoc import VOCClassification
from sklearn.metrics import average_precision_score

model_name = 'resnet'
num_epochs = 50
batch_size = 8*8
num_classes = 20
feature_extract = False
dim = 224

model_ft = models.resnet34(pretrained=True)
set_parameter_requires_grad(model_ft, feature_extract)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, num_classes)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = torchvision.transforms.Compose([torchvision.transforms.Resize(dim), torchvision.transforms.CenterCrop(dim), transforms.ToTensor(), normalize])

print("Initializing Datasets and Dataloaders...")
yr = '2012'
trainset = VOCClassification(root='./data', image_set='train', year=yr,
                                    download=True, transform=transform)
valset = VOCClassification(root='./data', image_set='val', year=yr,
                                   download=True, transform=transform)

image_datasets = {'train': trainset, 'val': valset}
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, num_workers=4) for x in ['train', 'val']}

# Send the model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.01, momentum=0.9)

# Setup the loss fxn
# criterion = nn.MultiLabelSoftMarginLoss()
criterion = nn.BCEWithLogitsLoss()

# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, device, num_epochs=num_epochs, is_inception=(model_name=="inception"))

# Save validation labels
labels = None
threshold = 0.5
model_ft.eval()
with torch.no_grad():
    for inputs, _ in dataloaders_dict['val']:
        data = inputs.to(device)
        outputs = model_ft(data)
        if labels is None:
            labels = torch.where(outputs.detach().cpu() > threshold, torch.tensor(1).cpu(), torch.tensor(0).cpu())
        else:
            new = torch.where(outputs.detach().cpu() > threshold, torch.tensor(1).cpu(), torch.tensor(0).cpu())
            labels = np.vstack((labels, new))
print(average_precision_score(labels, np.array(valset.labels), average='micro'))
np.savetxt('resnet34_labels.csv', labels, delimiter=',')
