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
batch_size = 2*2*2*8*8
num_classes = 20
feature_extract = False
dim = 224
threshold = 0.3

model_ft = models.resnet50(pretrained=True)
set_parameter_requires_grad(model_ft, feature_extract)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, num_classes)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
'''
train_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(dim),
						  torchvision.transforms.CenterCrop(dim),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  normalize])
'''
train_transform = torchvision.transforms.Compose([transforms.RandomResizedCrop(dim), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
val_transform = torchvision.transforms.Compose([torchvision.transforms.Resize(dim), torchvision.transforms.CenterCrop(dim), transforms.ToTensor(), normalize])

print("Initializing Datasets and Dataloaders...")
yr = '2012'
trainset = VOCClassification(root='./data', image_set='train', year=yr,
                                    download=True, transform=train_transform)
valset = VOCClassification(root='./data', image_set='val', year=yr,
                                   download=True, transform=val_transform)

image_datasets = {'train': trainset, 'val': valset}
dataloaders_dict = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4),
                    'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=4)}

# Send the model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
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
ct = 0
for name, child in model_ft.named_children():
    if ct < 7:
        for name2, params in child.named_parameters():
            params.requires_grad = False
    ct += 1

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.1, momentum=0.9, weight_decay=1e-4)
# optimizer_ft = optim.Adam(params_to_update, lr=0.0005)

# Setup the loss fxn
# criterion = nn.MultiLabelSoftMarginLoss()
criterion = nn.BCEWithLogitsLoss()

# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, device, num_epochs=num_epochs, threshold=threshold, is_inception=(model_name=="inception"))

# Save validation labels
preds = None
all_labels = None
model_ft.eval()
with torch.no_grad():
    for inputs, labels in dataloaders_dict['val']:
        data = inputs.to(device)
        outputs = model_ft(data)
        if preds is None:
            preds = torch.where(outputs.detach().cpu() > threshold, torch.tensor(1).cpu(), torch.tensor(0).cpu())
            all_labels = labels.data.cpu().numpy()
        else:
            new = torch.where(outputs.detach().cpu() > threshold, torch.tensor(1).cpu(), torch.tensor(0).cpu())
            preds = np.vstack((preds, new))
            all_labels = np.vstack((all_labels, labels.data.cpu().numpy()))
print(average_precision_score(all_labels, preds, average='micro'))
np.savetxt('resnet50_labels.csv', preds, delimiter=',')
