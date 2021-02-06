import torch
import torchvision
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision import transforms

# Defining the dataset class
class MaskDataset(Dataset):
    def __init__(self, df, directory, transform=None):
        self.transform = transform
        self.df = df
        self.directory = directory

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_url = os.path.join(self.directory, 'cropped images', self.df.iloc[idx, 0])
        img = Image.open(img_url).convert('RGB')
        label = torch.tensor(int(self.df.iloc[idx, 1]))

        if self.transform:
            img = self.transform(img)
        return img, label

# Calculates Accuracy
def calc_acc(model, loader, lr, epoch, mode):
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for x,y in loader:
                x , y = x.to(device) , y.to(device)
                y_hat = model(x)
                _ , pred = y_hat.max(1)
                correct += (pred == y).sum()
                total += pred.shape[0]
        print(f'{mode} score with learning rate {lr} , epoch {epoch + 1} = {correct/total}')
        model.train()
        return correct/total

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# DataFrame contains images names and labels
dataset_df = pd.read_csv('labels.csv')
root_dir = os.getcwd()

transform = transforms.Compose([
    transforms.Resize((120, 120)),
    transforms.ToTensor(),
    transforms.RandomVerticalFlip(0.7),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# Splitting and loading the data
dataset_len = len(dataset_df)
train = dataset_len // (10/7)
val = dataset_len // (10/1.5) + 1
test = dataset_len // (10/1.5) + 1

dataset = MaskDataset(dataset_df, root_dir, transform)
train_split, val_split, test_split = torch.utils.data.random_split(dataset, [int(train), int(val), int(test)])

train_loader = DataLoader(dataset=train_split, batch_size=8, shuffle=True)
val_loader = DataLoader(dataset=val_split, batch_size=8, shuffle=True)
test_loader = DataLoader(dataset=test_split, batch_size=8, shuffle=True)



criterion = torch.nn.CrossEntropyLoss()
num_epochs = 20
lr = 0.0005
# Using transfer learning
model = torchvision.models.resnet50(pretrained=True)
for layer, param in model.named_parameters():

    if 'layer4' not in layer:
        param.requires_grad = False

model.fc = torch.nn.Sequential(torch.nn.Linear(2048, 512),
                                 torch.nn.Linear(512, 3)
                               )

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_acc = []
val_acc = []

# Train
for epoch in range(num_epochs):
    for img, label in train_loader:
        img, label = img.to(device), label.to(device)
        y_hat = model(img)
        loss = criterion(y_hat, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_score = calc_acc(model, train_loader, lr, epoch, mode='training')
    train_acc.append(train_score)

    # Val Accuracy per epoch
    val_score = calc_acc(model, val_loader, lr, epoch, mode='val')
    val_acc.append(val_score)


# Test
correct = 0
total = 0
with torch.no_grad():
    model.eval()
    for x,y in test_loader:
            x , y = x.to(device) , y.to(device)
            y_hat = model(x)
            _ , pred = y_hat.max(1)
            correct += (pred == y).sum()
            total += pred.shape[0]
    print(f'test score = {correct/total}')


# Saving the model
def save_checkpoint(state, filename='model_checkpoint.pth.tar'):
    print('Saving model')
    torch.save(state, filename)


checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
save_checkpoint(checkpoint)
