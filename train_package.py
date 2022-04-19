import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import ResNet34
from dataset import data_set

model = ResNet34()
model.weight_init()
data_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
])
Epoch = 25
batch_size = 32
lr = 0.001
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, 2, 0.5)
train_data = data_set("./dataset/train_cat_10000", data_transform, train=True)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
val_data = data_set("./dataset/cat", data_transform, train=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#torch.device('cpu') #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

def fit(model, loader, train=True):
    if train:
        torch.set_grad_enabled(True)
        model.train()
    else:
        torch.set_grad_enabled(False)
        model.eval()
    running_loss = 0.0
    acc = 0.0
    max_step = 0
    for img, label in tqdm(loader, leave=False):
        max_step += 1
        if train:
            optimizer.zero_grad()
        label_pred = model(img.to(device, torch.float))
        pred = label_pred.argmax(dim=1)

        acc += (pred.data.cpu() == label.data).sum()###########################
        loss = loss_func(label_pred, label.to(device, torch.long))
        running_loss += loss
        if train:
            loss.backward()
            optimizer.step()
    running_loss = running_loss / (max_step)
    avg_acc = float(acc.item()) / ((max_step) * batch_size)
    #print(avg_acc , max_step , batch_size)
    #exit(0)
    if train:
        scheduler.step()
    return running_loss, avg_acc
    
def train(name):
    train_loss_list = []
    val_loss_list = []
    train_accuracy_list = []
    val_accuracy_list = []
    for epoch in range(Epoch):
        train_loss, train_acc = fit(model, train_loader, train=True)
        val_loss, val_acc = fit(model, val_loader, train=False)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_accuracy_list.append(train_acc)
        val_accuracy_list.append(val_acc)
        print('Epoch', epoch + 1, '| train_loss: %.4f' % train_loss, '|train_acc:%.4f' % train_acc, '| validation_loss: %.4f' % val_loss, '|validation_acc:%.4f' % val_acc)
    torch.save(model.state_dict(), "./" + name + ".pth")
    return train_loss_list, train_accuracy_list, val_loss_list, val_accuracy_list

def drew(train_loss_list, train_acc_list, val_loss_list, val_acc_list , name):
    plt.figure(figsize = (14,7))
    plt.suptitle(name)
    plt.subplot(121)
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(range(Epoch), train_loss_list, label="train")
    plt.plot(range(Epoch), val_loss_list, label="validation")
    plt.grid(color="k", linestyle=":")
    plt.legend()
    plt.subplot(122)
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(ymax=1, ymin=0)
    plt.plot(range(Epoch), train_acc_list, label="train")
    plt.plot(range(Epoch), val_acc_list, label="validation")
    plt.grid(color="k", linestyle=":")
    plt.legend()
    plt.savefig("./result/cat_others/" + name + ".png", dpi=600)
    #plt.show()

if __name__ == "__main__":
    path11 = '.\\dataset\\traing_others\\1000_1000(1)'
    path12 = '.\\dataset\\traing_others\\1000_1000(2)'
    path13 = '.\\dataset\\traing_others\\1000_1000(3)'
    path14 = '.\\dataset\\traing_others\\1000_1000(4)'
    path15 = '.\\dataset\\traing_others\\1000_1000(5)'
    path16 = '.\\dataset\\traing_others\\1000_1000(6)'
    path17 = '.\\dataset\\traing_others\\1000_1000(7)'
    path18 = '.\\dataset\\traing_others\\1000_1000(8)'
    path19 = '.\\dataset\\traing_others\\1000_1000(9)'
    path20 = '.\\dataset\\traing_others\\1000_1000(10)'
    path_list = [path11 , path12 , path13 , path14,path15,path16,path17 , path18,path19,path20]
    for i in range(10):
        print('traing----------------------------------------------------{}'.format(i))
        name_pth = 'train({})'.format(i)
        name_train = path_list[i]
        #######config加载
        model = ResNet34()
        model.weight_init()
        data_transform = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
        ])
        Epoch = 25
        batch_size = 32
        lr = 0.001
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 2, 0.5)
        train_data = data_set(name_train, data_transform, train=True)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
        val_data = data_set("./dataset/cat", data_transform, train=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
        device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#torch.device('cpu') #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        t_loss, t_acc, v_loss, v_acc = train(name_pth)
        drew(t_loss, t_acc, v_loss, v_acc , name_pth)
