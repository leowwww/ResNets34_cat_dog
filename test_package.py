from cv2 import minEnclosingCircle, split
import torch
import numpy
import pandas
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from model import ResNet34
from dataset import data_set

def model_init(parameters):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet34()
    parameters = torch.load(parameters, map_location=torch.device(device))
    model.load_state_dict(parameters)
    model.to(device)
    return model



'''device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet34()
parameters = torch.load('./train_cat.pth', map_location=torch.device(device))
model.load_state_dict(parameters)
model.to(device)'''
data_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
])
test_data = data_set("./dataset/test", data_transform, train=False)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
torch.set_grad_enabled(False)
#model.eval()
result_list = numpy.zeros([test_data.__len__(), 2], dtype=int)
step = 0

for img, name in tqdm(test_loader):

    model_0 = model_init('./train(0).pth')
    model_1 = model_init('./train(1).pth')
    model_2 = model_init('./train(2).pth')
    model_3 = model_init('./train(3).pth')
    model_4 = model_init('./train(4).pth')
    model_5 = model_init('./train(5).pth')
    model_6 = model_init('./train(6).pth')
    model_7 = model_init('./train(7).pth')
    model_8 = model_init('./train(8).pth')
    model_9 = model_init('./train(9).pth')
    model_0.eval()
    model_1.eval()
    model_2.eval()
    model_3.eval()
    model_4.eval()
    model_5.eval()
    model_6.eval()
    model_7.eval()
    model_8.eval()
    model_9.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pred_0= model_0(img.to(device, torch.float))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pred_1= model_1(img.to(device, torch.float))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pred_2= model_2(img.to(device, torch.float))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pred_3= model_3(img.to(device, torch.float))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pred_4= model_4(img.to(device, torch.float))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pred_5= model_5(img.to(device, torch.float))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pred_6= model_6(img.to(device, torch.float))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pred_7= model_7(img.to(device, torch.float))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pred_8= model_8(img.to(device, torch.float))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pred_9= model_9(img.to(device, torch.float))
    labe0 = int(pred_0.argmax(dim=1))
    labe1 = int(pred_1.argmax(dim=1))
    labe2 = int(pred_2.argmax(dim=1))
    labe3 = int(pred_3.argmax(dim=1))
    labe4 = int(pred_4.argmax(dim=1))
    labe5 = int(pred_5.argmax(dim=1))
    labe6 = int(pred_6.argmax(dim=1))
    labe7 = int(pred_7.argmax(dim=1))
    labe8 = int(pred_8.argmax(dim=1))
    labe9 = int(pred_9.argmax(dim=1))
    mi_list = [labe0 , labe1,labe2,labe3,labe4,labe5,labe6,labe7,labe8,labe9]
    #print(mi_list)
    count = mi_list.count(0)
    if count >0:
        label = 0
    else:
        label = 1
    if name[0].split('.')[0] == 'cat':
        name = 0
    else:
        name = 1
    result_list[step, 0] = name
    result_list[step, 1] = label
    step += 1

result_list = result_list[result_list[:, 0].argsort()]
'''header = ["id", "label"]
csv_data = pandas.DataFrame(columns=header, data=result_list)
csv_data.to_csv("./result/submission(cat).csv", encoding='utf-8', index=False)'''
count = 0.0
print(result_list[0][0] , result_list[0][1])
for i in range(len(result_list)):
    if result_list[i][1] == result_list[i][0]:
        count+=1
print(count , len(result_list))
print('测试结果为{}'.format(count / len(result_list)))
