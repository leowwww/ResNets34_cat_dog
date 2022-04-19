from distutils import filelist
import xml.sax
from xml.dom  import minidom
import os
import shutil
import time
import pandas as pd
import shutil

def xml_read(path_xml):
    xml_list = os.listdir(path_xml)
    cat = []
    print(len(xml_list))
    for i in range(len(xml_list)):
        try:
            dom = minidom.parse(os.path.join(path_xml , xml_list[i]))#创建dom对象
        except:
            continue
        root = dom.documentElement#获取根节点
        #print(root.nodeName)
        name_list = root.getElementsByTagName('name')#找到所有‘name'的内容
        for j in range(1,len(name_list)):
            if name_list[j].firstChild.data == 'cat':
                name , spl = os.path.splitext(xml_list[i])
                cat.append(name)
                continue
            #print(name_list[j].firstChild.data)
    return cat
def copy_cat_image (depath,filelist):
    for i in range(len(filelist)):
        sopath = os.path.join(image_path , filelist[i]+'.jpg')
        try:
            shutil.copy(sopath,depath)
        except:
            continue
def copy_qita_image(depath ,cat_list):
    filelist = os.listdir('F:\\数据集\\VOC07+12+test\\VOCdevkit\\VOC2007\\JPEGImages')
    for i in range(len(filelist)):
        sopath = os.path.join('F:\\数据集\\VOC07+12+test\\VOCdevkit\\VOC2007\\JPEGImages',filelist[i])
        name , p = os.path.splitext(filelist[i])
        if name in cat_list:
            os.remove(sopath)
        else:
            shutil.copy(sopath , depath)

def delet_image(path , filelist , other_image_path):
    '''a = 0
    for i in range(len(filelist)):
        name , shu = os.path.splitext(filelist[i])
        if 'cat' in name:
            a += 0
            shutil.copyfile(os.path.join(path , filelist[i]) , os.path.join('C:\\魏巍\\魏巍\\ResNet-master\\dataset\\validation(1)', filelist[i]) )'''
    for i in range(len(other_image_path)):
        if 1000<i<3500:
            shutil.copyfile(os.path.join('C:\\魏巍\\魏巍\\ResNet-master\\dataset\\train_qita_all' , other_image_path[i]) ,os.path.join('C:\\魏巍\\魏巍\\ResNet-master\\dataset\\validation(1)', other_image_path[i]) )
if __name__ == '__main__':
    cat_path = 'F:\\迅雷下载\\ResNet-master\\ResNet-master\\数据集\\cat'
    path = 'F:\\迅雷下载\\ResNet-master\\ResNet-master\\数据集\\train_qita_all'
    path_xml = 'F:\\数据集\\VOC07+12+test\\VOCdevkit\\VOC2007\\Annotations'
    image_path = 'F:\\数据集\\VOC07+12+test\\VOCdevkit\\VOC2007\\JPEGImages'
    path1 = 'C:\\魏巍\\魏巍\\ResNet-master\\dataset\\validation'
    #cat = xml_read(path_xml)
    #copy_cat_image(cat_path , cat)
    #copy_qita_image(path , cat)
 
    delet_image(path1 , os.listdir(path1) , os.listdir('C:\\魏巍\\魏巍\\ResNet-master\\dataset\\train_qita_all'))



