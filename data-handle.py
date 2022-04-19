from fileinput import filename
import os
import time
import shutil


    
    #time.sleep(5)
def copy_image(sopath , depath, filelist):
    for i in range(len(filelist)):
        fname , suffixname = os.path.splitext(filelist[i])
        list = fname.split('.')
        if list[0] == 'cat':
            if int(list[1]) <= 9999:
                shutil.copyfile(os.path.join(sopath , filelist[i]) , os.path.join(depath,filelist[i]))
                #os.remove(os.path.join(depath , filelist[i]))
                
        if list[0] == 'doggggggggg':
            if int(list[1]) <= 999:
                shutil.copyfile(os.path.join(sopath , filelist[i]) , os.path.join(depath,filelist[i]))
                #os.remove(os.path.join(depath , filelist[i]))
def copy_liuxin(depath  ,path):
    a = 0
    for root , dirs , files in os.walk(path):#有文件夹形式的
        for f in files:
            a += 1
            sopath = os.path.join(root , f)
            
            shutil.copyfile(sopath , os.path.join(depath , str(a)+f))#############
def copy_others(sopath , depath, filelist):
    for i in range(len(filelist)):
        if i <= 99999:
            shutil.copyfile(os.path.join(sopath , filelist[i]) , os.path.join(depath , filelist[i]))
def deleshuzi(path):
    filelist = os.listdir(path)
    for i in range(len(filelist)):
        name , shu = os.path.splitext(filelist[i])
        if len(shu) > 4:
            #print(shu)
            os.remove(os.path.join(path , filelist[i]))
def copy_dog(sopath , depath, filelist , key):
    for i in range(len(filelist)):
        fname , suffixname = os.path.splitext(filelist[i])
        list = fname.split('.')
        if list[0] == 'cat':
            if int(list[1]) <= 999:
                shutil.copyfile(os.path.join(sopath , filelist[i]) , os.path.join(depath,filelist[i]))
                #os.remove(os.path.join(depath , filelist[i]))
                
        if list[0] == 'dog':
            if int(list[1])>=key-1000 and int(list[1]) < key:
                shutil.copyfile(os.path.join(sopath , filelist[i]) , os.path.join(depath,filelist[i]))
def copy_others(sopath_1 , sopath_2,depath, filelist_1  , filelsit_2, key):
    for i in range(len(filelist_1)):
        fname , suffixname = os.path.splitext(filelist_1[i])
        list = fname.split('.')
        if list[0] == 'cat':
            if int(list[1]) <= 999:
                shutil.copyfile(os.path.join(sopath_1 , filelist_1[i]) , os.path.join(depath,filelist_1[i]))
                #os.remove(os.path.join(depath , filelist[i]))

    for i in range(1000):
        shutil.copyfile(os.path.join(sopath_2 , filelist_2[key + i]) , os.path.join(depath,filelist_2[key + i]))

if __name__ == '__main__':
    path = '.\\dataset\\orignal_data'
    path1 = '.\\dataset\\orignal_data\\train'
    path2 = '.\\dataset\\validation'
    path3 = '.\\dataset\\train_qita_all'
    path4 = '.\\dataset\\train1000_1000_dog'
    path5 = '.\\dataset\\train1000_10000_dog'
    path6 = '.\\dataset\\train1000_1000_qita'
    path5 = '.\\dataset\\train1000_10000_qita'
    path7 = '.\\dataset\\train'
    path8 = '.\\dataset\\train1000_100000_qita'
    path9 = '.\\dataset\\train_cat'
    path10 = '.\\dataset\\train_cat_10000'
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
    filelist_1 = os.listdir(path1)
    #copy_image(path1 , path4,filelist)
    #copy_liuxin(path3, '.\\dataset\\train\\street')#street
    #copy_others(path3 , path8 , filelist)
    #deleshuzi(path3)
    #copy_image(path1 , path10,filelist)
    filelist_2 = os.listdir(path3)
    path_list = [path11 , path12 , path13 , path14,path15,path16,path17 , path18,path19,path20]
    for path_index , key in enumerate(range(0,10000,1000)):
        copy_others(path1 ,path3 ,path_list[path_index] , filelist_1 ,filelist_2, key)
        
    
