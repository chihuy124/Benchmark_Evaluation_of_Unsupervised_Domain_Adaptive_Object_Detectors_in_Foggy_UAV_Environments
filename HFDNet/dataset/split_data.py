import os, shutil, random
random.seed(0)
import numpy as np
from sklearn.model_selection import train_test_split

val_size = 0.1
test_size = 0.2
postfix = 'jpg'
imgpath = '/home/lenovo/data/liujiaji/Oildata/allimage'
txtpath = '/home/lenovo/data/liujiaji/Oildata/alllabel'

os.makedirs('/home/lenovo/data/liujiaji/Oildata/images/train', exist_ok=True)
os.makedirs('/home/lenovo/data/liujiaji/Oildata/images/val', exist_ok=True)
os.makedirs('/home/lenovo/data/liujiaji/Oildata/images/test', exist_ok=True)
os.makedirs('/home/lenovo/data/liujiaji/Oildata/labels/train', exist_ok=True)
os.makedirs('/home/lenovo/data/liujiaji/Oildata/labels/val', exist_ok=True)
os.makedirs('/home/lenovo/data/liujiaji/Oildata/labels/test', exist_ok=True)

listdir = np.array([i for i in os.listdir(txtpath) if 'txt' in i])
random.shuffle(listdir)
train, val, test = listdir[:int(len(listdir) * (1 - val_size - test_size))], listdir[int(len(listdir) * (1 - val_size - test_size)):int(len(listdir) * (1 - test_size))], listdir[int(len(listdir) * (1 - test_size)):]
print(f'train set size:{len(train)} val set size:{len(val)} test set size:{len(test)}')

for i in train:
    shutil.copy('{}/{}.{}'.format(imgpath, i[:-4], postfix), '/home/lenovo/data/liujiaji/Oildata/images/train/{}.{}'.format(i[:-4], postfix))
    shutil.copy('{}/{}'.format(txtpath, i), '/home/lenovo/data/liujiaji/Oildata/labels/train/{}'.format(i))

for i in val:
    shutil.copy('{}/{}.{}'.format(imgpath, i[:-4], postfix), '/home/lenovo/data/liujiaji/Oildata/images/val/{}.{}'.format(i[:-4], postfix))
    shutil.copy('{}/{}'.format(txtpath, i), '/home/lenovo/data/liujiaji/Oildata/labels/val/{}'.format(i))

for i in test:
    shutil.copy('{}/{}.{}'.format(imgpath, i[:-4], postfix), '/home/lenovo/data/liujiaji/Oildata/images/test/{}.{}'.format(i[:-4], postfix))
    shutil.copy('{}/{}'.format(txtpath, i), '/home/lenovo/data/liujiaji/Oildata/labels/test/{}'.format(i))