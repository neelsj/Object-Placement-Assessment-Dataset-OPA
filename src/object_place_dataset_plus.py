import csv
import os

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as Ft
from PIL import Image
from torch.utils.data import Dataset

from config import opt

torch.random.manual_seed(1)

def aspect_pad(image, w_bg, h_bg):

    w = image.size[0]
    h = image.size[1]
    aspect_bg = w_bg/h_bg
    
    if (w > h):
        h = int(float(w)/aspect_bg + .5)
    else:
        w = int(float(h)*aspect_bg + .5)

    p_left = (w - image.size[0]) // 2
    p_top = (h - image.size[1]) // 2
    p_right = w - p_left - image.size[0]
    p_bottom = h - p_top - image.size[1]
    padding = (p_left, p_top, p_right, p_bottom)
    return Ft.pad(image, padding, 0, 'constant')

class ImageDatasetPlus(Dataset):
    def __init__(self, istrain=True):
        self.istrain = istrain
        with open(opt.train_data_path if istrain else opt.test_data_path, "r") as f:
            reader = csv.reader(f)
            reader = list(reader)
            reader = reader[1:]

        self.labels = []
        self.images_path = []
        self.mask_path = []
        self.scale = []  # foreground box
        self.target_box = []  # foreground box
        self.dic_name = []
        self.bg_id = []
        self.fg_id = []        
        for row in reader:
            label = int(row[-3])
            image_path = row[-2]
            mask_path = row[-1]
            scale = eval(row[3])
            target_box = eval(row[2])
            bg_id = eval(row[1])
            fg_id = eval(row[0])
            self.labels.append(label)
            self.images_path.append(os.path.join(opt.img_path, image_path))
            self.mask_path.append(os.path.join(opt.mask_path, mask_path))
            self.scale.append(scale)
            self.target_box.append(target_box)
            self.bg_id.append(bg_id)
            self.fg_id.append(fg_id)
            self.dic_name.append(image_path)

        self.img_transform = transforms.Compose([
            transforms.Resize((opt.img_size, opt.img_size)),
            transforms.ToTensor()
        ])

        self.transforms_flip = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1)
        ])

    def __getitem__(self, index):
        ## img & mask
        img = Image.open(self.images_path[index]).convert('RGB')
        w = img.width
        h = img.height
        img = self.img_transform(img)

        mask = Image.open(self.mask_path[index]).convert('L')  # gray
        mask = self.img_transform(mask)

        img_mask = torch.cat([img, mask], dim=0)

        bg_id = self.bg_id[index]
        fg_id = self.fg_id[index]

        background = Image.open("/mnt/e/Research/Images/new_OPA/background_all/%d.jpg" % bg_id)
        backgroundw = background.size[0]
        backgroundh = background.size[1]
        background = self.img_transform(background)  

        foreground = Image.open("/mnt/e/Research/Images/new_OPA/foreground_all/%d.jpg" % fg_id)            
        foreground = aspect_pad(foreground, backgroundw, backgroundh)
        foreground = self.img_transform(foreground)

        mask = Image.open("/mnt/e/Research/Images/new_OPA/foreground_all/mask_%d.jpg"% fg_id).convert('L') 
        mask = aspect_pad(mask, backgroundw, backgroundh)
        mask = self.img_transform(mask)

        foreground = torch.cat([foreground, mask], dim=0)
                 
        label = self.labels[index]
        target_box = self.target_box[index]
        x1, y1, bw, bh = target_box

        cx = (x1 + bw/2)/backgroundw
        cy = (y1 + bh/2)/backgroundh

        scale = self.scale[index]

        return img_mask, label, cx, cy, scale, foreground, background

    def __len__(self):
        return len(self.labels)


def get_train_dataloader():
    trainset = ImageDatasetPlus(istrain=True)
    print('Training images', len(trainset))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size,
                                               shuffle=True, num_workers=opt.num_workers,
                                               drop_last=True, pin_memory=True)
    return train_loader


def get_test_dataloader():
    testset = ImageDatasetPlus(istrain=False)
    print('Testing images', len(testset))
    test_loader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size * 2,
                                              shuffle=False, num_workers=opt.num_workers,
                                              drop_last=False, pin_memory=True)
    return test_loader


if __name__ == "__main__":
    train_loader = get_train_dataloader()
    for batch_index, (img, label, target_box) in enumerate(train_loader):
        print(img.shape, label.shape, target_box.shape)
        if batch_index > 10:
            break

    test_loader = get_test_dataloader()
    for batch_index, (img, label, target_box) in enumerate(test_loader):
        print(img.shape, label.shape, target_box.shape)
        if batch_index > 10:
            break
