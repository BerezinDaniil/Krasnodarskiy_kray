import os

import pandas
import pycocotools
import numpy as np
import torch
import torch.utils.data
import pandas as pd
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
from engine import train_one_epoch, evaluate
import utils
import transforms as T
from PIL import ImageDraw
import csv


def parse_one_annot(path_to_data_file, filename):
    data = pd.read_csv(path_to_data_file)
    boxes_array = data[data['filename'] == filename][['xmin', "ymin", 'xmax', 'ymax']].values
    return boxes_array


def parse_clases(path_to_data_file, filename):
    data = pd.read_csv(path_to_data_file)
    class_array = data[data['filename'] == filename][['class']].values
    ans = []
    for i in class_array:
        ans.append(i[0])
    return ans


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, data_file, transforms=None):
        self.root = ""
        self.transforms = transforms
        self.imgs = sorted(os.listdir(os.path.join(root, "train")))
        self.path_to_data_file = data_file

    def __getitem__(self, idx):
        # Загрузка боксов и картинок
        img_path = os.path.join(self.root, "train", self.imgs[idx])
        img = Image.open(img_path)
        box_list = parse_one_annot(self.path_to_data_file,
                                   self.imgs[idx])
        boxes = torch.as_tensor(box_list, dtype=torch.float32)
        num_objs = len(box_list)

        labels_list = parse_clases(self.path_to_data_file,
                                   self.imgs[idx])
        labels = torch.as_tensor(labels_list, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)


import cv2
from PIL import Image

full_labels = pd.read_csv("data.csv")


def drow_boxes(image_name):
    selected_value = full_labels[full_labels.filename == image_name]
    img = cv2.imread(f'train/{image_name}')
    for index, row in selected_value.iterrows():
        print(row['xmin'], row['ymin'], row['xmax'], row['ymax'])
        img = cv2.rectangle(img, (row['xmin'], row['ymin']), (row['xmax'], row['ymax']), (0, 255, 0), 1)
    return img


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


dataset = Dataset(root="",
                  data_file="data.csv",
                  transforms=get_transform(train=True))
dataset_test = Dataset(root="",
                       data_file="data.csv",
                       transforms=get_transform(train=False))

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-90])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-90:])

data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=0,
    collate_fn=utils.collate_fn)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)
print("We have: {} examples, {} are training and {} testing".format(len(indices), len(dataset), len(dataset_test)))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 71
model = get_model(num_classes)
model.to(device)
# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)
num_epochs = 3
for epoch in range(num_epochs):
    print(epoch)
    # train for one epoch
    train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq=5)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

#os.mkdir("pytorch object detection/")
torch.save(model.state_dict(), "pytorch object detection/model_10")

