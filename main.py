import os
import numpy as np
import torch
import torch.utils.data
import pandas as pd
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
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
        self.imgs = sorted(os.listdir(os.path.join(root, "test")))
        self.path_to_data_file = data_file

    def __getitem__(self, idx):
        # Загрузка боксов и картинок
        img_path = os.path.join(self.root, "test", self.imgs[idx])
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


dataset_test = Dataset(root="",
                       data_file="data_test.csv",
                       transforms=get_transform(train=False))

torch.manual_seed(1)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)
loaded_model = get_model(num_classes=71)
loaded_model.load_state_dict(torch.load("pytorch object detection/model_3"))

ANS = []

for idx in range(len(dataset_test)):
    print(idx, "/", len(dataset_test))
    img, _ = dataset_test[idx]
    loaded_model.eval()
    with torch.no_grad():
        prediction = loaded_model([img])
    image = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    draw = ImageDraw.Draw(image)
    # draw groundtruth
    '''
    for elem in range(len(label_boxes)):
        draw.rectangle([(label_boxes[elem][0], label_boxes[elem][1]),
                        (label_boxes[elem][2], label_boxes[elem][3])],
                       outline="green", width=1)
    for element in range(len(prediction[0]["boxes"])):
        boxes = prediction[0]["boxes"][element].cpu().numpy()
        score = np.round(prediction[0]["scores"][element].cpu().numpy(),
                         decimals=4)
        if score > 0.1:
            draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])],
                           outline="red", width=1)
            draw.text((boxes[0], boxes[1]), text=str(score))
    image.show()
    '''
    ans = {}
    for l in range(len(prediction[0]["labels"].cpu().numpy())):
        if prediction[0]["labels"].cpu().numpy()[l] not in ans.keys():
            ans[prediction[0]["labels"].cpu().numpy()[l]] = []
        s = np.round(prediction[0]["scores"].cpu().numpy(),
                     decimals=4)[l]
        ans[prediction[0]["labels"].cpu().numpy()[l]].append(s)
    ans_1 = []
    for j in ans.keys():
        if max(ans[j]) > 0.685:
            ans_1.append(j)
    while len(ans_1) < 8:
        ans_1.append(0)
    ANS.append(ans_1)

print(ANS)
import pandas as pd

full_labels = pd.read_csv('test.csv')["id"]

with open("answer_RCNN_3_685.csv", mode="w", encoding='utf-8') as w_file:
    file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
    file_writer.writerow(["id", "sing1", "sing2", "sing3", "sing4", "sing5", "sing6", "sing7", "sing8"])
    for i in range(len(ANS)):
        file_writer.writerow(
            [full_labels[i], ANS[i][0], ANS[i][1], ANS[i][2], ANS[i][3], ANS[i][4], ANS[i][5], ANS[i][6], ANS[i][7]])

# Image.fromarray(drow_boxes('img_test_24.png')).show()
