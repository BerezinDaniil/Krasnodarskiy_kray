import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import csv

arr = ['3.24', '1.16', '5.15.5', '5.19.1', '5.19.2', '1.20.1', '8.23',
       '2.1', '4.2.1', '8.22.1', '6.16', '1.22', '1.2', '5.16', '3.27',
       '6.10.1', '8.2.4', '6.12', '5.15.2', '3.13', '3.1', '3.20', '3.12',
       '7.14.2', '5.23.1', '2.4', '5.6', '4.2.3', '8.22.3', '5.15.1',
       '7.3', '3', '2.3.1', '3.11', '6.13', '5.15.4', '8.2.1', '1.34.3',
       '8.2.2', '5.15.3', '1.17', '4.1.1', '4.1.4', '3.25', '1.20.2',
       '8.22.2', '6.9.2', '3.2', '5.5', '5.15.7', '7.12', '8.2.3',
       '5.24.1', '1.25', '3.28', '5.9.1', '5.15.6', '8.1.1', '1.10',
       '6.11', '3.4', '6.10', '6.9.1', '8.2.5', '5.15', '4.8.2', '8.22',
       '5.21', '5.18']


def index(l):
    if l == "6.22":
        l = "8.23"
    for i in range(len(arr)):
        if l == arr[i]:
            return i + 1


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     index(str(member[0].text)),
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    image_path = os.path.join(os.getcwd(), 'train_1')
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv('data.csv')
    print('Successfully converted xml to csv.')


main()

full_labels = pd.read_csv('test.csv')["img"]
xml_list = []
for file in full_labels:
    value = (file,
             int(0),
             int(0),
             str(0),
             int(0),
             int(0),
             int(0),
             int(0)
             )
    xml_list.append(value)
column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
xml_df = pd.DataFrame(xml_list, columns=column_name)
xml_df.to_csv('data_test.csv')
