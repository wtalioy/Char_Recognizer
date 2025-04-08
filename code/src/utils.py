import pandas as pd
import os
import requests
import zipfile
import shutil
from glob import glob
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch as t
from config import data_dir, dataset_path

def download_dataset(csv_path="./mchar_data_list_0515.csv"):
    """
    Download and extract dataset based on links provided in CSV file
    """
    links = pd.read_csv(csv_path)
    print(f"数据集目录：{dataset_path}")
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    for i,link in enumerate(links['link']):
        file_name = links['file'][i]
        print(file_name, '\t', link)
        file_name = os.path.join(dataset_path, file_name)
        if not os.path.exists(file_name):
            response = requests.get(link, stream=True)
            with open(file_name, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
    
    # Extract zip files
    zip_list = ['mchar_train', 'mchar_test_a', 'mchar_val']
    for little_zip in zip_list:
        zip_name = os.path.join(dataset_path, little_zip)
        if not os.path.exists(zip_name):
            zip_file = zipfile.ZipFile(os.path.join(dataset_path, f"{little_zip}.zip"), 'r')
            zip_file.extractall(path=dataset_path)
    
    # Optionally remove Mac OS metadata folders
    if os.path.exists(os.path.join(dataset_path, '__MACOSX')):
        shutil.rmtree(os.path.join(dataset_path, '__MACOSX'))

def data_statistics():
    """
    Print dataset statistics
    """
    train_list = glob(data_dir['train_data'] + '*.png')
    test_list = glob(data_dir['test_data'] + '*.png')
    val_list = glob(data_dir['val_data'] + '*.png')
    print('Train image counts: %d' % len(train_list))
    print('Val image counts: %d' % len(val_list))
    print('Test image counts: %d' % len(test_list))

def look_train_json():
    """
    Display sample from training data JSON file
    """
    with open(data_dir['train_label'], 'r', encoding='utf-8') as f:
        content = f.read()

    content = json.loads(content)
    print(content['000000.png'])

def look_submit():
    """
    Display sample of submission file format
    """
    df = pd.read_csv(data_dir['submit_file'], sep=',')
    print(df.head(5))

def img_size_summary():
    """
    Analyze and visualize image sizes in the training dataset
    """
    sizes = []

    for img_path in glob(data_dir['train_data'] + '*.png'):
        img = Image.open(img_path)
        sizes.append(img.size)

    sizes = np.array(sizes)

    plt.figure(figsize=(10, 8))
    plt.scatter(sizes[:, 0], sizes[:, 1])
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.title('Image width-height summary')
    plt.show()

def bbox_summary():
    """
    Analyze and visualize bounding box dimensions in the training dataset
    """
    marks = json.loads(open(data_dir['train_label'], 'r').read())
    bboxes = []

    for img, mark in marks.items():
        for i in range(len(mark['label'])):
            bboxes.append([mark['left'][i], mark['top'][i], mark['width'][i], mark['height'][i]])

    bboxes = np.array(bboxes)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(bboxes[:, 2], bboxes[:, 3])
    ax.set_title('Bbox width-height summary')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    plt.show()

def label_summary():
    """
    Count number of digits in each image
    """
    marks = json.load(open(data_dir['train_label'], 'r'))
    dicts = {}
    for img, mark in marks.items():
        if len(mark['label']) not in dicts:
            dicts[len(mark['label'])] = 0
        dicts[len(mark['label'])] += 1

    dicts = sorted(dicts.items(), key=lambda x: x[0])
    for k, v in dicts:
        print('%d个数字的图片数目: %d' % (k, v))

def parse2class(prediction):
    """
    Convert model output prediction to class labels
    
    Params:
    prediction(tuple of tensor): model output for 4 digits
    """
    ch1, ch2, ch3, ch4 = prediction
    char_list = [str(i) for i in range(10)]
    char_list.append('')
    ch1, ch2, ch3, ch4 = ch1.argmax(1), ch2.argmax(1), ch3.argmax(1), ch4.argmax(1)
    ch1, ch2, ch3, ch4 = [char_list[i.item()] for i in ch1], [char_list[i.item()] for i in ch2], \
                    [char_list[i.item()] for i in ch3], [char_list[i.item()] for i in ch4] 
    res = [c1+c2+c3+c4 for c1, c2, c3, c4 in zip(ch1, ch2, ch3, ch4)]             
    return res

def write2csv(results, csv_path):
    """
    Write results to CSV file
    
    Args:
        results (list): List of [filename, predicted_code] pairs 
        csv_path (str): Path to save the CSV file
    """
    df = pd.DataFrame(results, columns=['file_name', 'file_code'])
    df['file_name'] = df['file_name'].apply(lambda x: x.split('/')[-1])
    save_name = csv_path
    df.to_csv(save_name, sep=',', index=None)
    print('Results saved to %s' % save_name)
