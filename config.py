class Config:
    batch_size = 64
    lr = 1e-3
    momentum = 0.9
    weights_decay = 1e-4
    class_num = 11
    eval_interval = 1
    checkpoint_interval = 5
    print_interval = 50
    checkpoints = './checkpoints'   # 自己创建一个文件夹用来储存权重
    pretrained = None 
    start_epoch = 0
    epoches = 2
    smooth = 0.1
    erase_prob = 0.5

# 构建数据集路径索引
dataset_path = "./dataset"
data_dir = {
    'train_data': f'{dataset_path}/mchar_train/',
    'val_data': f'{dataset_path}/mchar_val/',
    'test_data': f'{dataset_path}/mchar_test_a/',
    'train_label': f'{dataset_path}/mchar_train.json',
    'val_label': f'{dataset_path}/mchar_val.json',
    'submit_file': f'{dataset_path}/mchar_sample_submit_A.csv'
}

config = Config()