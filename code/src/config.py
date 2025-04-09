class Config:
    batch_size = 32
    lr = 1e-4
    momentum = 0.9
    weights_decay = 1e-4
    class_num = 11
    eval_interval = 1
    checkpoint_interval = 5
    print_interval = 50
    checkpoints = '../user_data'
    pretrained = None 
    start_epoch = 0
    epoches = 10
    smooth = 0.1
    erase_prob = 0.5
    prompt_num = 3

dataset_path = "../tc_data"
data_dir = {
    'train_data': f'{dataset_path}/mchar_train/',
    'val_data': f'{dataset_path}/mchar_val/',
    'test_data': f'{dataset_path}/mchar_test_a/',
    'train_label': f'{dataset_path}/mchar_train.json',
    'val_label': f'{dataset_path}/mchar_val.json',
    'submit_file': f'{dataset_path}/mchar_sample_submit_A.csv'
}

config = Config()