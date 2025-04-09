from torchvision import transforms
class Config:
    batch_size = 32
    lr = 5e-3
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

dataset_path = "../tc_data"
data_dir = {
    'train_data': f'{dataset_path}/mchar_train/',
    'val_data': f'{dataset_path}/mchar_val/',
    'test_data': f'{dataset_path}/mchar_test_a/',
    'train_label': f'{dataset_path}/mchar_train.json',
    'val_label': f'{dataset_path}/mchar_val.json',
    'submit_file': f'{dataset_path}/mchar_sample_submit_A.csv'
}

train_transform = transforms.Compose([
    transforms.Resize(232),
    transforms.CenterCrop(224),
    transforms.RandomRotation(45),
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.RandomGrayscale(0.1),
    transforms.RandomAffine(15, translate=(0.05, 0.1), shear=5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform = {
    'train': train_transform,
    'val': test_transform,
    'test': test_transform
}

config = Config()