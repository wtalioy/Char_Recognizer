import torch as t
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import config
from model import DigitsResnet
from dataset import DigitsDataset
from utils import parse2class, write2csv

def predicts(model_path, csv_path):
    """
    Run prediction on test dataset and save results to CSV
    
    Args:
        model_path (str): Path to the trained model checkpoint
        csv_path (str): Path to save the prediction results
    
    Returns:
        list: Prediction results as list of [filename, predicted_code] pairs
    """
    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
    test_set = DigitsDataset(mode='test', aug=False)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False,
                    num_workers=16, pin_memory=True, drop_last=False, 
                    persistent_workers=True, collate_fn=test_set.collect_fn)
    results = []
    
    # Load model
    model = DigitsResnet(class_num=config.class_num).to(device)
    model.load_state_dict(t.load(model_path)['model'])
    print('Load model from %s successfully' % model_path)
    
    # Run inference
    tbar = tqdm(test_loader)
    model.eval()
    with t.no_grad():
        for i, (img, img_names) in enumerate(tbar):
            img = img.to(device)
            cls_preds, _ = model(img)
            results += [[name, code] for name, code in zip(img_names, parse2class(cls_preds))]
    
    # Sort results by filename
    results = sorted(results, key=lambda x: x[0])
    
    # Save results to CSV
    write2csv(results, csv_path)
    
    return results