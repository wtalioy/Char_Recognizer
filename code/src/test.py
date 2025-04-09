import torch as t
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import config
from model import digitsvit
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
    test_loader = DataLoader(DigitsDataset(mode='test', aug=False), batch_size=config.batch_size, shuffle=False,
                    num_workers=8, pin_memory=True, drop_last=False, persistent_workers=True)
    results = []
    
    # Load model
    model = digitsvit(config.prompt_num).to(device)
    model.load_state_dict(t.load(model_path)['model'])
    print('Load model from %s successfully' % model_path)
    
    # Run inference
    tbar = tqdm(test_loader)
    model.eval()
    with t.no_grad():
        for i, (img, img_names) in enumerate(tbar):
            img = img.to(device)
            pred = model(img)
            results += [[name, code] for name, code in zip(img_names, parse2class(pred))]
    
    # Sort results by filename
    results = sorted(results, key=lambda x: x[0])
    
    # Save results to CSV
    write2csv(results, csv_path)
    
    return results