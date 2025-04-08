import argparse
from utils import download_dataset, data_statistics, look_train_json, look_submit, img_size_summary, bbox_summary, label_summary
from train import Trainer
from test import predicts

def parse_args():
    parser = argparse.ArgumentParser(description='Character Recognition')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'analyze'],
                        help='Run mode: train, test, or analyze data')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to the model checkpoint for testing')
    parser.add_argument('--csv-path', type=str, default='../tc_data/result.csv',
                        help='Path to save prediction results')
    parser.add_argument('--download', action='store_true',
                        help='Whether to download dataset')
    return parser.parse_args()

def main():
    """
    Main entry point
    """
    args = parse_args()
    
    # Download dataset if requested
    if args.download:
        download_dataset()
    
    if args.mode == 'train':
        # Train model
        trainer = Trainer()
        trainer.train()
        # Run prediction using the best model
        if trainer.best_checkpoint_path:
            print(f"Running prediction with best model: {trainer.best_checkpoint_path}")
            predicts(trainer.best_checkpoint_path, args.csv_path)
        
    elif args.mode == 'test':
        # Run prediction with specified model
        if args.model_path is None:
            print("Error: Must provide --model-path for test mode")
            return
        predicts(args.model_path, args.csv_path)
        
    elif args.mode == 'analyze':
        # Run data analysis functions
        data_statistics()
        look_train_json()
        look_submit()
        img_size_summary()
        bbox_summary()
        label_summary()

if __name__ == '__main__':
    main()