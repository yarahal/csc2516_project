import argparse
from torch_geometric.loader import DataLoader
import torch
from torch import nn
from datasets import load_data, prepare_datasets
from model import SiameseNetwork, GAT
from trainer import Trainer
from metrics import Metrics
from drug_discovery import DrugDiscovery
import pickle

def parse_args(mode):
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--n_heads', type=int, default=32, help='number of attention heads')
    parser.add_argument('--output_size', type=int, default=32, help='output size')
    parser.add_argument('--embedding_size', type=int, default=1024, help='embedding size')
    parser.add_argument('--radius', type=int, default=3, help='molecular fingerprint radius')
    parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--patience', type=int, default=5, help='patience for early stopping')
    parser.add_argument('--step_size', type=int, default=30, help='step size for lr scheduler')
        
    if mode == 'evaluate':
        parser.add_argument('--model_path', type=str, default='../models/gat_model.pth', help='path to saved model weights')
        parser.add_argument('--pos_threshold', type=float, default=0.75, help='positive threshold')
        parser.add_argument('--neg_threshold', type=float, default=0.0, help='negative threshold')
    
    elif mode == 'discovery':
        parser.add_argument('--drug_id', type=str, help='drug id for drug discovery')
    
    args = parser.parse_args()

    return args

def create_config(args):
    config = {
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'n_heads': args.n_heads,
        'output_size': args.output_size,
        'embedding_size': args.embedding_size,
        'radius': args.radius,
        'n_epochs' : args.n_epochs,
        'patience': args.patience,
        'step_size': args.step_size
    }

    return config

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'discovery'], help='mode of operation')
    mode = parser.parse_args().mode
    
    args = parse_args(mode)
    config = create_config(args)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    scores, metadata = load_data()
    train_dataset, val_dataset, test_dataset = prepare_datasets(scores,metadata,0.2,0.1,20)
    train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, config['batch_size'],shuffle=False)
    test_loader = DataLoader(test_dataset, config['batch_size'], shuffle=False)

    gat_model = SiameseNetwork(GAT,config).to(device)

    if mode == 'train':
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(gat_model.parameters(), lr=config['learning_rate'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,config['step_size'])

        print('Training with a GAT backend')
        print(config)

        trainer = Trainer(gat_model, loss_function, optimizer, scheduler, config, device)
        loss_log = trainer.train(train_loader,val_loader)

        torch.save(gat_model.state_dict(), '../models/gat_model.pth')
        filename = '../models/gat_model_loss_log.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(loss_log, f)

    elif mode == 'evaluate':
        gat_model.load_model(args.model_path)

        metrics = Metrics(gat_model, config, device)
        y_hat, y = metrics.predict(test_loader)

        pos_threshold = args.pos_threshold
        neg_threshold = args.neg_threshold

        metrics_dict = metrics.evaluate_metrics(y, y_hat, pos_threshold,neg_threshold)
        metrics.print_metrics(metrics_dict)

        for k in [1,2,5]:
            metrics_top_kpct = metrics.evaluate_k_metrics(y, y_hat, pos_threshold, neg_threshold, k)
            print(f'@ {k}%')
            metrics.print_metrics(metrics_top_kpct)
            print()

    elif mode == 'discovery':
        zinc_path = "../data/in-trials.csv"
        drug_id = args.drug_id

        drug_discovery = DrugDiscovery(gat_model, zinc_path, scores, metadata, device)
        drug_discovery.load_zinc()

        cmap_scores = drug_discovery.drug_discovery(drug_id)
        top_n = drug_discovery.get_top_n(cmap_scores)

        for row in top_n:
            print(row[0], f'{row[1]:4f}')
    
if __name__ == '__main__':
    main()