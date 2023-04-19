import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from scipy.stats import pearsonr
from utils import discretize

class Metrics:
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def predict(self, test_loader):
        y_hat = torch.tensor([])
        y = torch.tensor([])
        
        self.model.eval()

        for i, data in enumerate(test_loader):
            molecule_A, molecule_B, target_cmap = data[0].to(self.device), data[1].to(self.device), data[2].float().to(self.device)
            
            # pass through model
            pred_cmap = self.model(molecule_A, molecule_B)

            y_hat = torch.cat((y_hat,pred_cmap.detach().cpu()))
            y = torch.cat((y,target_cmap.detach().cpu()))

            if i % 50 == 0:
                print(f'Batch {i} done')

        return y_hat.numpy(), y.numpy()
    
    def evaluate_metrics(self, y_true, y_pred, pos_threshold=0.75, neg_threshold=0.0):
        metrics = {}

        # continuous metrics
        metrics['mse'] = mean_squared_error(y_true,y_pred)
        metrics['corr'] = pearsonr(y_pred, y_true)[0]
        metrics['r2'] = r2_score(y_true,y_pred)

        # discrete metrics
        y_dis = discretize(y_true,0.9,-0.9)
        y_pred_dis = discretize(y_pred,pos_threshold,neg_threshold)  
        metrics['precision'] = precision_score(y_dis,y_pred_dis)
        metrics['recall'] = recall_score(y_dis,y_pred_dis)
        metrics['f1_score'] = f1_score(y_dis,y_pred_dis)
        metrics['accuracy'] = accuracy_score(y_dis,y_pred_dis)
        metrics['auc'] = roc_auc_score(y_dis,y_pred_dis)
        return metrics
    
    def evaluate_k_metrics(self, y_true, y_pred, k, pos_threshold=0.75, neg_threshold=0.0):
        combined = np.vstack((y_true, y_pred)).T
        combined_sorted = combined[combined[:,1].argsort()[::-1]]
        idx_top_kpct = round(k/100.0 * y_true.shape[0])
        top_kpct = combined_sorted[:idx_top_kpct]
        y_true_kpct, y_pred_kpct = top_kpct[:,0], top_kpct[:,1]

        metrics = {}
        # continuous metrics
        metrics['mse'] = mean_squared_error(y_true_kpct,y_pred_kpct)

        # discrete metrics
        y_dis = discretize(y_true_kpct,0.9,-0.9)
        y_pred_dis = discretize(y_pred_kpct,pos_threshold,neg_threshold)  
        metrics['precision'] = precision_score(y_dis,y_pred_dis)
        metrics['recall'] = recall_score(y_dis,y_pred_dis)
        metrics['f1_score'] = f1_score(y_dis,y_pred_dis)
        metrics['accuracy'] = accuracy_score(y_dis,y_pred_dis)

        return metrics
    
    def print_metrics(self, metrics):
        for metric in metrics:
            print(f'{metric} = {metrics[metric]:.4f}')
    
