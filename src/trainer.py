
import torch
import time

class Trainer:
    def __init__(self, model, loss_function, optimizer, scheduler, config, device):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
    
    def train_loop(self, train_loader, val_loader):
        train_loss = 0
        val_loss = 0
        n = 0
        self.model.train()

        for _, data in enumerate(train_loader):
            molecule_A, molecule_B, target_cmap = data[0].to(self.device), data[1].to(self.device), data[2].float().to(self.device)
            
            # pass through model
            pred_cmap = self.model(molecule_A, molecule_B)

            # compute MSE between predicted and target
            loss = self.loss_function(pred_cmap, target_cmap)
            batch_size = molecule_A.batch.unique().size(0)
            train_loss += batch_size * loss.item()
            n += batch_size

            # backpropagate and step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        train_loss /= n
        val_loss = self.evaluate_val_loss(val_loader)
        self.scheduler.step()

        return train_loss, val_loss
    
    def evaluate_val_loss(self,val_loader):
        val_loss = 0
        n = 0
        self.model.eval()

        with torch.no_grad():
            for _, data in enumerate(val_loader):
                molecule_A, molecule_B, target_cmap = data[0].to(self.device), data[1].to(self.device), data[2].float().to(self.device)
                
                # pass through model
                pred_cmap = self.model(molecule_A, molecule_B)
                
                # compute MSE between predicted and target
                loss = self.loss_function(pred_cmap, target_cmap)
                val_loss += self.config['batch_size'] * loss.item()
                n += self.config['batch_size']

        val_loss /= n
        return val_loss
    
    def train(self, train_loader, val_loader):
        n_epochs = self.config['n_epochs']
        loss_log = {'train':[], 'val':[]}

        early_stopping_counter = 0
        best_val_loss = float('inf')

        for i in range(n_epochs):
            start_time = time.time()
            train_loss, val_loss = self.train_loop(train_loader, val_loader)

            duration = time.time() - start_time
            loss_log['train'].append(train_loss)
            loss_log['val'].append(val_loss)

            print(f'Epoch {i+1}/{n_epochs}, train_loss = {train_loss:.4f}, val_loss = {val_loss:.4f} ------- {duration / 60 :.4f} minutes')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= self.config['patience']:
                    print(f'Early stopping after epoch {i+1}')
                    break

        return loss_log