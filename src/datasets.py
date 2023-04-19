import pandas as pd
from torch_geometric.data import Dataset
from torch_geometric.utils import from_smiles
from sklearn.model_selection import train_test_split

class CMapDataset(Dataset):
    def __init__(self, smiles_data, cmap_scores):
        super().__init__()
        self.smiles_data = smiles_data
        self.cmap_scores = cmap_scores

    def len(self):
        return self.cmap_scores.shape[0]
    
    def get(self,idx):
        molecule_A_ID = self.cmap_scores.loc[idx,'id_a']
        molecule_B_ID = self.cmap_scores.loc[idx,'id_b']
        molecule_A = from_smiles(self.smiles_data.loc[molecule_A_ID,'smiles'])
        molecule_B = from_smiles(self.smiles_data.loc[molecule_B_ID,'smiles'])
        cmap = self.cmap_scores.loc[idx,'score']
        return molecule_A, molecule_B, cmap

def load_data():
    scores_path = "../data/labels.txt"
    metadata_path = "../data/dds_drug_smiles.csv"
    scores = pd.read_csv(scores_path, delimiter=" ", names=['id_a','id_b','score'])
    metadata = pd.read_csv(metadata_path, header=0, index_col=0)
    
    return scores, metadata
    
def prepare_datasets(scores, metadata, test_size,val_size,random_state):
    train_scores, test_scores = train_test_split(scores, test_size=test_size, random_state=random_state)

    val_size = (scores.shape[0] * val_size) / train_scores.shape[0]
    train_scores, val_scores = train_test_split(train_scores, test_size=val_size, random_state=random_state)
    
    train_dataset = CMapDataset(metadata, train_scores.reset_index())
    val_dataset = CMapDataset(metadata, val_scores.reset_index())
    test_dataset = CMapDataset(metadata, test_scores.reset_index())
    
    return train_dataset, val_dataset, test_dataset

