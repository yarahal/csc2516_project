import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch_geometric.utils import from_smiles
from tqdm import tqdm

class DrugDiscovery:
    def __init__(self, model, zinc_path, scores, metadata, device):
        self.model = model
        self.zinc_path = zinc_path
        self.scores = scores
        self.metadata = metadata
        self.device = device
    
    def load_zinc(self):
        zinc_smiles = pd.read_csv(self.zinc_path)
        zinc_smiles_np = zinc_smiles['smiles'].to_numpy()
        self.zinc_smiles = zinc_smiles_np

    def drug_discovery(self, drug_id):
        cmap_scores = torch.tensor([])

        smiles = self.metadata[self.metadata['drug_id'] == drug_id]['smiles'].item()
        molecule = from_smiles(smiles).to(self.device)
        molecule.batch = torch.zeros(molecule.num_nodes,dtype=torch.int64,device=self.device)

        for drug in tqdm(self.zinc_smiles):
            zinc_molecule = from_smiles(drug).to(self.device)
            zinc_molecule.batch = torch.zeros(zinc_molecule.num_nodes,dtype=torch.int64,device=self.device)
            pred_cmap = self.model(molecule, zinc_molecule)
            cmap_scores = torch.cat((cmap_scores,pred_cmap.detach().cpu()))
            
        return cmap_scores

    def get_top_n(self,cmap_scores,n=10):
        combined = np.vstack((self.zinc_smiles, cmap_scores)).T
        combined_sorted = combined[combined[:,1].argsort()[::-1]]
        top_n = combined_sorted[:n]
        return top_n