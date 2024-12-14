import torch
import torch.nn as nn
import numpy as np

class MetaLearner(nn.Module):
    """Meta-learner for ensemble model."""
    
    def __init__(self, input_size):
        super(MetaLearner, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(100, 1)
        )
    
    def forward(self, x):
        return self.network(x)

class GAFEWGANEnsemble:
    """Ensemble of GAF-WGAN models."""
    
    def __init__(self, n_models=10, device='cuda'):
        self.n_models = n_models
        self.device = device
        self.base_models = []
        
        # Initialize base models
        for _ in range(n_models):
            generator = Generator().to(device)
            discriminator = Discriminator().to(device)
            model = GAFWGAN(generator, discriminator, device)
            self.base_models.append(model)
        
        # Initialize meta-learner
        self.meta_learner = MetaLearner(n_models).to(device)
        self.meta_optimizer = torch.optim.Adam(self.meta_learner.parameters())
        
    def train_base_models(self, train_loader, epochs):
        """Train all base models."""
        for i, model in enumerate(self.base_models):
            print(f"Training base model {i+1}/{self.n_models}")
            trainer = GAFWGANTrainer(model, train_loader, None, self.device)
            trainer.train(epochs)
    
    def train_meta_learner(self, val_loader, epochs):
        """Train meta-learner on validation set."""
        for epoch in range(epochs):
            total_loss = 0
            
            for data, price in val_loader:
                data = data.to(self.device)
                price = price.to(self.device)
                
                # Get predictions from all base models
                base_preds = []
                for model in self.base_models:
                    pred = model.generator(data)
                    base_preds.append(pred)
                
                base_preds = torch.stack(base_preds, dim=1)
                ensemble_pred = self.meta_learner(base_preds)
                
                # Calculate loss and update
                loss = nn.MSELoss()(ensemble_pred, price)
                self.meta_optimizer.zero_grad()
                loss.backward()
                self.meta_optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Meta-learner epoch {epoch+1}/{epochs}, Loss: {total_loss/len(val_loader):.4f}")
    
    def predict(self, data):
        """Generate ensemble prediction."""
        base_preds = []
        for model in self.base_models:
            pred = model.generator(data)
            base_preds.append(pred)
        
        base_preds = torch.stack(base_preds, dim=1)
        return self.meta_learner(base_preds)
