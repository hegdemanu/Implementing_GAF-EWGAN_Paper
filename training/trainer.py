

class GAFWGANTrainer:
    """Trainer for GAF-WGAN model."""
    
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.generator.train()
        self.model.discriminator.train()
        
        total_d_loss = 0
        total_g_loss = 0
        
        for batch_idx, (data, price) in enumerate(self.train_loader):
            data = data.to(self.device)
            price = price.to(self.device)
            
            d_loss, g_loss = self.model.train_step(data, price)
            total_d_loss += d_loss
            total_g_loss += g_loss
            
        return total_d_loss / len(self.train_loader), total_g_loss / len(self.train_loader)
    
    def validate(self):
        """Validate model."""
        self.model.generator.eval()
        self.model.discriminator.eval()
        
        total_loss = 0
        
        with torch.no_grad():
            for data, price in self.val_loader:
                data = data.to(self.device)
                price = price.to(self.device)
                
                pred_price = self.model.generator(data)
                loss = nn.MSELoss()(pred_price, price)
                total_loss += loss.item()
                
        return total_loss / len(self.val_loader)
    
    def train(self, epochs):
        """Full training loop."""
        for epoch in range(epochs):
            d_loss, g_loss = self.train_epoch()
            val_loss = self.validate()
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"D_loss: {d_loss:.4f}, G_loss: {g_loss:.4f}, Val_loss: {val_loss:.4f}")
