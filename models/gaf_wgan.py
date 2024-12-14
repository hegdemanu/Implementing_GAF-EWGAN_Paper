

import torch
import torch.nn as nn
import torch.autograd as autograd

class GAFWGAN:
    """Single GAF-WGAN implementation."""
    
    def __init__(self, generator, discriminator, device='cuda'):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        
        # Optimizers
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=1e-4, betas=(0.5, 0.9)
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9)
        )
        
    def gradient_penalty(self, real_samples, fake_samples):
        """Calculate gradient penalty for WGAN-GP."""
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1, 1).to(self.device)
        
        # Interpolate between real and fake samples
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.requires_grad_(True)
        
        # Calculate gradients of discriminator output w.r.t. interpolates
        d_interpolates = self.discriminator(interpolates)
        fake = torch.ones(batch_size, 1).to(self.device)
        
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Calculate gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    def train_step(self, real_data, real_price):
        """Single training step."""
        batch_size = real_data.size(0)
        
        # Train Discriminator
        self.d_optimizer.zero_grad()
        
        # Generate fake data
        z = torch.randn(batch_size, 100).to(self.device)  # Latent vector
        fake_data = self.generator(z)
        
        # Calculate discriminator outputs
        real_validity = self.discriminator(real_data)
        fake_validity = self.discriminator(fake_data.detach())
        
        # Gradient penalty
        gp = self.gradient_penalty(real_data, fake_data.detach())
        
        # Discriminator loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + 10.0 * gp
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train Generator
        if self.iteration % 5 == 0:  # Update generator less frequently
            self.g_optimizer.zero_grad()
            
            # Generate fake data
            fake_data = self.generator(z)
            fake_validity = self.discriminator(fake_data)
            
            # Generator loss
            g_loss = -torch.mean(fake_validity)
            g_loss.backward()
            self.g_optimizer.step()
            
            return d_loss.item(), g_loss.item()
        
        return d_loss.item(), 0.0
