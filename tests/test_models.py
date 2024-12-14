import unittest
import torch
from models.generator import Generator
from models.discriminator import Discriminator

class TestGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = Generator()
        self.batch_size = 32
        self.seq_len = 10
        self.height = 60
        self.width = 60
        self.channels = 3
    
    def test_forward_pass(self):
        """Test generator forward pass."""
        x = torch.randn(self.batch_size, self.seq_len, self.height, self.width, self.channels)
        output = self.generator(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 1))
        
        # Check output is valid (no NaN values)
        self.assertFalse(torch.isnan(output).any())

class TestDiscriminator(unittest.TestCase):
    def setUp(self):
        self.discriminator = Discriminator()
        self.batch_size = 32
    
    def test_forward_pass(self):
        """Test discriminator forward pass."""
        x = torch.randn(self.batch_size, 1, 11)  # 11 is sequence length
        output = self.discriminator(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 1))
        
        # Check output is valid (no NaN values)
        self.assertFalse(torch.isnan(output).any())
