

import unittest
import numpy as np
from preprocessing.gaf import GAFConverter

class TestGAFConverter(unittest.TestCase):
    def setUp(self):
        self.converter = GAFConverter(size=60)
        self.sample_data = np.sin(np.linspace(0, 4*np.pi, 60))
    
    def test_scaling(self):
        """Test data scaling to [-1, 1] range."""
        scaled = self.converter._scale(self.sample_data)
        self.assertTrue(np.all(scaled >= -1))
        self.assertTrue(np.all(scaled <= 1))
    
    def test_polar_encoding(self):
        """Test polar encoding transformation."""
        scaled = self.converter._scale(self.sample_data)
        phi, r = self.converter._polar_encoding(scaled)
        
        # Test phi is in valid range [0, π]
        self.assertTrue(np.all(phi >= 0))
        self.assertTrue(np.all(phi <= np.pi))
        
        # Test r is properly normalized
        self.assertTrue(np.all(r >= 0))
        self.assertTrue(np.all(r <= 1))
    
    def test_transform(self):
        """Test complete GAF transformation."""
        gaf = self.converter.transform(self.sample_data)
        
        # Check output shape
        self.assertEqual(gaf.shape, (60, 60))
        
        # Check output values are in valid range [-1, 1]
        self.assertTrue(np.all(gaf >= -1))
        self.assertTrue(np.all(gaf <= 1))
