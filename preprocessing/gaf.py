# 

import numpy as np

class GAFConverter:
    """Converts time series data to Gramian Angular Field format."""
    
    def __init__(self, size=60):
        self.size = size
    
    def _scale(self, data):
        """Scale data to [-1, 1] range using min-max scaling."""
        return 2 * ((data - np.min(data)) / (np.max(data) - np.min(data))) - 1
    
    def _polar_encoding(self, scaled_data):
        """Convert scaled data to polar coordinates."""
        phi = np.arccos(scaled_data)
        r = np.array([i/self.size for i in range(len(scaled_data))])
        return phi, r
    
    def transform(self, data):
        """Transform time series to GAF matrix."""
        scaled_data = self._scale(data)
        phi, r = self_polar_encoding(scaled_data)
        
        # Calculate GAF matrix
        cos_phi = np.cos(phi)
        gaf = np.outer(cos_phi, cos_phi) - np.sqrt(1 - cos_phi**2).reshape(-1, 1) @ \
              np.sqrt(1 - cos_phi**2).reshape(1, -1)
              
        return gaf
