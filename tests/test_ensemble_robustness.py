

class TestEnsembleRobustness(unittest.TestCase):
    def setUp(self):
        self.ensemble = GAFEWGANEnsemble(n_models=5)
        self.test_data = self.generate_test_data()
    
    def generate_test_data(self):
        """Generate synthetic test data with various market conditions."""
        conditions = {
            'normal': np.random.normal(0, 0.01, 100),
            'trend': np.linspace(0, 0.2, 100),
            'volatile': np.random.normal(0, 0.03, 100),
            'crash': np.concatenate([
                np.random.normal(0, 0.01, 50),
                -0.05 * np.ones(20),
                np.random.normal(0, 0.02, 30)
            ])
        }
        return conditions
    
    def test_ensemble_disagreement(self):
        """Test ensemble model disagreement levels."""
        for condition, data in self.test_data.items():
            predictions = [model.predict(data) for model in self.ensemble.base_models]
            disagreement = np.std(predictions, axis=0)
            
            if condition == 'crash':
                # Higher uncertainty during crashes
                self.assertGreater(np.mean(disagreement), 0.02)
            elif condition == 'normal':
                # Lower uncertainty during normal periods
                self.assertLess(np.mean(disagreement), 0.01)
    
    def test_meta_learner_adaptation(self):
        """Test meta-learner adaptation to different market conditions."""
        for condition, data in self.test_data.items():
            base_predictions = [model.predict(data) for model in self.ensemble.base_models]
            meta_prediction = self.ensemble.meta_learner(base_predictions)
            
            # Calculate correlation with individual models
            correlations = [np.corrcoef(meta_prediction, pred)[0,1] 
                          for pred in base_predictions]
            
            if condition == 'crash':
                # Should rely more on conservative models
                self.assertLess(np.max(correlations), 0.9)
            else:
                # Can follow stronger signals in normal conditions
                self.assertGreater(np.max(correlations), 0.7)
