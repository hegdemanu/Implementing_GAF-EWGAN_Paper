
class MarketStateTracker:
    def __init__(self):
        self.states = {}
        
    def update_state(self, symbol: str, data: dict):
        if symbol not in self.states:
            self.states[symbol] = {}
        self.states[symbol].update(data)
    
    def get_state(self, symbol: str) -> dict:
        return self.states.get(symbol, {})
