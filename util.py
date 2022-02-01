
class Average:
    """
    Keep running average of a series of observations
    """
    def __init__(self):
        self.n = self.sum = 0
    
    def update(self, num, n=1):
        self.n += n
        self.sum += num

    def get(self):
        return self.sum / self.n if self.n else 0