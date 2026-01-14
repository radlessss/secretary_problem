
class ThresholdAgent:
    def __init__(self, threshold):
        self.threshold = threshold
        self.max_obs = 0
    
    def reset(self):
        self.max_obs = 0

    def get_action(self, obs, environment, epsilon=0):
        step, _, current_quality = obs
        if self.threshold >= step:
            self.max_obs = max(self.max_obs, current_quality)
            return 0 
        elif current_quality >= self.max_obs:
            return 1
        else:
            return 0
    


# to do add threshold agent implementation