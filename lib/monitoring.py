import numpy as np

class PrecisionMonitor:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.losses = []
        self.metrics = {
            'dice': [],
            'sensitivity': [],
            'specificity': []
        }

    def update(self, loss, metrics):
        self.losses.append(loss)
        for k, v in metrics.items():
            if k in self.metrics:
                self.metrics[k].append(v)

        if len(self.losses) > self.window_size:
            self.losses.pop(0)
            for k in self.metrics:
                self.metrics[k].pop(0)

    def check_stability(self):
        if len(self.losses) < self.window_size:
            return True

        loss_std = np.std(self.losses[-self.window_size:])
        loss_mean = np.mean(self.losses[-self.window_size:])
        loss_cv = loss_std / (loss_mean + 1e-8)

        metrics_stable = True
        for k, v in self.metrics.items():
            if len(v) >= self.window_size:
                metric_std = np.std(v[-self.window_size:])
                metric_mean = np.mean(v[-self.window_size:])
                metric_cv = metric_std / (metric_mean + 1e-8)
                if metric_cv > 0.1:
                    metrics_stable = False
                    break

        return loss_cv < 0.1 and metrics_stable

