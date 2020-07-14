class AverageMeter(object):
    """Computes and stores the average value"""

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return str(self.avg)
