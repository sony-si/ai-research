class PartoBest(object):
    def __init__(self):
        """
        This class check if a given values are Pareto best.
        """
        self.val_a = None
        self.val_b = None

    def _is_best(self, val_a, val_b):
        return self.val_a < val_a and self.val_b < val_b

    def _update_best(self, val_a, val_b):
        self.val_a = val_a
        self.val_b = val_b

    def is_pareto_best(self, val_a, val_b):
        """
        This function returns the status (checking if a and b are Pareto best) of the current values and
        update the best values.
        :param val_a: The current a value
        :param val_b: The current b value
        :return: Return a boolean True if current values are Pareto best else False
        """
        if (self.val_a is None and self.val_b is None) or self._is_best(val_a, val_b):
            self._update_best(val_a, val_b)
            print("Updating Pareto Best")
            return True
        else:
            return False
