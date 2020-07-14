class IsBest(object):
    def __init__(self):
        """
        This class check if a given value is the best so far
        """
        self.val = None

    def is_best(self, val) -> bool:
        """
        This function returns the status of the current value and update the best value.
        :param val: The current value
        :return: Return a boolean True if current value is the best else False
        """
        if self.val is None or (val > self.val):
            self.val = val
            print("Updating Best")
            return True
        else:
            return False
