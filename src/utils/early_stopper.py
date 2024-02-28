class EarlyStopper:
    def __init__(self, max_count=5, delta=0.2):
        """
        Constructor

        Parameters
        ----------
        max_count: int
            Number of epochs before stopping
        delta: double
            Delta
        """
        self.max_count = max_count
        self.counter = 0
        self.delta = delta
        self.min_loss = float('inf')

    def stop(self, loss):
        """
        Checks whether early stopping criterion is met

        Parameters
        ----------
        loss: double
            Current validation loss
        """
        # Compare the current loss with the minimum loss
        if loss < self.min_loss:
            # If new minimum is met => reset the counter and save the value
            self.min_loss = loss
            self.counter = 0
        elif loss - self.min_loss >= self.delta:
            # Greater than the minimum loss (within the delta margin) =>
            # increment the counter and check if the stop condition is met
            self.counter += 1
            if self.counter >= self.max_count:
                return True
        return False
