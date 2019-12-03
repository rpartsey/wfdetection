import numpy as np


class EarlyStopping_:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, mode='min', patience=7):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode


    # 1 2 3
    # 3 2 1
    # Saved -3
    #

    def __call__(self, metric):
        score = metric
        if self.mode == 'max':
            score = -score
        if self.best_score is None:
            self.best_score = score
        elif metric >= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss):
        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.message(val_loss)
        elif score >= self.best_score:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.message(val_loss)
            self.counter = 0

    def message(self, val_loss):
        if self.verbose:
            print('Validation loss decreased ({.6f} --> {.6f}).  Saving model ...'.format(self.val_loss_min, val_loss))
        # torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss
