from typing import Optional

import numpy as np
import torch
import torchmetrics


class SulciErrorLocal(torchmetrics.Metric):
    def __init__(self, ignore_index: list = [0], epsilon: float = 1e-8):
        """Constructor for the class.

        Args:
            ignore_index (list, optional): List of indexes to ignore
                                           during metric calculation.
                                           Defaults to [0].
            epsilon (float, optional): Constant added to avoid division by zero error.
                                       Defaults to 1e-8.
        """
        super().__init__()
        self.epsilon = epsilon
        self.ignore_index = ignore_index

        self.add_state("sulci_errors", default=[], dist_reduce_fx="sum")

    def update(self, pred_proba: torch.Tensor, target: torch.Tensor):
        # one-hot probabilities to labels (batch, channels,...) to (batch, ...)
        preds = torch.argmax(pred_proba, dim=1)

        assert preds.shape == target.shape

        total_sulci_errors = []
        for batch in range(preds.shape[0]):
            # array storing sum of errors for each sulcus per batch
            sulci_local_errors = []
            for ch in range(pred_proba.shape[1]):
                if ch not in self.ignore_index:
                    TP = torch.sum((preds == ch) & (target == ch))
                    FP = torch.sum((preds == ch) & (target != ch))
                    FN = torch.sum((preds != ch) & (target == ch))
                    # this will be the total error from all batches
                    sulci_local_errors.append((FP + FN) / (FP + FN + TP + self.epsilon))
            total_sulci_errors.append(sulci_local_errors)

        # mean error per batch per sulci
        total_sulci_errors = np.mean(np.stack(total_sulci_errors), axis=0)

        # store the mean error per sulcus of a batch
        self.sulci_errors.append(torch.tensor(total_sulci_errors))

    def compute(self):
        """ Returns an array with average error for each sulcus per epoch"""
        return torch.mean(torch.stack(self.sulci_errors),
                          dim=0, dtype=torch.float32)


class SulciErrorSubject(torchmetrics.Metric):
    def __init__(self, ignore_index: list = [0], epsilon: float = 1e-8):
        """Constructor for the class.

        Args:
            ignore_index (list, optional): List of indexes to ignore
                                            during metric calculation.
                                            Defaults to [0].
            epsilon (float, optional): Constant added to avoid division by zero error.
                                        Defaults to 1e-8.
        """
        super().__init__()
        self.epsilon = epsilon
        self.ignore_index = ignore_index

        self.add_state("sulci_errors", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred_proba: torch.Tensor, target: torch.Tensor):
        # one-hot probabilities to labels (batch, channels,...) to (batch, ...)
        preds = torch.argmax(pred_proba, dim=1)

        assert preds.shape == target.shape

        # calculate error per subject in a batch
        batch_errors = torch.tensor(0, dtype=torch.float32)
        for batch in range(preds.shape[0]):
            # error at the subject scale E_si
            sulci_sizes = []
            sulci_errors = []
            for ch in range(pred_proba.shape[1]):
                if ch not in self.ignore_index:
                    TP = torch.sum((preds[batch] == ch) & (target[batch] == ch))
                    FP = torch.sum((preds[batch] == ch) & (target[batch] != ch))
                    FN = torch.sum((preds[batch] != ch) & (target[batch] == ch))
                    sulci_sizes.append(FN + TP)
                    sulci_errors.append((FP + FN) / (FP + FN + 2*TP + self.epsilon))

            sulci_sizes = np.array(sulci_sizes)
            sulci_errors = np.array(sulci_errors)
            batch_errors += np.sum(sulci_errors * (sulci_sizes / np.sum(sulci_sizes)))
        self.sulci_errors += batch_errors
        self.total += preds.shape[0]

    def compute(self):
        """ Returns the average error for all subjects per epoch"""
        return self.sulci_errors / self.total


def compute_sulci_Elocal(pred_proba: torch.Tensor,
                         target: torch.Tensor,
                         ignore_index: list = [0],
                         epsil: float = 1e-8):

    predicted = torch.argmax(pred_proba, dim=1)
    # error at the sulcus scale E_local

    # sulci_local_errors is one when the sulcus was absent and labeled by the model
    # or when it was present but not labeled by the model
    # small sulci are frequently absent so this explains highly variable error rates
    # when averaging the error rates per subject
    sulci_local_errors = {}
    for ch in range(pred_proba.shape[1]):

        if ch not in ignore_index:
            TP = torch.sum((predicted == ch) & (target == ch))
            FP = torch.sum((predicted == ch) & (target != ch))
            FN = torch.sum((predicted != ch) & (target == ch))

            sulci_local_errors[ch] = (FP + FN) / (FP + FN + TP + epsil)
    return sulci_local_errors


def compute_subject_Esi(pred_proba: torch.Tensor,
                        target: torch.Tensor,
                        ignore_index: list = [0],
                        epsil: float = 1e-8):

    predicted = torch.argmax(pred_proba, dim=1)
    # error at the subject scale E_si
    subject_error = 0
    sulci_sizes = []
    sulci_errors = []
    for ch in range(pred_proba.shape[1]):

        if ch not in ignore_index:
            TP = torch.sum((predicted == ch) & (target == ch))
            FP = torch.sum((predicted == ch) & (target != ch))
            FN = torch.sum((predicted != ch) & (target == ch))
            sulci_sizes.append(FN + TP)
            sulci_errors.append((FP + FN) / (FP + FN + 2*TP + epsil))

    sulci_sizes = np.array(sulci_sizes)
    sulci_errors = np.array(sulci_errors)
    subject_error = np.sum(sulci_errors * (sulci_sizes / np.sum(sulci_sizes)))
    return subject_error
