# Copyright 2021 MosaicML. All Rights Reserved.

from __future__ import annotations

from typing import Optional, Type, Union

import torch

from composer.core import Algorithm, Event, State
from composer.loggers import Logger

class SAF(Algorithm):
    """Implements the SAF algorithm.

    Args:
        training_set: The training set to be used.
        network: The network with weights to be optimized.
        learning_rate: The learning rate for the optimization.
        epochs: The number of epochs for the training.
        iterations_per_epoch: The number of iterations per epoch.
        saf_starting_epoch: The epoch at which to start SAF.
        saf_coefficients: The coefficients for the SAF algorithm.
        temperature: The temperature parameter for the SAF algorithm.
        saf_hyperparameter: The hyperparameter for the SAF algorithm.
        ema_decay_factor: The decay factor for the Exponential Moving Average (EMA).
    """

    def __init__(self, training_set, network, learning_rate, epochs, iterations_per_epoch, saf_starting_epoch, saf_coefficients, temperature, saf_hyperparameter, ema_decay_factor):
        self.training_set = training_set
        self.network = network
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.iterations_per_epoch = iterations_per_epoch
        self.saf_starting_epoch = saf_starting_epoch
        self.saf_coefficients = saf_coefficients
        self.temperature = temperature
        self.saf_hyperparameter = saf_hyperparameter
        self.ema_decay_factor = ema_decay_factor

    def forward(self):
        # Steps 1 to 9 of the SAF algorithm will be implemented here.
        # This method will output a flat minimum solution.
        pass
