from dataclasses import dataclass
import numpy as np
from scipy import signal
from typing import Union

@dataclass
class Strategy:
    action: np.array        # 0 = remain, 1 = dies, 2 = live
    kernel: Union[np.array, str]  # convolution kernel

StandardMask = np.array([
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

DefaultStrategy = Strategy(
    np.array([1, 1, 0, 2, 1, 1, 1, 1, 1]),
    'standard'
)

VirusStrategy = Strategy(
    np.array([1, 2, 1, 2, 1, 2, 1, 2, 1]),
    'standard'
)

Immutable = Strategy(
    np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
    'standard'
)

UnitBlinker = Strategy(
    np.array([2, 1]),
    np.array([[1]])
)

class Simulator:
    def __init__(
        self,
        strategies,
        index_grid: np.array,
        state_grid: np.array,
    ):
        """
        strategies : list of Strategy objects
        index_grid : 2D numpy array of integers (indices into strategies)
        state_grid : 2D numpy array of 0s and 1s representing simulator state
        """
        self.index_masks = []
        self.has_standard = False
        for index, strategy in enumerate(strategies):
            if strategy.kernel == 'standard':
                self.has_standard = True
            self.index_masks.append(
                index == index_grid
            )
        self.strategies = strategies
        self.index_grid = index_grid
        self.state_grid = state_grid
        self.cell_accumulator = np.zeros_like(self.state_grid)
        self.accumulator = 0
    def step(self):
        # We want to re-use the convolution if possible, since computing it for large sizes can be very expensive.
        
        standard_conv = signal.convolve2d(self.state_grid, StandardMask, mode="same")
        new_state = np.zeros_like(self.state_grid)
        for index, strategy in enumerate(self.strategies):
            convolution = standard_conv if strategy.kernel == 'standard' else signal.convolve2d(
                self.state_grid,
                strategy.kernel,
                mode="same"
            )
            action = strategy.action[convolution]
            np.copyto(
                new_state,
                np.choose(
                    action,
                    [
                        self.state_grid,
                        0,
                        1
                    ]
                ),
                where = self.index_masks[index]
            )
        self.cell_accumulator += np.where(self.state_grid != new_state, 1, 0)
        self.state_grid = new_state
        self.accumulator += 1

    def print_current_state(self):
        print(self.state_grid)
    def print_averages(self):
        print(self.cell_accumulator / self.accumulator)
    def run(self, steps = 100):
        for i in range(steps):
            self.step()
            
simulation = Simulator(
    [UnitBlinker],
    np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]),
    np.array([
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0],
    ])
)

simulation.run(99)
simulation.print_current_state()
simulation.print_averages()