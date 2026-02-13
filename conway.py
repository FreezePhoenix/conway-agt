from dataclasses import dataclass
import numpy as np
from scipy import signal
from typing import Union

@dataclass
class Strategy:
    action: np.array        # 0 = remain, 1 = dies, 2 = live
    conv_mask: Union[np.array, str]  # 3x3 numpy array

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
            if strategy.conv_mask == 'standard':
                self.has_standard = True
            self.index_masks.append(
                index == index_grid
            )
        self.strategies = strategies
        self.index_grid = index_grid
        self.state_grid = state_grid
        self.cell_accumulator = self.state_grid
        self.accumulator = 1
    def step(self):
        # We want to re-use the convolution if possible, since computing it for large sizes can be very expensive.
        
        standard_conv = signal.convolve2d(self.state_grid, StandardMask, mode="same")
        new_state = np.zeros_like(self.state_grid)
        for index, strategy in enumerate(self.strategies):
            convolution = standard_conv if strategy.conv_mask == 'standard' else signal.convolve2d(
                self.state_grid,
                strategy.conv_mask,
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
        self.state_grid = new_state
        self.cell_accumulator += self.state_grid
        self.accumulator += 1

    def print_current_state(self):
        print(self.state_grid)
    def print_averages(self):
        print(self.cell_accumulator / self.accumulator)
    def run(self, steps = 100):
        for i in range(steps):
            self.step()
            
simulation = Simulator(
    [DefaultStrategy],
    np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]),
    np.array([
        [0, 0, 0, 0],
        [1, 1, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
)

simulation.print_current_state()
simulation.step()
simulation.print_current_state()
simulation.step()
simulation.print_current_state()
simulation.step()
simulation.print_current_state()
simulation.print_averages()