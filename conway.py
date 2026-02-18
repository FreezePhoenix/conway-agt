from dataclasses import dataclass
import numpy as np
from scipy import signal
from typing import Optional

@dataclass
class Strategy:
    action: np.array        # 0 = remain, 1 = dies, 2 = live
    kernel: Optional[np.array] = None # Convolution kernel if present, else use standard.

StandardMask = np.array([
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])

DefaultStrategy = Strategy(
    np.array([1, 1, 0, 2, 1, 1, 1, 1, 1], np.uint8)
)

VirulentStrategy = Strategy(
    np.array([1, 2, 1, 2, 1, 2, 1, 2, 1], np.uint8)
)

TrueVirulentStrategy = Strategy(
    np.array([1, 2, 2, 2, 2], np.uint8),
    np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ], np.uint8)
)

Immutable = Strategy(
    np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
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
            if strategy.kernel is None:
                self.has_standard = True
            self.index_masks.append(
                index == index_grid
            )
        self.strategies = strategies
        self.index_grid = index_grid.astype(int)
        self.state_grid = state_grid.astype(bool)
        self.last_state = np.zeros_like(self.state_grid)
        self.cell_accumulator = np.zeros_like(self.state_grid, int)
        self.count_accumulator = np.zeros((10,) + self.state_grid.shape, int)
        self.accumulator = 0
    def step(self):
        # We want to re-use the convolution if possible, since computing it for large sizes can be very expensive.
        if self.has_standard:
            standard_conv = signal.convolve2d(self.state_grid, StandardMask, mode="same")
        
        new_state = self.last_state
        for index, strategy in enumerate(self.strategies):
            convolution = standard_conv if strategy.kernel is None else signal.convolve2d(
                self.state_grid,
                strategy.kernel,
                mode="same"
            )
            action = strategy.action[convolution]
            self.count_accumulator[
                convolution,
                np.arange(convolution.shape[0])[:,None],
                np.arange(convolution.shape[1])[None,:]
            ] += 2 * self.state_grid - 1
            np.copyto(
                new_state,
                np.choose(
                    action,
                    [
                        self.state_grid,
                        False,
                        True
                    ]
                ),
                where = self.index_masks[index]
            )
        np.add(self.cell_accumulator, 1, self.cell_accumulator, where = self.state_grid != new_state)
        self.last_state = self.state_grid
        self.state_grid = new_state
        self.accumulator += 1
    def reset_best_response(self):
        self.count_accumulator = np.zeros((10,) + self.state_grid.shape, int)

    def print_best_response_analysis(self):
        for index, strategy in enumerate(self.strategies):
            print(f"Strategy {index} has the following kernel: ")
            print(strategy.kernel)
            print(f"And the following response dynamic:")
            print(strategy.action)
            print(f"The following biases were received for each number of neighbors:")
            counts = np.sum(self.count_accumulator, (1, 2))
            counts.resize(strategy.action.shape)
            print(counts)
            print(f"The following would be the optimal response dynamics:")
            print(
                np.choose(
                    np.sign(counts) + 1,
                    [
                        2,
                        0,
                        1
                    ]
                )
            )

    def print_current_state(self):
        print(self.state_grid)
    def print_averages(self):
        print(self.cell_accumulator / self.accumulator)
    def run(self, steps = 100, print = False):
        for i in range(steps):
            self.step()
            if print:
                self.print_current_state()
    def print_average(self):
        print(np.sum(self.cell_accumulator) / (np.size(self.cell_accumulator) * self.accumulator))

# simulation = Simulator(
#     [VirulentStrategy],
#     np.array([
#         [0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0],
#     ]),
#     np.array([
#         [0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 1, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0],
#     ])
# )

# simulation = Simulator(
#     [TrueVirulentStrategy],
#     np.zeros((9,9)),
#     np.array([
#         [0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 1, 0, 1, 0, 0, 0],
#         [0, 0, 0, 0, 1, 0, 0, 0, 0],
#         [0, 0, 0, 1, 0, 1, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0],
#     ])
# )

simulation = Simulator(
    [TrueVirulentStrategy],
    np.zeros((5,5)),
    np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ])
)

simulation.run(100, False)
simulation.print_averages()
simulation.print_average()
simulation.print_best_response_analysis()