import abc
import numpy as np
from typing import List, Union, Optional

class Obstacle(metaclass=abc.ABCMeta):
    """An abstract base class for representing an environment obstacle.
    
    Given a state grid, it can compute its own
    distance grid and occupancy grid.
    """

    @abc.abstractmethod
    def distance_grid(
        self,
        state_grid: np.ndarray,
    ) -> np.ndarray:
        """Returns its distance grid.
        
        Args:
            state_grid: The state grid to compute distances for.
                Its shape should be [..., state_dim].

        Returns:
            The distance grid computed from the state grid.
            Its shape should be [...].
        """

    def occupancy_grid(
        self,
        state_grid: np.ndarray,
    ) -> np.ndarray:
        """Returns its occupancy grid.
        
        Args:
            state_grid: The state grid to compute the occupancies for.
                Its shape should be [..., state_dim].

        Returns:
            The occupancy grid computed from the state grid.
            Its shape should be [...].
        """
        return (self.distance_grid(state_grid) <= 0).astype(np.float_)

class SphericalObstacle(Obstacle):
    """A spherical obstacle in an environment.
    
    Inherits from the abstract Obstacle class.
    
    Given a state grid, it can compute its own
    distance grid and occupancy grid.
    """

    def __init__(
        self,
        center: List[float],
        dims: List[int],
        radius: float,
    ):
        """Initializes a spherical obstacle.
        
        Args:
            center: The center state of the obstacle.
                It should be a list of length state_dim (or at least the largest dim in the dims argument).
            dims: The state dimensions that the obstacle resides in. Each specified dim must be smaller than the length of the center argument.
            radius: The radius of the obstacle.
        """
        self.center = np.array(center)
        self.dims = np.array(dims)
        self.radius = radius

    def distance_grid(
        self,
        state_grid: np.ndarray,
    ) -> np.ndarray:
        return np.linalg.norm(
            state_grid[..., self.dims] - self.center[self.dims],
            axis=-1
        ) - self.radius

# class RectangularObstacle(Obstacle):
#     def __init__(
#             self,
#             center: List[float],
#             dims: List[int],
#             x_length: float,
#             y_length: float
#     ):
#         """Initializes a rectangular obstacle

#         Args:
#             center (List[float]): Center state of the recatngular obstacle. [x,y]
#             dims (List[int]): State dimensions in which the obstacle resides
#             x_length (float): length of the rectangle in the x direction
#             y_length (float): length of the rectangle in the y direction
#         """
#         self.center = np.array(center)
#         self.dims = np.array(dims)
#         self.x_length = x_length
#         self.y_length = y_length
#         self.x_min = center[0] - self.x_length/2
#         self.x_max = center[0] + self.x_length/2
#         self.y_min = center[1] - self.y_length/2
#         self.y_max = center[1] + self.y_length/2
    
#     def distance_grid(
#             self,
#             state_grid: np.ndarray
#     ) -> np.ndarray:
#         state_grid_obstacle = state_grid[..., self.dims]        # this is (100, 100, 60, 2) array
#         for elem in state_grid_obstacle[..., :]:
#             print(np.shape(elem))
#         for i in range(state_grid_obstacle.shape[0]):
#             point = state_grid_obstacle[i]
#             print(point)
#             px, py = point[0], point[1]
#             inside_x = self.x_max < px < self.x_max
#             inside_y = self.y_min < py < self.y_max
#             if inside_x and inside_y:
#                 dx = min(px - self.x_max, self.x_max - px)
#                 dy = min(py - self.y_min, self.y_max - py)
#                 signed_distance[i] = -min(dx, dy)
#             else:
#                 cx = max(self.x_max, min(px, self.x_max))
#                 cy = max(self.y_min, min(py, self.y_max))
#                 signed_distance[i] = ((px - cx)**2 + (py - cy)**2)**0.5
#         return signed_distance        
    
class CylindricalObstacle2D(SphericalObstacle):
    """A 2D cylindrical obstacle in an environment.
    
    Inherits from SphericalObstacle.
    
    Given a state grid, it can compute its own
    distance grid and occupancy grid.
    It also has a height for visualization purposes only.
    """

    def __init__(
        self,
        center: List[float],
        radius: float,
        height: float,
    ):
        """Initializes a 2D cylindrical obstacle in the Dubins4D environment.
        
        Args:
            center: The [x, y] position of the obstacle.
            radius: The radius of the obstacle.
            height: The height of the obstacle (for visualization purposes only).
        """
        super().__init__(center=center, dims=[0, 1], radius=radius)
        self.height = height