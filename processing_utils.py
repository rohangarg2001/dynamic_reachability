"""
NOTE: utils file for generating the dataset
"""

import numpy as np
import pickle
import os
import environment
import matplotlib.pyplot as plt

def sample_free_space(occupancy_grid, coordinate_vecs, batch_size=1):
    """
    NOTE: this functions takes in the occupancy grid which is (100, 100, 60) and has integer value indices. And takes in the coordinate vecs to output 
    the position in the grid.

    Returns : ret_matrix : [batch_size, 3]: points in absolute sense, ret_matrix_grid[batch_size, 3]: random points on the grid
    """
    free_indices = np.where(occupancy_grid == 0)
    num_free_cells = len(free_indices[0])
    
    if num_free_cells == 0:
        return None  # No free cells
    
    ret_matrix = np.zeros((batch_size, 3), dtype=float)
    ret_matrix_grid = np.zeros((batch_size, 3), dtype=float)
    
    for i in range(batch_size):
        """
        ret_matrix is array containing the actual positions, ret_matrix_grid is the free space on the grid indices.
        """
        random_index = np.random.randint(num_free_cells)
        ret_matrix_grid[i, 0] = free_indices[0][random_index]
        ret_matrix_grid[i, 1] = free_indices[1][random_index]
        ret_matrix_grid[i, 2] = free_indices[2][random_index]
        
        ret_matrix[i, 0] = coordinate_vecs[0][(free_indices[0][random_index])]
        ret_matrix[i, 1] = coordinate_vecs[1][free_indices[1][random_index]]
        ret_matrix[i, 2] = coordinate_vecs[2][free_indices[2][random_index]]

    return ret_matrix, ret_matrix_grid

def find_closest_grid_points(points, coordinate_vectors):
    """
    Finds the closest grid points to a batch of points in the environment.

    Args:
        points (np.ndarray): A [N, 3] array of 3D points in the environment.
        coordinate_vectors (list): A list of 3 coordinate vectors for x, y, and theta.

    Returns:
        np.ndarray: The closest grid points to the given points, [N, 3].
        list: List of closest indices, each element is [N,]
    """
    closest_indices = []
    closest_points = np.zeros_like(points)
    for i in range(3):
        indices = np.argmin(np.abs(coordinate_vectors[i] - points[:, i][:, None]), axis=1)
        closest_indices.append(indices)
        closest_points[:, i] = coordinate_vectors[i][indices]
    return closest_points, np.array(closest_indices, dtype=int).T

def sample_non_occluded_points(
    origin: np.ndarray,
    lidar_observation: np.ndarray,
    num_points: int
):
    """Generate non-occluded points in the grid. A point is outside the grid if its x axis is out of [-5,5] and similarly for y axis. theta values are
    unchanged.

    Args:
        origin (np.ndarray): [batch_size, 3]
        lidar_observation (np.ndarray): [batch_size, num_rays] has the data in terms of distance it saw
        num_points (int): number of points to be sampled per origin. We have batch_size number of origins
    """
    num_rays = lidar_observation.shape[1]
    batch_size = origin.shape[0]
    theta_vals = np.linspace(-np.pi, np.pi, num_rays)
    # theta_vals = np.tile(np.linspace(-np.pi, np.pi, num_rays), (batch_size, 1))  # [batch_size, num_rays]

    sampled_points = np.zeros((batch_size, num_points, 3))
    
    for i in range(batch_size):
        ## this is the ith point
        num_points_sampled = 0
        origin_i = origin[i]            # (3,)
        while(num_points_sampled < num_points):
            # select the ray randomly
            ray_index = np.random.randint(0, num_rays)            
            lidar_reading = lidar_observation[i, ray_index]             # between 0.2 and 10
            distance_sampled = np.random.uniform(lidar_reading)
            angle = theta_vals[ray_index]                               # between -pi to pi

            dx = distance_sampled * np.cos(angle)
            dy = distance_sampled * np.sin(angle)

            point = np.array([origin_i[0] + dx, origin_i[1] + dy, origin_i[2]])
            if (np.abs(point[0]) < 5) and np.abs(point[1]) < 5:
                sampled_points[i, num_points_sampled, :] = point
                num_points_sampled += 1
    return sampled_points
                