import jax.numpy as jnp

from hj_reachability import dynamics
from hj_reachability import sets
from typing import List

class Dubins3D(dynamics.ControlAndDisturbanceAffineDynamics):
    """Dubins3D dynamics implementation to be used by the hj_reachability package.
    
    The dynamics are:
    dx  = v * cos(th) + c1 + dst_dx                     # what are these constants
    dy  = v * sin(th) + c2 + dst_dy                    
    dth = w           + c3 + dst_dth

    with control:
    [v, w]
    v_min <= v <= v_max
    w_min <= w <= w_max

    and disturbance:
    [dst_dx, dst_dy, dst_dth]
    || [dst_dx, dst_dy] || <= dst_dxdy_max
    | dst_dth | <= dst_dth_max
    """

    def __init__(
        self,
        control_min: List[float],
        control_max: List[float],
        disturbance_norm_bounds: List[float],
        c1, c2, c3,
        control_mode: str = 'max',
        disturbance_mode: str = 'min',
    ):
        """Initializes Dubins3D dynamics.
        
        Args:
            control_min: The lower bounds of the control space. It should be a list of length control_dim.
            control_max: The upper bounds of the control space. It should be a list of length control_dim.
            disturbance_norm_bounds: The disturbance norm bounds [dst_dxdy_max, dst_dth_max].
            control_mode: Either 'max' or 'min'.
            disturbance_mode: Either 'max' or 'min'.
        """
        self.control_min = control_min
        self.control_max = control_max
        self.disturbance_norm_bounds = disturbance_norm_bounds
        self.c1, self.c2, self.c3 = c1, c2, c3
        control_space = sets.Box(jnp.asarray(control_min), jnp.asarray(control_max))
        # custom disturbance spaces
        self.dxdy_disturbance_space = sets.Ball(jnp.asarray([0, 0]), disturbance_norm_bounds[0])
        self.dth_disturbance_space = sets.Ball(jnp.asarray([0]), disturbance_norm_bounds[1])
        super().__init__(control_mode, disturbance_mode, control_space, None)

    def open_loop_dynamics(self, state, time):
        return jnp.array([
            self.c1,
            self.c2,
            self.c3,
        ])
    
    def control_jacobian(self, state, time):
        _, _, th = state
        return jnp.array([
            [jnp.cos(th), 0.],
            [jnp.sin(th), 0.],
            [0., 1.],
        ])
    
    def disturbance_jacobian(self, state, time):
        return jnp.array([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
        ])

    def optimal_control_and_disturbance(self, state, time, grad_value):
        """Computes the optimal control and disturbance realized by the HJ PDE Hamiltonian."""
        control_direction = grad_value @ self.control_jacobian(state, time)
        if self.control_mode == "min":
            control_direction = -control_direction
        disturbance_direction = grad_value @ self.disturbance_jacobian(state, time)
        if self.disturbance_mode == "min":
            disturbance_direction = -disturbance_direction
        dxdy_disturbance_extreme_point = self.dxdy_disturbance_space.extreme_point(disturbance_direction[:2])
        dth_disturbance_extreme_point = self.dth_disturbance_space.extreme_point(disturbance_direction[2:])
        return (self.control_space.extreme_point(control_direction),
                jnp.concatenate((dxdy_disturbance_extreme_point, dth_disturbance_extreme_point), axis=0))
    
    def partial_max_magnitudes(self, state, time, value, grad_value_box):
        """Computes the max magnitudes of the Hamiltonian partials over the `grad_value_box` in each dimension."""
        del value, grad_value_box  # unused
        # An overestimation; see Eq. (25) from https://www.cs.ubc.ca/~mitchell/ToolboxLS/toolboxLS-1.1.pdf.
        return (jnp.abs(self.open_loop_dynamics(state, time)) +
                jnp.abs(self.control_jacobian(state, time)) @ self.control_space.max_magnitudes +
                jnp.abs(self.disturbance_jacobian(state, time)) @ jnp.concatenate((self.dxdy_disturbance_space.max_magnitudes, self.dth_disturbance_space.max_magnitudes), axis=0))