"""Exponentiation of a stationary velocity field"""

# NOTE:
# .. VERY ALPHA STAGE
# .. move everything related to *stationary* velocity fields here
# .. implement our own backward pass
#       > may be more efficient (ram, cpu) than autograd
#       > allows the backward pass to be called as a forward pass

import torch
from ._grid import grid_pull, channel2grid, grid2channel, identity_grid
from ._finite_differences import diff
from ..core import linalg
from warnings import warn

__all__ = ['exp']


def exp(vel, inverse=False, steps=None, interpolation='linear', bound='dft',
        displacement=False, inplace=False):
    """Exponentiate a stationary velocity field

    Parameters
    ----------
    vel : (batch, ..., ndim)
        Velocity field
    inverse : bool, default=False
        Compute the inverse rather than the forward deformation
    steps : int, default=8
        Number of squaring and scaling steps
        (Equivalent to ``2**steps`` integration steps)
    interpolation : InterpolationType, default='linear'
    bound : BoundType, default='dft'
    displacement : bool, default=False
        If ``True``, return a displacement field.
        Else return a transformation field.
    inplace : bool, default=False
        If ``True``, can reuse ``vel``'s memory.
        Enforced to ``False`` if ``vel.requires_grad is True``

    Returns
    -------
    grid : (batch, ..., ndim)
        Transformation or displacement grid.

    """
    # Deal with inplace computation
    if vel.requires_grad:
        if inplace:
            warn('Inplace computation would break the computational graph.',
                 RuntimeWarning)
        inplace = False
    if not inplace:
        vel = vel.clone()

    opt = {'inverse': inverse, 'steps': steps, 'interpolation': interpolation,
           'bound': bound, 'displacement': displacement}

    return _exp_sq(vel, **opt)


def _exp_sq(vel, jac=False, inverse=False, steps=8,
            interpolation='linear', bound='dft', displacement=False):
    # /!\ This function may process inplace without warning

    if steps is None or steps == float('Inf'):
        steps = 8

    # Precompute identity + aliases
    dtype = vel.dtype
    device = vel.device
    id = identity_grid(vel.shape[1:-1], dtype=dtype, device=device)
    opt = {'interpolation': interpolation, 'bound': bound}

    # initialise jacobian
    jac = _jacobian(vel, bound=bound) if jac is True else jac

    def scl_outplace(v, j=None):
        v = v / (2**steps)
        for i in range(steps):
            g = id + v
            if j is not None:
                j = _compose_jac(j, g, j, **opt)
            v = v + _pull_vel(v, g, **opt)
        return v, j

    def scl_inplace(v, j=None):
        v /= (2**steps)
        for i in range(steps):
            g = id + v
            if j is not None:
                j = _compose_jac(j, g, j, **opt)
            v += _pull_vel(v, g, **opt)
        return v, j

    if vel.requires_grad:
        scl = scl_outplace
        smalldef = lambda v: id + v
        if inverse:
            vel = -vel
    else:
        scl = scl_inplace
        smalldef = lambda v: v.__iadd__(id)
        if inverse:
            torch.neg(vel, out=vel)

    vel, jac = scl(vel, jac)
    if not displacement:
        vel = smalldef(vel)
    return vel if jac is False else (vel, jac)


def _exp_sq_backward(vel, grad, hess=None, steps=8, interpolation='linear', bound='dft'):

    if steps is None or steps == float('Inf'):
        steps = 8

    # Precompute identity + aliases
    dtype = vel.dtype
    device = vel.device
    id = identity_grid(vel.shape[1:-1], dtype=dtype, device=device)[None, ...]
    opt = {'interpolation': interpolation, 'bound': bound}
    scl = 1/(2**steps)

    # # scaling and squaring
    # disp, jac = _exp_sq(vel, jac=True, displacement=True, **opt)
    #
    # # roll back Jacobian by one step
    # jac = _jac_div_smalldef(jac, vel*scl)

    vel = vel.neg_().__imul__(scl)
    grid = id + vel
    jac = _jacobian(vel)

    # pullback gradient
    for k in range(steps):
        # transpose jacobian + compute determinant
        det = torch.det(jac)
        jac = jac.transpose_(-1, -2)

        # push gradient
        grad2 = _pull_vel(grad, grid, **opt)
        grad2 = linalg.matvec(jac, grad2)
        grad2 *= det[..., None]
        grad += grad2
        del grad2

        # push hessian
        if hess is not None:
            hess2 = _pull_jac(hess, grid, **opt)
            hess2 = torch.matmul(jac, hess2)
            hess2 = torch.matmul(hess2, jac.transpose(-1, -2))
            hess2 *= det[..., None, None]
            hess += hess2

        # squaring step
        jac = jac.transpose_(-1, -2)
        jac = _compose_jac(jac, grid, jac, **opt)
        vel += _pull_vel(vel, grid, **opt)
        grid = id + vel



def _jac_div_smalldef(jac, vel):
    jac2 = _jacobian(vel)
    jac = linalg.rmdiv(jac, jac2)
    return jac


def _jacobian(vel, **opt):
    """Compute the Jacobian of a displacement field.

    Parameters
    ----------
    vel : (B, ..., D) tensor
        Input velocity or displacement field
    bound : BoundType, default='dct2'
        Boundary condition

    Returns
    -------
    jac : (B, ..., D, D) tensor
        Output Jacobian of the input field

    """
    ndims = vel.shape[-1]
    jac = diff(vel, dim=list(range(1, vel.dim()-1)), side='c', **opt)
    jac = jac + torch.eye(ndims, dtype=jac.dtype, device=jac.device)
    return jac


def _pull_vel(vel, grid, *args, **kwargs):
    """Interpolate a velocity/grid/displacement field.

    Parameters
    ----------
    vel : (batch, ..., ndim) tensor
        Velocity
    grid : (batch, ..., ndim) tensor
        Transformation field
    opt : dict
        Options to ``grid_pull``

    Returns
    -------
    pulled_vel : (batch, ..., ndim) tensor
        Velocity

    """
    return channel2grid(grid_pull(grid2channel(vel), grid, *args, **kwargs))


def _pull_jac(jac, grid, out=None, *args, **kwargs):
    """Interpolate a jacobian field.

    Parameters
    ----------
    jac : (batch, ..., ndim, ndim) tensor
        Jacobian
    grid : (batch, ..., ndim) tensor
        Transformation field
    opt : dict
        Options to ``grid_pull``

    Returns
    -------
    pulled_jac : (batch, ..., ndim, ndim) tensor
        Velocity

    """
    jacs = []
    for d in range(jac.shape[-1]):
        jac1 = jac[..., d]
        jac1 = _pull_vel(jac1, grid, *args, **kwargs)
        if out is not None:
            out[..., d] = jac1
        else:
            jacs.append(jac1)
    if out is None:
        out = torch.stack(jacs, dim=-1)
    return out


def _compose_vel(vel_a, grid, vel_c=None, *args, **kwargs):
    """

    Parameters
    ----------
    vel_a : (B, ..., D) tensor
        Velocity to sample.
    grid : (B, ..., D) tensor
        Sampling grid
    vel_c : (B, ..., D) tensor, optional
        Placeholder for the output
    opt : dict
        Options for grid_pull

    Returns
    -------
    vel_c : (B, ..., D) tensor
        Sampled velocity

    """
    grid = _pull_vel(vel_a, grid, *args, **kwargs)
    vel_c = torch.add(vel_a, grid, out=vel_c)
    return vel_c


def _compose_jac(jac_a, grid, jac_b=None, jac_c=None, *args, **kwargs):
    """

    Parameters
    ----------
    jac_a : (B, ..., D, D) tensor
        Jacobian to sample.
    grid : (B, ..., D) tensor
        Sampling grid
    jac_b : (B, ..., D, D) tensor
        Jacobian to multiply with.
    jac_c : (B, ..., D, D) tensor, optional
        Placeholder for the output
    opt : dict
        Options for grid_pull

    Returns
    -------
    jac_c : (B, ..., D, D) tensor
        Sampled Jacobian

    """
    jac_c = _pull_vel(jac_a, grid, out=jac_c, *args, **kwargs)
    jac_c = torch.matmul(jac_b, jac_c, out=jac_c)
    return jac_c
