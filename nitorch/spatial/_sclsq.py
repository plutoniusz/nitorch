"""Integrate stationary velocity fields."""

from ._grid import grid_pull, grid_push, identity_grid, grid_jacobian
from ._regularisers import regulariser_grid
from ._shoot import greens, greens_apply
from nitorch.core import utils, py, linalg
import torch

__all__ = ['exp', 'exp_forward', 'exp_backward']


def exp(vel, inverse=False, steps=8, interpolation='linear', bound='dft',
        displacement=False):
    """Exponentiate a stationary velocity field by scaling and squaring.

    Parameters
    ----------
    vel : ([batch], *spatial, dim) tensor
        Stationary velocity field.
    inverse : bool, default=False
        Generate the inverse transformation instead of the forward.
    steps : int, default=8
        Number of scaling and squaring steps
        (corresponding to 2**steps integration steps).
    interpolation : {0..7}, default=1
        Interpolation order
    bound : str, default='dft'
        Boundary conditions
    displacement : bool, default=False
        Return a displacement field rather than a transformation field

    Returns
    -------
    grid : ([batch], *spatial, dim) tensor
        Exponentiated tranformation

    """
    return exp_forward(vel, inverse, steps, interpolation, bound, displacement)


def exp_forward(vel, inverse=False, steps=8, interpolation='linear',
                bound='dft', displacement=False):
    """Exponentiate a stationary velocity field by scaling and squaring.

    Parameters
    ----------
    vel : ([batch], *spatial, dim) tensor
        Stationary velocity field.
    inverse : bool, default=False
        Generate the inverse transformation instead of the forward.
    steps : int, default=8
        Number of scaling and squaring steps
        (corresponding to 2**steps integration steps).
    interpolation : {0..7}, default=1
        Interpolation order
    bound : str, default='dft'
        Boundary conditions
    displacement : bool, default=False
        Return a displacement field rather than a transformation field

    Returns
    -------
    grid : ([batch], *spatial, dim) tensor
        Exponentiated tranformation

    """

    vel = -vel if inverse else vel.clone()

    # Precompute identity + aliases
    dim = vel.shape[-1]
    spatial = vel.shape[-1-dim:-1]
    id = identity_grid(spatial, **utils.backend(vel))
    opt = {'interpolation': interpolation, 'bound': bound}

    if vel.requires_grad:
        iadd = lambda x, y: x.add(y)
    else:
        iadd = lambda x, y: x.add_(y)

    vel /= (2**steps)
    for i in range(steps):
        vel = iadd(vel, _pull_vel(vel, id + vel, **opt))

    if not displacement:
        vel += id
    return vel


def _jhj(jac, hess):
    """J*H*J', where H is symmetric and stored sparse"""

    # Matlab symbolic toolbox
    #
    # 2D:
    # out[00] = h00*j00^2 + h11*j01^2 + 2*h01*j00*j01
    # out[11] = h00*j10^2 + h11*j11^2 + 2*h01*j10*j11
    # out[01] = h00*j00*j10 + h11*j01*j11 + h01*(j01*j10 + j00*j11)
    #
    # 3D:
    # out[00] = h00*j00^2 + 2*h01*j00*j01 + 2*h02*j00*j02 + h11*j01^2 + 2*h12*j01*j02 + h22*j02^2
    # out[11] = h00*j10^2 + 2*h01*j10*j11 + 2*h02*j10*j12 + h11*j11^2 + 2*h12*j11*j12 + h22*j12^2
    # out[22] = h00*j20^2 + 2*h01*j20*j21 + 2*h02*j20*j22 + h11*j21^2 + 2*h12*j21*j22 + h22*j22^2
    # out[01] = j10*(h00*j00 + h01*j01 + h02*j02) + j11*(h01*j00 + h11*j01 + h12*j02) + j12*(h02*j00 + h12*j01 + h22*j02)
    # out[02] = j20*(h00*j00 + h01*j01 + h02*j02) + j21*(h01*j00 + h11*j01 + h12*j02) + j22*(h02*j00 + h12*j01 + h22*j02)
    # out[12] = j20*(h00*j10 + h01*j11 + h02*j12) + j21*(h01*j10 + h11*j11 + h12*j12) + j22*(h02*j10 + h12*j11 + h22*j12)

    # TODO: Should I wrap this in a torchscript function?

    dim = jac.shape[-1]
    hess = utils.movedim(hess, -1, 0)
    jac = utils.movedim(jac, [-2, -1], [0, 1])
    out = torch.empty_like(hess)
    if dim == 1:
        out[0] = jac[0, 0].square() * hess[0]
    elif dim == 2:
        out[0] = (jac[0, 0].square() * hess[0] +
                  jac[0, 1].square() * hess[1] +
                  2 * jac[0, 0] * jac[0, 1] * hess[2])
        out[1] = (jac[1, 0].square() * hess[0] +
                  jac[1, 1].square() * hess[1] +
                  2 * jac[1, 0] * jac[1, 1] * hess[2])
        out[2] = (jac[0, 0] * jac[1, 0] * hess[0] +
                  jac[0, 1] * jac[1, 1] * hess[1] +
                  jac[0, 1] * jac[1, 0] * hess[2] +
                  jac[0, 0] * jac[1, 1] * hess[2])
    elif dim == 3:
        h00 = hess[0]
        h11 = hess[1]
        h22 = hess[2]
        h01 = hess[3]
        h02 = hess[4]
        h12 = hess[5]
        j00 = jac[0, 0]
        j01 = jac[0, 1]
        j02 = jac[0, 2]
        j10 = jac[1, 0]
        j11 = jac[1, 1]
        j12 = jac[1, 2]
        j20 = jac[2, 0]
        j21 = jac[2, 1]
        j22 = jac[2, 2]
        out[0] = h00*j00.square() + 2*h01*j00*j01 + 2*h02*j00*j02 + h11*j01.square() + 2*h12*j01*j02 + h22*j02.square()
        out[1] = h00*j10.square() + 2*h01*j10*j11 + 2*h02*j10*j12 + h11*j11.square() + 2*h12*j11*j12 + h22*j12.square()
        out[2] = h00*j20.square() + 2*h01*j20*j21 + 2*h02*j20*j22 + h11*j21.square() + 2*h12*j21*j22 + h22*j22.square()
        out[3] = j10*(h00*j00 + h01*j01 + h02*j02) + j11*(h01*j00 + h11*j01 + h12*j02) + j12*(h02*j00 + h12*j01 + h22*j02)
        out[4] = j20*(h00*j00 + h01*j01 + h02*j02) + j21*(h01*j00 + h11*j01 + h12*j02) + j22*(h02*j00 + h12*j01 + h22*j02)
        out[5] = j20*(h00*j10 + h01*j11 + h02*j12) + j21*(h01*j10 + h11*j11 + h12*j12) + j22*(h02*j10 + h12*j11 + h22*j12)
    out = utils.movedim(out, 0, -1)
    return out


def exp_backward(vel, grad, hess=None, inverse=False, steps=8,
                 interpolation='linear', bound='dft', rotate_grad=False):
    """Backward pass of SVF exponentiation.

    This should be much more memory-efficient than the autograd pass
    as we don't have to store intermediate grids.

    I am using DARTEL's derivatives (from the code, not the paper).
    From what I get, it corresponds to pushing forward the gradient
    (computed in observation space) recursively while squaring the
    (inverse) transform.
    Remember that the push forward of g by phi is
                    |iphi| iphi' * g(iphi)
    where iphi is the inverse of phi. We could also have implemented
    this operation as: inverse(phi)' * push(g, phi), since
    push(g, phi) \approx |iphi| g(iphi). It has the advantage of using
    push rather than pull, which might preserve better positive-definiteness
    of the Hessian, but requires the inversion of (potentially ill-behaved)
    Jacobian matrices.

    Note that gradients must first be rotated using the Jacobian of
    the exponentiated transform so that the denominator refers to the
    initial velocity (we want dL/dV0, not dL/dPsi).
    THIS IS NOT DONE INSIDE THIS FUNCTION YET (see _dartel).

    Parameters
    ----------
    vel : (..., *spatial, dim) tensor
        Velocity
    grad : (..., *spatial, dim) tensor
        Gradient with respect to the output grid
    hess : (..., *spatial, dim*(dim+1)//2) tensor, optional
        Symmetric hessian with respect to the output grid.
    inverse : bool, default=False
        Whether the grid is an inverse
    steps : int, default=8
        Number of scaling and squaring steps
    interpolation : str or int, default='linear'
    bound : str, default='dft'
    rotate_grad : bool, default=False
        If True, rotate the gradients using the Jacobian of exp(vel).

    Returns
    -------
    grad : (..., *spatial, dim) tensor
        Gradient with respect to the SVF
    hess : (..., *spatial, dim*(dim+1)//2) tensor, optional
        Approximate (block diagonal) Hessian with respect to the SVF

    """
    opt = dict(bound=bound, interpolation=interpolation)
    dim = vel.shape[-1]
    shape = vel.shape[-dim-1:-1]
    id = identity_grid(shape, **utils.backend(vel))
    vel = vel.clone()
    vel /= (-1 if not inverse else 1) * (2**steps)

    if rotate_grad:
        # It forces us to perform a forward exponentiation, which
        # is a bit annoying...
        grid = exp_forward(vel, steps=steps, displacement=True, **opt)
        grid = grid_jacobian(grid, bound=bound, type='disp')
        grid = grid.transpose(-1, -2)
        grad = linalg.matvec(grid, grad)
        if hess is not None:
            hess = _jhj(grid, hess)
        del grid

    for _ in range(steps):
        jac = grid_jacobian(vel, bound=bound, type='disp')
        det = jac.det()
        jac = jac.transpose(-1, -2)
        grad0 = grad
        grad = _pull_vel(grad, id + vel, **opt)  # \
        grad = linalg.matvec(jac, grad)          # | push forward
        grad *= det[..., None]                   # /
        grad += grad0                            # add all scales (SVF)
        if hess is not None:
            hess0 = hess
            hess = _pull_vel(hess, id + vel, **opt)
            hess = _jhj(jac, hess)
            hess *= det[..., None]
            hess += hess0
        vel += _pull_vel(vel, id + vel, **opt)   # squaring

    if inverse:
        grad.neg_()

    grad /= (2**steps)
    if hess is not None:
        hess /= (2**steps)

    return (grad, hess) if hess is not None else grad


class _Exp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vel, inverse, steps, interpolation, bound, displacement):
        if vel.requires_grad:
            ctx.save_for_backward(vel)
            ctx.args = {'steps': steps, 'inverse': inverse,
                        'interpolation': interpolation, 'bound': bound}
        return exp_forward(vel, inverse, steps, interpolation, bound, displacement)

    @staticmethod
    def backward(ctx, grad):
        vel, = ctx.saved_tensors
        grad = exp_backward(vel, grad,
                            steps=ctx.args['steps'],
                            inverse=ctx.args['inverse'],
                            interpolation=ctx.args['interpolation'],
                            bound=ctx.args['bound'],
                            rotate_grad=True)
        return (grad,) + (None,)*5


def _pull_vel(vel, grid, *args, **kwargs):
    """Interpolate a velocity/grid/displacement field.

    Notes
    -----
    Defaults differ from grid_pull:
    - bound -> dft
    - extrapolate -> True

    Parameters
    ----------
    vel : ([batch], *spatial, ndim) tensor
        Velocity
    grid : ([batch], *spatial, ndim) tensor
        Transformation field
    kwargs : dict
        Options to ``grid_pull``

    Returns
    -------
    pulled_vel : ([batch], *spatial, ndim) tensor
        Velocity

    """
    kwargs.setdefault('bound', 'dft')
    kwargs.setdefault('extrapolate', True)
    dim = grid.shape[-1]
    vel = utils.movedim(vel, -1, -dim-1)
    vel_no_batch = vel.dim() == dim + 1
    grid_no_batch = grid.dim() == dim + 1
    if vel_no_batch:
        vel = vel[None]
    if grid_no_batch:
        grid = grid[None]
    vel = grid_pull(vel, grid, *args, **kwargs)
    vel = utils.movedim(vel, -dim-1, -1)
    if vel_no_batch and grid_no_batch:
        vel = vel[0]
    return vel


def _push_vel(vel, grid, *args, **kwargs):
    """Push a velocity/grid/displacement field.

    Notes
    -----
    Defaults differ from grid_pull:
    - bound -> dft
    - extrapolate -> True

    Parameters
    ----------
    vel : ([batch], *spatial, ndim) tensor
        Velocity
    grid : ([batch], *spatial, ndim) tensor
        Transformation field
    kwargs : dict
        Options to ``grid_pull``

    Returns
    -------
    pulled_vel : ([batch], *spatial, ndim) tensor
        Velocity

    """
    kwargs.setdefault('bound', 'dft')
    kwargs.setdefault('extrapolate', True)
    dim = grid.shape[-1]
    vel = utils.movedim(vel, -1, -dim-1)
    vel_no_batch = vel.dim() == dim + 1
    grid_no_batch = grid.dim() == dim + 1
    if vel_no_batch:
        vel = vel[None]
    if grid_no_batch:
        grid = grid[None]
    vel = grid_push(vel, grid, *args, **kwargs)
    vel = utils.movedim(vel, -dim-1, -1)
    if vel_no_batch and grid_no_batch:
        vel = vel[0]
    return vel
