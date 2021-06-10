from nitorch.core import utils, py, math, constants
from nitorch import spatial
import torch


def mse(moving, fixed, lam=1, dim=None, grad=True, hess=True):
    """Mean-squared error loss for optimisation-based registration.

    Parameters
    ----------
    moving : ([B], K, *spatial) tensor
        Moving image
    fixed : ([B], K, *spatial) tensor
        Fixed image
    lam : float or ([B], K|1, [*spatial]) tensor_like
        Gaussian noise precision.
    dim : int, default=`fixed.dim() - 1`
        Number of spatial dimensions
    grad : bool, default=True
        Compute and return gradient
    hess : bool, default=True
        Compute and return Hessian

    Returns
    -------
    ll : () tensor
        Negative log-likelihood
    g : (..., K, *spatial) tensor, optional
        Gradient with respect to the moving imaged
    h : (..., K, *spatial) tensor, optional
        (Diagonal) Hessian with respect to the moving image

    """
    fixed, moving, lam = utils.to_max_backend(fixed, moving, lam)
    dim = dim or (fixed.dim() - 1)
    if lam.dim() <= 2:
        if lam.dim() == 0:
            lam = lam.flatten()
        lam = utils.unsqueeze(lam, -1, dim)  # pad spatial dimensions

    ll = (moving - fixed).square_().mul_(lam).sum()
    out = [ll]
    if grad:
        g = (moving - fixed).mul_(lam)
        out.append(g)
    if hess:
        h = lam
        out.append(h)
    return tuple(out) if len(out) > 1 else out[0]


def cat(moving, fixed, dim=None, acceleration=0, grad=True, hess=True):
    """Categorical loss for optimisation-based registration.

    Parameters
    ----------
    moving : (..., K, *spatial) tensor
        Moving image of log-probabilities (pre-softmax).
        The background class should be omitted.
    fixed : (..., K, *spatial) tensor
        Fixed image of probabilities
    dim : int, default=`fixed.dim() - 1`
        Number of spatial dimensions.
    acceleration : (0..1), default=0
        Weight the contributions of the true Hessian and Boehning bound.
        The Hessian is a weighted sum between the Boehning bound and the
        Gauss-Newton Hessian: H = a * H_gn + (1-a) * H_bnd
        The Gauss-Newton Hessian is less stable but allows larger jumps
        than the Boehning Hessian, so increasing `a` can lead to an
        accelerated convergence.
    grad : bool, default=True
        Compute and return gradient
    hess : bool, default=True
        Compute and return Hessian

    Returns
    -------
    ll : () tensor
        Negative log-likelihood
    g : (..., K, *spatial) tensor, optional
        Gradient with respect to the moving image
    h : (..., K*(K+1)//2, *spatial) tensor, optional
        Hessian with respect to the moving image.
        Its spatial dimensions are singleton when `acceleration == 0`.

    """
    fixed, moving = utils.to_max_backend(fixed, moving)
    dim = dim or (fixed.dim() - 1)
    nc = moving.shape[-dim-1]                               # nb classes - bck
    fixed = utils.slice_tensor(fixed, slice(nc), -dim-1)    # remove bkg class

    # log likelihood
    ll = moving*fixed
    ll -= math.logsumexp(moving, dim=-dim-1, implicit=True)            # implicit lse
    ll = ll.sum().neg()
    out = [ll]

    if grad or (hess and acceleration > 0):
        # implicit softmax
        moving = math.softmax(moving, dim=-dim-1, implicit=True)

    # gradient
    if grad:
        g = moving - fixed
        out.append(g)

    # hessian
    if hess:
        # compute true Hessian
        def allocate_h():
            nch = nc*(nc+1)//2
            shape = list(moving.shape)
            shape[-dim-1] = nch
            h = moving.new_empty(shape)
            return h

        h = None
        if acceleration > 0:
            h = allocate_h()
            h_diag = utils.slice_tensor(h, slice(nc), -dim-1)
            h_diag.copy_(moving*(1 - moving))
            # off diagonal elements
            c = 0
            for i in range(nc):
                pi = utils.slice_tensor(moving, i, -dim-1)
                for j in range(i+1, nc):
                    pj = utils.slice_tensor(moving, j, -dim-1)
                    out = utils.slice_tensor(h, nc+c, -dim-1)
                    out.copy_(pi*pj).neg_()
                    c += 1

        # compute Boehning Hessian
        def allocate_hb():
            nch = nc*(nc+1)//2
            h = moving.new_empty(nch)
            return h

        if acceleration < 1:
            hb = allocate_hb()
            hb[:nc] = 1 - 1/(nc+1)
            hb[nc:] = -1/(nc+1)
            hb = utils.unsqueeze(hb, -1, dim)
            hb.div_(2)
            if acceleration > 0:
                hb.mul_(1-acceleration)
                h.mul_(acceleration).add_(hb)
            else:
                h = hb

        out.append(h)

    return tuple(out) if len(out) > 1 else out[0]


class OptimizationLoss:
    """Base class for losses used in 'old school' optimisation-based stuff."""

    def __init__(self):
        """Specify parameters"""
        pass

    def loss(self, *args, **kwargs):
        """Returns the loss (to be minimized) only"""
        raise NotImplementedError

    def loss_grad(self, *args, **kwargs):
        """Returns the loss (to be minimized) and its gradient
        with respect to the *first* argument."""
        raise NotImplementedError

    def loss_grad_hess(self, *args, **kwargs):
        """Returns the loss (to be minimized) and its gradient
        and hessian with respect to the *first* argument.

        In general, we expect a block-diagonal positive-definite
        approximation of the true Hessian (in general, correlations
        between spatial elements -- voxels -- are discarded).
        """
        raise NotImplementedError


class MSE(OptimizationLoss):
    """Mean-squared error"""

    def __init__(self, lam=1, dim=None):
        """

        Parameters
        ----------
        lam : (K|1,) tensor_like
            Precision
        dim : int, default=1fixed.dim() - 1`
            Number of spatial dimensions
        """
        super().__init__()
        self.lam = lam
        self.dim = dim

    def loss(self, moving, fixed, **kwargs):
        """Compute the [weighted] mse (* 0.5)

        Parameters
        ----------
        moving : ([B], K, *spatial) tensor
            Moving image
        fixed : ([B], K, *spatial) tensor
            Fixed image

        Returns
        -------
        ll : () tensor
            Loss
        grad : ([B], K, *spatial) tensor
            Gradient

        """
        lam = kwargs.get('lam', self.lam)
        dim = kwargs.get('dim', self.dim)
        return mse(moving, fixed, lam=lam, dim=dim, grad=False, hess=False)

    def loss_grad(self, moving, fixed, **kwargs):
        """Compute the [weighted] mse (* 0.5)

        Parameters
        ----------
        moving : ([B], K, *spatial) tensor
            Moving image
        fixed : ([B], K, *spatial) tensor
            Fixed image

        Returns
        -------
        ll : () tensor
            Loss

        """
        lam = kwargs.get('lam', self.lam)
        dim = kwargs.get('dim', self.dim)
        return mse(moving, fixed, lam=lam, dim=dim, hess=False)

    def loss_grad_hess(self, moving, fixed, **kwargs):
        """Compute the [weighted] mse (* 0.5)

        Parameters
        ----------
        moving : ([B], K, *spatial) tensor
            Moving image
        fixed : ([B], K, *spatial) tensor
            Fixed image

        Returns
        -------
        ll : () tensor
            Loss
        grad : ([B], K, *spatial) tensor
            Gradient
        hess : ([B], K, *spatial) tensor
            Diagonal Hessian

        """
        lam = kwargs.get('lam', self.lam)
        dim = kwargs.get('dim', self.dim)
        return mse(moving, fixed, lam=lam, dim=dim)


class Cat(OptimizationLoss):
    """Categorical cross-entropy"""

    def __init__(self, acceleration=0, dim=None):
        """

        Parameters
        ----------
        acceleration : (0..1) float
            Acceleration
        dim : int, default=1fixed.dim() - 1`
            Number of spatial dimensions
        """
        super().__init__()
        self.acceleration = acceleration
        self.dim = dim

    def loss(self, moving, fixed, **kwargs):
        """Compute the [weighted] mse (* 0.5)

        Parameters
        ----------
        moving : (..., K, *spatial) tensor
            Moving image
        fixed : (..., K, *spatial) tensor
            Fixed image

        Returns
        -------
        ll : () tensor
            Loss

        """
        acceleration = kwargs.get('acceleration', self.acceleration)
        dim = kwargs.get('dim', self.dim)
        return cat(moving, fixed, acceleration=acceleration, dim=dim,
                   grad=False, hess=False)

    def loss_grad(self, moving, fixed, **kwargs):
        """Compute the [weighted] mse (* 0.5)

        Parameters
        ----------
        moving : (..., K, *spatial) tensor
            Moving image
        fixed : (..., K, *spatial) tensor
            Fixed image

        Returns
        -------
        ll : () tensor
            Loss
        g : (..., K, *spatial) tensor, optional
            Gradient with respect to the moving image

        """
        acceleration = kwargs.get('acceleration', self.acceleration)
        dim = kwargs.get('dim', self.dim)
        return cat(moving, fixed, acceleration=acceleration, dim=dim,
                   hess=False)

    def loss_grad_hess(self, moving, fixed, **kwargs):
        """Compute the [weighted] mse (* 0.5)

        Parameters
        ----------
        moving : (..., K, *spatial) tensor
            Moving image
        fixed : (..., K, *spatial) tensor
            Fixed image

        Returns
        -------
        ll : () tensor
            Loss
        g : (..., K, *spatial) tensor, optional
            Gradient with respect to the moving image
        h : (..., K*(K+1)//2, *spatial) tensor, optional
            Hessian with respect to the moving image.
            Its spatial dimensions are singleton when `acceleration == 0`.

        """
        acceleration = kwargs.get('acceleration', self.acceleration)
        dim = kwargs.get('dim', self.dim)
        return cat(moving, fixed, acceleration=acceleration, dim=dim)



# WORK IN PROGRESS
# def nmi(fixed, moving, dim=None, bins=64, order=3, limits=None,
#         normalized='studholme'):
#     fixed, moving = utils.to_max_backend(fixed, moving)
#     dim = dim or (fixed.dim() - 1)
#
#     # compute histograms limits
#     if not isinstance(limits, dict):
#         limits = dict(fixed=limits, moving=limits)
#     limits['fixed'] = py.make_list(limits['fixed'], 2)
#     limits['moving'] = py.make_list(limits['moving'], 2)
#     if limits['fixed'][0] is None:
#         limits['fixed'][0] = math.min(fixed, dim=range(-dim, 0), keepdim=True)
#     if limits['fixed'][1] is None:
#         limits['fixed'][1] = math.max(fixed, dim=range(-dim, 0), keepdim=True)
#     if limits['moving'][0] is None:
#         limits['moving'][0] = math.min(moving, dim=range(-dim, 0), keepdim=True)
#     if limits['moving'][1] is None:
#         limits['moving'][1] = math.max(moving, dim=range(-dim, 0), keepdim=True)
#
#     def pnorm(x, dims=-1):
#         """Normalize a tensor so that it's sum across `dims` is one."""
#         dims = py.make_list(dims)
#         x = x.clamp_min_(constants.eps(x.dtype))
#         s = math.sum(x, dim=dims, keepdim=True)
#         return x/s, s
#
#     vmin = (limits['fixed'][0], limits['moving'][0])
#     vmax = (limits['fixed'][1], limits['moving'][1])
#     pxy = utils.histc2(
#         torch.stack([fixed, moving], -1), bins, vmin, vmax,
#         dim=range(-dim-1, -1), order=order, bound='zero')
#
#     # compute probabilities
#     px, sx = pnorm(pxy.sum(dim=-2))  # -> [B, C, nb_bins]
#     py, sy = pnorm(pxy.sum(dim=-1))  # -> [B, C, nb_bins]
#     pxy, sxy = pnorm(pxy, [-1, -2])
#
#     # compute entropies
#     hx = -(px * px.log()).sum(dim=-1)  # -> [B, C]
#     hy = -(py * py.log()).sum(dim=-1)  # -> [B, C]
#     hxy = -(pxy * pxy.log()).sum(dim=[-1, -2])  # -> [B, C]
#
#     # mutual information
#     mi = (hx + hy) - hxy
#     if normalized == 'studholme':
#         mi /= hxy
#     elif normalized == 'arithmetic':
#         mi /= (hx + hy)
#
#     # gradient
#     gxy = pxy.log()
#     gy = py.log()
#     if normalized == 'studholme':
#         gxy = (gxy + (1 + pxy.log()) * mi) / hxy
#         gy /= hxy
#     elif normalized == 'arithmetic':
#         gy = (gy + (1 + py.log()) * mi) / (hx + hy)
#         gxy /= (hx + hy)
#     gy *= 1/sy - py.square()
#     gxy *= 1/sxy - pxy.square()
#     gxy += gy
#     gxy = gxy.sum(dim=-2)
#     gxy = spatial.grid_pull()
#
#     # take negative
#     mi = 1 - mi
#     g = g.neg_()
#     return mi, g, h