import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from abc import ABC, abstractmethod

class AbstractDTW(ABC):
    @abstractmethod
    def minimum(self, *args):
        pass

    def __call__(self, prediction, target):
        D = distance_matrix(prediction, target)
        # wlog: H >= W
        if D.shape[0] < D.shape[1]:
            D = D.T    
        H, W = D.shape

        rows = []
        for row in range(H):
            rows.append( pad_inf(D[row], row, H-row-1) )

        model_matrix = jnp.stack(rows, axis=1)

        init = (
            pad_inf(model_matrix[0], 1, 0),
            pad_inf(model_matrix[1] + model_matrix[0, 0], 1, 0)
        )

        def scan_step(carry, current_antidiagonal):
            two_ago, one_ago = carry

            diagonal = two_ago[:-1]
            right    = one_ago[:-1]
            down     = one_ago[1:]
            best     = self.minimum(jnp.stack([diagonal, right, down], axis=-1))

            next_row = best + current_antidiagonal
            next_row = pad_inf(next_row, 1, 0)

            return (one_ago, next_row), next_row

        # Manual unrolling:
        # carry = init
        # for i, row in enumerate(model_matrix[2:]):
        #     carry, y = scan_step(carry, row)

        carry, ys = jax.lax.scan(scan_step, init, model_matrix[2:], unroll=4)
        return carry[1][-1]


class DTW(AbstractDTW):
    """
    SoftDTW as proposed in the paper "Dynamic programming algorithm optimization for spoken word recognition"
    by Hiroaki Sakoe and Seibi Chiba (https://arxiv.org/abs/1703.01541)

    Expects inputs of the shape [T, D], where T is the time dimension
    and D is the feature dimension.
    """
    __name__ = 'DTW'

    def minimum(self, args):
        return jnp.min(args, axis=-1)


class SoftDTW(AbstractDTW):
    """
    SoftDTW as proposed in the paper "Soft-DTW: a Differentiable Loss Function for Time-Series"
    by Marco Cuturi and Mathieu Blondel (https://arxiv.org/abs/1703.01541)

    Expects inputs of the shape [T, D], where T is the time dimension
    and D is the feature dimension.
    """
    __name__ = 'SoftDTW'

    def __init__(self, gamma=1.0):
        assert gamma > 0, "Gamma needs to be positive."
        self.gamma = gamma
        self.__name__ = f'SoftDTW({self.gamma})'
        self.minimum_impl = self.make_softmin(gamma)

    def make_softmin(self, gamma, custom_grad=True):
        """
        We need to manually define the gradient of softmin
        to ensure (1) numerical stability and (2) prevent nans from
        propagating over valid values.
        """
        def softmin_raw(array):
            return -gamma * logsumexp(array / -gamma, axis=-1)
        
        if not custom_grad:
            return softmin_raw

        softmin = jax.custom_vjp(softmin_raw)

        def softmin_fwd(array):
            return softmin(array), (array / -gamma, )

        def softmin_bwd(res, g):
            scaled_array, = res
            grad = jnp.where(jnp.isinf(scaled_array),
                jnp.zeros(scaled_array.shape),
                jax.nn.softmax(scaled_array) * jnp.expand_dims(g, 1)
            )
            return grad,

        softmin.defvjp(softmin_fwd, softmin_bwd)
        return softmin

    def minimum(self, args):
        return self.minimum_impl(args)


# Utility functions
def distance_matrix(a, b):
    has_features = len(a.shape) > 1
    a = jnp.expand_dims(a, axis=1)
    b = jnp.expand_dims(b, axis=0)
    D = jnp.square(a - b)
    if has_features:
        D = jnp.sum(D, axis=-1)
    return D


def pad_inf(inp, before, after):
    return jnp.pad(inp, (before, after), constant_values=jnp.inf)
