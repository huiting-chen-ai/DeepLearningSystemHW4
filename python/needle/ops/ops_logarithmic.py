from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        log_sum_exp_op = LogSumExp(axes=(1,))
        return Z - log_sum_exp_op.compute(Z).reshape((-1, 1))
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        log_softmax_output = node
        softmax = exp(log_softmax_output)
        sum_out_grad = summation(out_grad, axes=(1,))
        sum_out_grad = reshape(sum_out_grad, (Z.shape[0], 1))
        sum_out_grad = broadcast_to(sum_out_grad, Z.shape)
        return out_grad - multiply(softmax, sum_out_grad)
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # M = array_api.max(Z, axis=self.axes, keepdims=True)
        M = Z.max(axis=self.axes, keepdims=True)
        r = array_api.log(array_api.sum(array_api.exp(Z-M.broadcast_to(Z.shape)), axis=self.axes, keepdims=True))+M
        new_shape = list(r.shape)
        if isinstance(self.axes, int):
            new_shape.pop(self.axes)
        else:
            for ax in reversed(self.axes):
                new_shape.pop(ax)
        return r.reshape(new_shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        M = array_api.max(Z.realize_cached_data(), axis=self.axes, keepdims=True)
        Z = Z-M
        expZ = exp(Z)
        S = summation(expZ, axes=self.axes)
        if self.axes is None:
            new_shape = [1] * len(Z.shape)
        else:
            new_shape = [1 if i in self.axes else Z.shape[i] for i in range(len(Z.shape))]
        S = reshape(S, new_shape)
        softmax = divide(expZ, broadcast_to(S, Z.shape))
        out_grad_reshaped = reshape(out_grad, new_shape)
        out_grad_broadcasted = broadcast_to(out_grad_reshaped, Z.shape)
        return multiply(out_grad_broadcasted, softmax)
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)