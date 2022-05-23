# Copyright 2022 The GPflow Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=unused-argument  # Bunch of fake functions below has unused arguments.
# pylint: disable=no-member  # PyLint struggles with TensorFlow.

from time import perf_counter
from typing import Any, Callable, Optional

import numpy as np
import pytest
import tensorflow as tf
from packaging.version import Version

from gpflow.base import AnyNDArray
from gpflow.experimental.check_shapes import check_shape as cs
from gpflow.experimental.check_shapes import (
    check_shapes,
    disable_check_shapes,
    inherit_check_shapes,
)
from gpflow.experimental.check_shapes.exceptions import ShapeMismatchError


def test_check_shapes__numpy() -> None:
    @check_shapes(
        "a: [d1, d2]",
        "b: [d1, d3] if b is not None",
        "return: [d2, d3]",
    )
    def f(a: AnyNDArray, b: AnyNDArray) -> AnyNDArray:
        return np.zeros((3, 4))

    f(np.zeros((2, 3)), np.zeros((2, 4)))  # Don't crash...


def test_check_shapes__tensorflow() -> None:
    @check_shapes(
        "a: [d1, d2]",
        "b: [d1, d3] if b is not None",
        "return: [d2, d3]",
    )
    def f(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        return tf.zeros((3, 4))

    f(tf.zeros((2, 3)), tf.zeros((2, 4)))  # Don't crash...


def test_check_shapes__tensorflow__keras() -> None:
    # pylint: disable=arguments-differ,abstract-method,no-value-for-parameter,unexpected-keyword-arg

    class SuperLayer(tf.keras.layers.Layer):
        def __init__(self) -> None:
            super().__init__()
            self._b = tf.Variable(0.0)

        @check_shapes(
            "x: [batch, input_dim]",
            "y: [batch, 1]",
            "return: [batch, input_dim]",
        )
        def call(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
            return x + y + self._b

    class SubLayer(SuperLayer):
        @inherit_check_shapes
        def call(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
            return x - y + self._b

    class MyModel(tf.keras.Model):
        def __init__(self, join: SuperLayer) -> None:
            super().__init__()
            self._join = join

        @check_shapes(
            "xy: [batch, input_dim_plus_one]",
            "return: [batch, input_dim]",
        )
        def call(self, xy: tf.Tensor) -> tf.Tensor:
            x = cs(xy[:, :-1], "[batch, input_dim]")
            y = cs(xy[:, -1:], "[batch, 1]")
            return self._join(x, y)

    x = tf.ones((32, 3))
    y = tf.zeros((32, 1))
    xy = tf.concat([x, y], axis=1)
    y_bad = tf.zeros((32, 2))
    targets = tf.ones((32, 3))

    def test_layer(join: SuperLayer) -> None:
        join(x, y)

        with pytest.raises(ShapeMismatchError):
            join(x, y_bad)

        model = MyModel(join)
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.25),
            loss="mean_squared_error",
        )
        model.fit(x=xy, y=targets)

    test_layer(SuperLayer())
    test_layer(SubLayer())


_Err = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
_Loss = Callable[[], tf.Tensor]

_IdWrapper = lambda x: x
_TfFunction = tf.function
_ShapedTfFunctionErr = tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.float64),
        tf.TensorSpec(shape=[], dtype=tf.float64),
    ]
)

_ShapedTfFunctionLoss = tf.function(input_signature=[])
_UnshapedTfFunctionErr = tf.function(
    input_signature=[
        tf.TensorSpec(shape=None, dtype=tf.float64),
        tf.TensorSpec(shape=None, dtype=tf.float64),
    ]
)
_UnshapedTfFunctionLoss = tf.function(input_signature=[])
_RelaxedTfFunction = tf.function(experimental_relax_shapes=True)

_NoneShape = None
_TargetShape = tf.TensorShape([])
_VShape = tf.TensorShape([50])
_UnknownShape = tf.TensorShape(None)


@pytest.mark.parametrize(
    "err_wrapper,loss_wrapper",
    [
        (_IdWrapper, _IdWrapper),
        (_TfFunction, _TfFunction),
        (_ShapedTfFunctionErr, _ShapedTfFunctionLoss),
        (_UnshapedTfFunctionErr, _UnshapedTfFunctionLoss),
        (_RelaxedTfFunction, _RelaxedTfFunction),
    ],
)
@pytest.mark.parametrize("target_shape", [_NoneShape, _TargetShape, _UnknownShape])
@pytest.mark.parametrize("v_shape", [_NoneShape, _VShape, _UnknownShape])
def test_check_shapes__tensorflow_compilation(
    err_wrapper: Callable[[_Err], _Err],
    loss_wrapper: Callable[[_Loss], _Loss],
    target_shape: Optional[tf.TensorShape],
    v_shape: Optional[tf.TensorShape],
) -> None:
    # Yeah, this test seems to be pushing the limits of TensorFlow compilation (which is probably
    # good), but a bunch of this is fragile.
    tf_version = Version(tf.__version__)

    if (target_shape is _UnknownShape) or (v_shape is _UnknownShape):
        if (err_wrapper is _TfFunction) or (err_wrapper is _RelaxedTfFunction):
            if Version("2.7.0") <= tf_version < Version("2.8.0"):
                pytest.skip("TensorFlow 2.7.* segfaults when trying to compile this.")
            if Version("2.8.0") <= tf_version < Version("2.9.0"):
                pytest.skip("TensorFlow 2.8.* is missing a TraceType(?) when trying compile this.")

    # See: https://github.com/tensorflow/tensorflow/issues/56414
    if err_wrapper is _RelaxedTfFunction:
        if Version("2.9.0") <= tf_version < Version("2.10.0"):
            err_wrapper = _TfFunction

    if Version(tf.__version__) < Version("2.5.0"):
        # TensorFlow <= 2.4.0 doesn't like the optional `z` argument:

        class SqErr:
            @check_shapes(
                "x: [broadcast n...]",
                "y: [broadcast n...]",
                "return: [n...]",
            )
            def __call__(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
                return (x - y) ** 2

    else:

        class SqErr:  # type: ignore[no-redef]
            @check_shapes(
                "x: [broadcast n...]",
                "y: [broadcast n...]",
                "z: [broadcast n...]",
                "return: [n...]",
            )
            def __call__(
                self, x: tf.Tensor, y: tf.Tensor, z: Optional[tf.Tensor] = None
            ) -> tf.Tensor:
                # z only declared to test the case of `None` arguments.
                return (x - y) ** 2

    sq_err = err_wrapper(SqErr())

    dtype = np.float64
    target = tf.Variable(0.5, dtype=dtype, shape=target_shape)
    v = tf.Variable(np.linspace(0.0, 1.0), dtype=dtype, shape=v_shape)

    @loss_wrapper
    @check_shapes(
        "return: [1]",
    )
    def loss() -> tf.Tensor:
        # keepdims is just to add an extra dimension to make the check more interesting.
        return tf.reduce_sum(sq_err(v, target), keepdims=True)

    optimiser = tf.keras.optimizers.SGD(learning_rate=0.25)
    for _ in range(10):
        optimiser.minimize(loss, var_list=[v])

    np.testing.assert_allclose(target, v.numpy(), atol=0.01)


@pytest.mark.parametrize("func_wrapper", [lambda x: x, tf.function], ids=["none", "tf.function"])
def test_check_shapes__disable__speed(func_wrapper: Callable[[Any], Any]) -> None:
    if func_wrapper is tf.function:
        pytest.skip(
            "This test is super flaky with tf.function, because the overhead of compiling"
            " seems to dominate any difference caused by check_shapes. However we probably"
            " do want some kind of test of the speed with tf.function, so we keep this"
            " skipped test around to remind us."
        )

    x = tf.zeros((3, 4, 5))
    y = tf.ones((3, 4, 5))

    def time_no_checks() -> float:
        before = perf_counter()

        def f(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
            return a + b

        f = func_wrapper(f)
        for _ in range(10):
            f(x, y)

        after = perf_counter()
        return after - before

    def time_disabled_checks() -> float:
        with disable_check_shapes():
            before = perf_counter()

            @check_shapes(
                "a: [d...]",
                "b: [d...]",
                "return: [d...]",
            )
            def f(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
                return a + b

            f = func_wrapper(f)
            for _ in range(10):
                f(x, y)

            after = perf_counter()
            return after - before

    def time_with_checks() -> float:
        before = perf_counter()

        @check_shapes(
            "a: [d...]",
            "b: [d...]",
            "return: [d...]",
        )
        def f(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
            return a + b

        f = func_wrapper(f)
        for _ in range(10):
            f(x, y)

        after = perf_counter()
        return after - before

    time_no_checks()  # Warm-up.
    t_no_checks = time_no_checks()

    time_disabled_checks()  # Warm-up.
    t_disabled_checks = time_disabled_checks()

    time_with_checks()  # Warm-up.
    t_with_checks = time_with_checks()

    assert t_no_checks < t_with_checks
    assert t_disabled_checks < t_with_checks
