from pathlib import Path

import jax.numpy as np
import matplotlib.pyplot as plt
from gpflow.base import AnyNDArray
from gpflow.experimental.check_shapes import check_shape as cs
from gpflow.experimental.check_shapes import check_shapes
from jax import grad, jit, random
from matplotlib.axes import Axes

from .clastion.utilities import multi_set, root, to_loss_function
from .covariances import RBF
from .gpr import GPR
from .means import PolynomialMeanFunction

OUT_DIR = Path(__file__).parent.parent


# TODO:
# * What other frameworks are there for learning in JAX?
# * Multiple inputs
# * Sparse GPs
# * Variational GPs
# * Multiple outputs
# * Clastion error messages
# * Clastion inheritance / interfaces
# * Clastion multi_get/set support:
#   - dicts
#   - all elements of collections
# * Allow check_shape in @derived


@check_shapes()
def main() -> None:

    dtype = np.float64

    @check_shapes()
    def plot_model(model: GPR, name: str) -> None:
        n_rows = 3
        n_columns = 1
        plot_width = n_columns * 6.0
        plot_height = n_rows * 4.0
        _fig, (sample_ax, f_ax, y_ax) = plt.subplots(
            nrows=n_rows, ncols=n_columns, figsize=(plot_width, plot_height)
        )

        plot_x = cs(np.linspace(0.0, 10.0, num=100, dtype=dtype)[:, None], "[n_plot, 1]")
        model = model(x_predict=plot_x)

        key = random.PRNGKey(20220506)
        key, *keys = random.split(key, num=5)
        for i, k in enumerate(keys):
            plot_y = cs(
                random.multivariate_normal(k, model.f_mean[:, 0], model.f_covariance)[:, None],
                "[n_plot, 1]",
            )
            sample_ax.plot(plot_x, plot_y, label=str(i))
        sample_ax.set_title("Samples")

        @check_shapes(
            "plot_mean: [n_plot, 1]",
            "plot_full_cov: [n_plot, n_plot]",
        )
        def plot_dist(
            ax: Axes, title: str, plot_mean: AnyNDArray, plot_full_cov: AnyNDArray
        ) -> None:
            plot_cov = cs(np.diag(plot_full_cov), "[n_plot]")
            plot_std = cs(np.sqrt(plot_cov), "[n_plot]")
            plot_lower = cs(plot_mean[:, 0] - plot_std, "[n_plot]")
            plot_upper = cs(plot_mean[:, 0] + plot_std, "[n_plot]")
            (mean_line,) = ax.plot(plot_x, plot_mean)
            color = mean_line.get_color()
            ax.fill_between(plot_x[:, 0], plot_lower, plot_upper, color=color, alpha=0.3)
            ax.scatter(model.x_data, model.y_data, color=color)
            ax.set_title(title)

        plot_dist(f_ax, "f", model.f_mean, model.f_covariance)
        plot_dist(y_ax, "y", model.y_mean, model.y_covariance)

        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{name}.png")
        plt.close()

    model = GPR(
        mean_func=PolynomialMeanFunction(coeffs=[1.0, 0.0]),
        covariance_func=RBF(variance=1.0),
        noise_var=0.1,
        x_data=np.zeros((0, 1)),
        y_data=np.zeros((0, 1)),
    )
    plot_model(model, "prior")

    x1 = cs(np.array([[1.0], [2.0], [3.0]], dtype=dtype), "[n_data, 1]")
    y1 = cs(np.array([[0.0], [2.0], [1.0]], dtype=dtype), "[n_data, 1]")
    model_2 = model(x_data=x1, y_data=y1)
    plot_model(model_2, "posterior")

    loss, params = to_loss_function(
        model_2,
        [
            root.mean_func.coeffs.u,
            root.noise_var.u,
            root.covariance_func.variance.u,
            root.covariance_func.lengthscale.u,
        ],
        root.log_likelihood,
    )
    loss_grad = jit(grad(loss))

    for i in range(100):
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print(i, loss(params), params)
        param_grads = loss_grad(params)
        params = {k: v + 0.1 * param_grads[k] for k, v in params.items()}

    model_3 = multi_set(model_2, params)
    plot_model(model_3, "trained")


if __name__ == "__main__":
    main()
