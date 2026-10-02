

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def tensor(array):
    return torch.as_tensor(np.ascontiguousarray(array), dtype=torch.float64)


def design_matrix(z):
    """Explicit column of ones supplies the intercept."""
    return np.column_stack([np.ones(len(z)), z])


def ols_coefficients(a, y):
    """Ordinary least squares, written directly as w = A^+ y."""
    with torch.no_grad():
        return (torch.linalg.pinv(tensor(a)) @ tensor(y)).numpy()


def ridge_coefficients(a, y, alphas):
    """Ridge with an unpenalized intercept and the smaller equivalent system.

    Centering here analytically eliminates Ridge's intercept. This does not
    affect the separate OLS calculation, which uses A^+ y directly.
    Solving the dual system when columns exceed rows avoids a huge p-by-p
    matrix; both forms solve the same Ridge objective.
    """
    with torch.no_grad():
        z, target = tensor(a[:, 1:]), tensor(y)
        mean_z, mean_y = z.mean(dim=0), target.mean()
        z, target = z-mean_z, target-mean_y
        n, p = z.shape
        primal = p <= n
        gram = z.T @ z if primal else z @ z.T
        rhs = z.T @ target if primal else target
        identity = torch.eye(gram.shape[0], dtype=torch.float64)
        for alpha in alphas:
            solution = torch.linalg.solve(gram + n*alpha*identity, rhs)
            weights = solution if primal else z.T @ solution
            intercept = mean_y - mean_z @ weights
            yield float(alpha), torch.cat([intercept.reshape(1), weights]).numpy()


def lasso_fit(model, z, y):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(z, y)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return np.r_[model.intercept_, model.coef_], converged, int(model.n_iter_)


def load_data(path, feature_count):
    df = pd.read_csv(path)
    features = [f"x{i}" for i in range(1, feature_count + 1)]
    required = features + ["y"]
    if set(df.columns) != set(required):
        raise ValueError(f"{path}: expected exactly {required}, found {list(df.columns)}")
    values = df[required].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: missing or non-finite values")
    if np.var(values[:, -1]) == 0:
        raise ValueError(f"{path}: target is constant; R-squared is undefined")
    return values[:, :-1], values[:, -1], features


def make_folds(x, count, seed):
    """Keep identical inputs together; randomize group labels reproducibly."""
    unique, groups = np.unique(x, axis=0, return_inverse=True)
    if len(unique) == len(x):
        return list(KFold(count, shuffle=True, random_state=seed).split(x)), "shuffled KFold"
    if len(unique) < count:
        raise ValueError("Fewer unique inputs than requested folds")
    labels = np.random.default_rng(seed).permutation(len(unique))[groups]
    splits = list(GroupKFold(count).split(x, groups=labels))
    return splits, "GroupKFold (identical inputs grouped; seeded group labels)"


def summarize(scores):
    return (scores.groupby(["degree", "method", "alpha"], as_index=False)
            .agg(train_mse=("train_mse", "mean"), val_mse=("val_mse", "mean"),
                 val_mse_std=("val_mse", "std"), val_r2=("val_r2", "mean"),
                 folds=("fold", "count"), all_converged=("converged", "all")))


def plot_errors(summary, folds, variant):
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, color in [("ols", "tab:blue"), ("ridge", "tab:orange"),
                          ("lasso", "tab:green")]:
        group = summary[(summary.method == method) & summary.all_converged
                        & (summary.folds == folds)].dropna(subset=["val_mse"])
        if group.empty:
            continue
        best = group.loc[group.groupby("degree").val_mse.idxmin()].sort_values("degree")
        ax.plot(best.degree, best.val_mse, "o-", color=color, label=f"{method}: validation")
        ax.plot(best.degree, best.train_mse, "--", color=color, label=f"{method}: training")
    ax.set(xlabel="Polynomial degree", ylabel="Mean squared error",
           title=f"var{variant}: {folds}-fold CV")
    ax.set_yscale("symlog", linthresh=1e-8)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


class Results:
    """Keep scores in memory only; retries replace unfinished fold results."""
    def __init__(self, folds):
        self.folds, self.rows = folds, {}

    def add(self, degree, fold, method, alpha, weights, data, ok=True, iterations=0,
            stage="initial"):
        a_train, a_val, y_train, y_val = data
        train_pred, val_pred = a_train @ weights, a_val @ weights
        finite = np.isfinite(train_pred).all() and np.isfinite(val_pred).all()
        row = dict(degree=degree, fold=fold, method=method, alpha=float(alpha),
                   train_mse=mean_squared_error(y_train, train_pred) if finite else np.nan,
                   val_mse=mean_squared_error(y_val, val_pred) if finite else np.nan,
                   val_r2=r2_score(y_val, val_pred) if finite else np.nan,
                   converged=bool(ok and finite), iterations=int(iterations), stage=stage)
        self.rows[(degree, fold, method, float(alpha))] = row

    def summary(self):
        return summarize(pd.DataFrame(self.rows.values()))

    def eligible(self, degree=None):
        frame = self.summary()
        mask = frame.all_converged & (frame.folds == self.folds) & np.isfinite(frame.val_mse)
        if degree is not None:
            mask &= frame.degree == degree
        return frame[mask].copy()

    def best(self, degree=None):
        frame = self.eligible(degree)
        if frame.empty:
            raise RuntimeError("No fully converged finite candidate available")
        frame["method_order"] = frame.method.map({"ols": 0, "ridge": 1, "lasso": 2})
        return frame.sort_values(["val_mse", "degree", "method_order", "alpha"]).iloc[0]


def fold_data(expanded, y, indices):
    train_idx, val_idx = indices
    scaler = StandardScaler()
    z_train = scaler.fit_transform(expanded[train_idx])
    z_val = scaler.transform(expanded[val_idx])
    y_train, y_val = y[train_idx], y[val_idx]
    return design_matrix(z_train), design_matrix(z_val), y_train, y_val


def retry_competitive_lasso(args, degree, expanded, y, folds, results):
    """One bounded retry; never let incomplete fits determine the winner."""
    eligible = results.eligible(degree)
    if eligible.empty:
        return
    threshold = 1.1 * eligible.val_mse.min()
    summary = results.summary()
    candidates = summary[(summary.degree == degree) & (summary.method == "lasso")
                         & ~summary.all_converged & (summary.folds == args.folds)
                         & (summary.val_mse <= threshold)]
    for alpha in candidates.alpha:
        records = [r for r in results.rows.values()
                   if r["degree"] == degree and r["method"] == "lasso" and r["alpha"] == alpha]
        if not all(np.isfinite(r["val_mse"]) for r in records):
            continue
        print(f"    Retrying competitive Lasso alpha={alpha:g}, failed folds only", flush=True)
        for fold, indices in enumerate(folds, 1):
            old = results.rows[(degree, fold, "lasso", float(alpha))]
            if old["converged"] or old["stage"] == "retry":
                continue
            data = fold_data(expanded, y, indices)
            a_train, _, y_train, _ = data
            model = Lasso(alpha=alpha, fit_intercept=True, max_iter=args.retry_iter, tol=args.tol)
            weights, ok, iterations = lasso_fit(model, np.asfortranarray(a_train[:, 1:]), y_train)
            results.add(degree, fold, "lasso", alpha, weights, data, ok, iterations, "retry")


def stopping_update(previous_best, current, stale, min_improvement):
    significant = (not np.isfinite(previous_best)
                   or (current < previous_best and current <= previous_best * (1-min_improvement)))
    return min(previous_best, current), 0 if significant else stale + 1


def train_problem(args, variant, max_degree, inputs):
    x, y, features, test_frame, x_test = inputs
    folds, strategy = make_folds(x, args.folds, args.seed)
    if any(len(val) < 2 or np.var(y[val]) == 0 for _, val in folds):
        raise ValueError("Each validation fold needs at least two rows and a nonconstant target")
    results = Results(args.folds)
    previous_best, stale = np.inf, 0
    reason = "maximum degree reached"
    print(f"\nvar{variant}: {len(y)} rows; {strategy}; {args.folds} folds", flush=True)
    for degree in range(1, max_degree + 1):
        expanded = PolynomialFeatures(degree, include_bias=False).fit_transform(x)
        if not np.isfinite(expanded).all():
            raise ValueError("Polynomial expansion overflowed")
        print(f"  Degree {degree}/{max_degree}: {expanded.shape[1]} terms + intercept", flush=True)
        for fold, indices in enumerate(folds, 1):
            data = fold_data(expanded, y, indices)
            a_train, _, y_train, _ = data
            weights = ols_coefficients(a_train, y_train)
            results.add(degree, fold, "ols", 0.0, weights, data)
            for alpha, weights in ridge_coefficients(a_train, y_train, args.alphas):
                results.add(degree, fold, "ridge", alpha, weights, data)
            model = Lasso(fit_intercept=True, warm_start=True, max_iter=args.max_iter,
                          tol=args.tol, selection="cyclic")
            z_fortran = np.asfortranarray(a_train[:, 1:])
            failed = 0
            for alpha in sorted(args.alphas, reverse=True):
                model.set_params(alpha=alpha)
                weights, ok, iterations = lasso_fit(model, z_fortran, y_train)
                failed += int(not ok)
                results.add(degree, fold, "lasso", alpha, weights, data, ok, iterations)
            print(f"    Fold {fold}/{args.folds}; unfinished Lasso fits: {failed}", flush=True)
        retry_competitive_lasso(args, degree, expanded, y, folds, results)
        degree_best = results.best(degree)
        previous_best, stale = stopping_update(previous_best, float(degree_best.val_mse),
                                               stale, args.min_improvement)
        print(f"  Best this degree: {degree_best.method}, alpha={degree_best.alpha:g}, "
              f"MSE={degree_best.val_mse:.6g}, R2={degree_best.val_r2:.6g}; "
              f"degrees without >={100*args.min_improvement:g}% improvement: "
              f"{stale}/{args.patience}", flush=True)
        if stale >= args.patience:
            reason = f"{args.patience} successive degrees without the required relative improvement"
            print(f"  Stopping degree search: {reason}", flush=True)
            break
    last_degree = degree

    # A single local edge check, not another full grid search.
    best = results.best()
    extra_alpha = None
    if best.method != "ols":
        if best.alpha == min(args.alphas):
            extra_alpha = float(best.alpha) / 10
        elif best.alpha == max(args.alphas):
            extra_alpha = float(best.alpha) * 10
    if extra_alpha is not None:
        degree, method = int(best.degree), str(best.method)
        print(f"  One grid-edge check: degree={degree}, {method}, alpha={extra_alpha:g}", flush=True)
        expanded = PolynomialFeatures(degree, include_bias=False).fit_transform(x)
        for fold, indices in enumerate(folds, 1):
            data = fold_data(expanded, y, indices)
            a_train, _, y_train, _ = data
            if method == "ridge":
                _, weights = next(ridge_coefficients(a_train, y_train, [extra_alpha]))
                ok, iterations = True, 0
            else:
                model = Lasso(alpha=extra_alpha, fit_intercept=True,
                              max_iter=args.max_iter, tol=args.tol)
                weights, ok, iterations = lasso_fit(model, np.asfortranarray(a_train[:, 1:]), y_train)
            results.add(degree, fold, method, extra_alpha, weights, data, ok, iterations, "edge_check")
        retry_competitive_lasso(args, degree, expanded, y, folds, results)

    best = results.best()
    degree, method, alpha = int(best.degree), str(best.method), float(best.alpha)
    print(f"Selected var{variant}: degree={degree}, {method}, alpha={alpha:g}, "
          f"CV MSE={best.val_mse:.6g}, CV R2={best.val_r2:.6g}", flush=True)
    poly = PolynomialFeatures(degree, include_bias=False)
    scaler = StandardScaler()
    z = scaler.fit_transform(poly.fit_transform(x))
    a = design_matrix(z)
    a_test = design_matrix(scaler.transform(poly.transform(x_test)))
    if not np.isfinite(a_test).all():
        raise ValueError("Nonfinite test polynomial values")
    if method == "ols":
        weights = ols_coefficients(a, y)
    elif method == "ridge":
        _, weights = next(ridge_coefficients(a, y, [alpha]))
    else:
        model = Lasso(alpha=alpha, fit_intercept=True, max_iter=args.max_iter,
                      tol=args.tol, warm_start=True)
        weights, ok, _ = lasso_fit(model, np.asfortranarray(z), y)
        if not ok:
            model.set_params(max_iter=args.retry_iter)
            weights, ok, _ = lasso_fit(model, np.asfortranarray(z), y)
        if not ok:
            raise RuntimeError("Final Lasso did not converge; predictions were not saved")
    prediction = a_test @ weights
    if not np.isfinite(prediction).all():
        raise RuntimeError("Final predictions are nonfinite")
    output = pd.DataFrame({"y": prediction})
    output["y"] = prediction
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.roll}_pred_var{variant}.csv"
    output.to_csv(output_path, index=False)
    print(f"Saved {len(output)} predictions: {output_path.resolve()}", flush=True)
    if not args.no_plot:
        plot_errors(results.summary(), args.folds, variant)
    return dict(problem=f"var{variant}", degree=degree, method=method, alpha=alpha,
                validation_MSE=float(best.val_mse), validation_R2=float(best.val_r2))


def load_inputs(args, variant):
    """Validate both train and test inputs before spending time on CV.

    Test targets (including placeholders) are ignored. Additional test columns
    such as IDs are ignored; the output contains only predicted y values.
    """
    x, y, features = load_data(args.data_dir / f"{args.roll}_train_var{variant}.csv",
                               6 if variant == 1 else 3)
    path = args.data_dir / f"{args.roll}_test_var{variant}.csv"
    test = pd.read_csv(path)
    missing = set(features) - set(test.columns)
    if missing:
        raise ValueError(f"{path}: missing feature columns {sorted(missing)}")
    x_test = test[features].to_numpy(dtype=np.float64)
    if len(x_test) == 0 or not np.isfinite(x_test).all():
        raise ValueError(f"{path}: empty test file or nonfinite inputs")
    return x, y, features, test, x_test


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("predictions"))
    parser.add_argument("--y-only", action="store_true", help="Compatibility flag: output always contains only y")
    parser.add_argument("--no-plot", action="store_true", help="Disable plot display")
    parser.add_argument("--roll", default="BT2024190")
    parser.add_argument("--problem", choices=["both", "1", "2"], default="both")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-degree-var1", type=int, default=10)
    parser.add_argument("--max-degree-var2", type=int, default=20)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.001, 0.01, 0.1])
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--retry-iter", type=int, default=10000)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-improvement", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.folds < 2 or min(args.max_iter, args.patience, args.threads) < 1:
        parser.error("Require folds >= 2 and positive iteration, patience, and thread settings")
    if args.retry_iter < args.max_iter or not np.isfinite(args.tol) or args.tol <= 0:
        parser.error("Require retry-iter >= max-iter and a positive finite tolerance")
    if not 0 <= args.min_improvement < 1:
        parser.error("min-improvement must lie in [0, 1)")
    if not 1 <= args.max_degree_var1 <= 10 or not 1 <= args.max_degree_var2 <= 20:
        parser.error("Degree bounds: var1 1..10; var2 1..20")
    if any(not np.isfinite(a) or a <= 0 for a in args.alphas):
        parser.error("Regularization strengths must be positive and finite")
    args.alphas = sorted(set(args.alphas))
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    chosen = [(v, d) for v, d in [(1, args.max_degree_var1), (2, args.max_degree_var2)]
              if args.problem in ("both", str(v))]
    # Check every input file up front, before starting any expensive training.
    inputs = {v: load_inputs(args, v) for v, _ in chosen}
    final_scores = [train_problem(args, v, maximum, inputs[v]) for v, maximum in chosen]
    print("\nSelected configurations: mean cross-validation scores (not test scores)")
    print(pd.DataFrame(final_scores).to_string(index=False, float_format=lambda x: f"{x:.8g}"))
    if not args.no_plot:
        print("\nDisplaying plots. Close the plot windows to exit; plots are not saved.", flush=True)
        plt.show()



if __name__ == "__main__":
    main()
