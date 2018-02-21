from scipy import interpolate, stats, optimize
from functools import partial
import matplotlib.pyplot as plt
import numpy as np

from bokeh.plotting import figure, show, output_notebook, ColumnDataSource
from bokeh.models import HoverTool, WheelZoomTool, ZoomInTool, ZoomOutTool, ResetTool, BoxZoomTool

# Metrics
def r2(y, residuals):
    total_var = np.std(y)**2 * y.size
    res_var = np.sum(residuals**2)
    return 1 - res_var / total_var

def mse(rs, x, y):
    return np.mean(np.concatenate([get_residuals(r, x, y) for r in rs])**2)

def mad(rs, x, y):
    return np.median(np.abs(np.concatenate([get_residuals(r, x, y) for r in rs])))

# Utils
def get_extremums(arr, lag=1):
    maximums = np.where((arr[lag:-lag] > arr[2*lag:]) & (arr[lag:-lag] > arr[:-2*lag]))[0] + lag
    minimums = np.where((arr[lag:-lag] < arr[2*lag:]) & (arr[lag:-lag] < arr[:-2*lag]))[0] + lag

    return np.sort(np.concatenate([[0], minimums, maximums, [arr.size - 1]]))

def get_residuals(r, x, y):
    r, start, end = r
    return y[start:end] - (r.slope * x[start:end] + r.intercept)

def plot_regression(x, y, smoothing=1e5, rs=None, legend_position="top_left"):
    if rs is None:
        rs, xs, ys = piecewise_linear(x, y, smoothing=smoothing, refine=True, verbose=False)
    
    breaking_points = [r[1] for r in rs] + [rs[-1][2]]
    breaking_points = refine_optimize(x, y, breaking_points)
    rs, xs, ys = piecewise_linear(x, y, breaking_points=breaking_points, refine=True, verbose=False)
        
    r2_full = r2(y, np.concatenate([get_residuals(r, x, y) for r in rs]))
    title = "Smoothing: {:.0f}. R^2: {:.3f}. MSE: {:.3f}. MAD: {:.3f}. #Segments: {}.".format(
        smoothing, r2_full, mse(rs, x, y), mad(rs, x, y), len(rs))
    output_notebook(hide_banner=True)

    source = ColumnDataSource(data=dict(
        x = xs[:-1], y = [y[0] for y in ys],
        x_line = list(zip(xs[:-1], xs[1:])), y_line = ys,
        mse = [np.mean(get_residuals(r, x, y)**2) for r in rs],
        mad = [np.median(np.abs(get_residuals(r, x, y))) for r in rs],
        r2 = [r2(y[r[1]:r[2]], get_residuals(r, x, y)) for r in rs]
    ))
    hover = HoverTool(tooltips=[
        ("index", "$index"),
        ("(x,y)", "($x, $y)"),
        ("R^2", "@r2"),
        ("MSE", "@mse"),
        ("MAD", "@mad")
    ], names=["points", "segments"])

    p = figure(title=title, x_axis_label='Day index', y_axis_label='Price', tools=[hover])
    for tool in [WheelZoomTool, ZoomInTool, ZoomOutTool, ResetTool, BoxZoomTool]:
        p.add_tools(tool())

    p.line(x, y, legend="Stock data", color="green", line_width=1, alpha=0.7)
    # p.line(xs, [y[0] for y in ys] + [ys[-1][1]], legend="Segments", line_width=3);

    p.multi_line('x_line', 'y_line', legend="Segments", name="segments", line_width=3, source=source);
    p.circle('x', 'y', legend="Inflection points", name="points", size=10, source=source)
    p.legend.location = legend_position
    p.legend.click_policy="hide"
    p.toolbar.logo = None

    show(p)
    return rs, xs, ys

# Refine
def merge_regressions(x, y, rs, id):
    rs = list(rs)
    r1, s1, e1 = rs[id]
    r2, s2, e2 = rs[id + 1]
    res_before = np.abs(np.r_[get_residuals(rs[id], x, y), get_residuals(rs[id + 1], x, y)])
    
    rs[id + 1] = stats.linregress(x[s1:e2], y[s1:e2]), s1, e2
    res_after = np.abs(get_residuals(rs[id + 1], x, y))

    pvalue = stats.distributions.norm.cdf(stats.ranksums(res_before, res_after)[0])

    return rs[:id] + rs[id + 1:], pvalue

def merge_redundant_regressions(x, y, rs, std_pvalue_min = 1e-2):
    i = 0
    while True:
        if i == len(rs) - 1:
            break

        rs2, pvalue = merge_regressions(x, y, rs, i)
        if pvalue < std_pvalue_min:
            i += 1
            continue

        if i == len(rs) - 2:
            rs = rs2
            break

        max_pvalue, max_rs = pvalue, rs2
        for j in range(i + 1, len(rs) - 1):
            rs2, pvalue = merge_regressions(x, y, rs, j)
            if pvalue < std_pvalue_min:
                break

            if pvalue > max_pvalue:
                max_pvalue, max_rs = pvalue, rs2

        rs = max_rs
    
    return rs

def refine_regression(x, y, rs):
    n_segments = len(rs)
    while True:
        rs = merge_redundant_regressions(x, y, rs)
        breaking_points = [r[1] for r in rs] + [rs[-1][2]]
        rs, xs, ys = piecewise_linear(x, y, breaking_points, refine=False)
        if len(rs) == n_segments:
            break
        
        n_segments = len(rs)
    
    return rs, xs, ys

def refine_optimize(x, y, breaking_points, max_iters=10, rel_tol=0.1, verbose=False):
    def opt_iter(point, id, x, y, breaking_points):
        breaking_points[id] = int(round(point))
        return mse(piecewise_linear(x, y, breaking_points=breaking_points[(id-1):(id+2)])[0], x, y)
    
    penalty = np.inf
    for it in range(max_iters):
        breaking_points = list(breaking_points)
        for point_id in range(1, len(breaking_points) - 1):
            opt_func = partial(opt_iter, id=point_id, x=x, y=y, breaking_points=breaking_points)
            bounds = (breaking_points[point_id - 1] + 1, breaking_points[point_id + 1] - 1)
            breaking_points[point_id] = int(round(optimize.minimize_scalar(opt_func, bounds = bounds, 
                                                                           method='Bounded').x))
        
        penalty_prev, penalty = penalty, mse(piecewise_linear(x, y, breaking_points=breaking_points)[0], x, y)
        if verbose:
            print("Iteration {}. Penalty: {:.2f}".format(it, penalty))
        if (penalty_prev - penalty) / penalty_prev < rel_tol:
            break
    
    return breaking_points

# Piecewice linear estimation
def estimate_breaking_points(spl, x):
    breaking_points = get_extremums(spl(x))
    breaking_points2 = get_extremums(spl.derivative(2)(x))

    removed_points = [np.argmin(np.abs(breaking_points2 - p)) for p in breaking_points]
    inflection_points = breaking_points2[list(set(range(breaking_points2.size)) - set(removed_points))]

    breaking_points = np.sort(np.concatenate([breaking_points, inflection_points]))
    
    return breaking_points

def piecewise_linear(x, y, breaking_points=None, smoothing=1e5, refine=True, verbose=False):
    if breaking_points is None:
        spl = interpolate.UnivariateSpline(x, y, s=smoothing, k=3)
        breaking_points = estimate_breaking_points(spl, x)

    if verbose:
        print("Smoothing: {:.2f}".format(smoothing))
    
    rs = [(stats.linregress(x[s:e], y[s:e]), s, e) for s, e in zip(breaking_points[:-1], breaking_points[1:])]
    
    xs = x[[r[1] for r in rs] + [rs[-1][2]]]
    ys = [r.slope * x[[start, end]] + r.intercept for r, start, end in rs]

    for i in range(len(ys) - 1):
        ys[i][1] = ys[i + 1][0] = (ys[i][1] + ys[i + 1][0]) / 2
        
    if verbose:
        print("Number of segments: {}".format(len(rs)))
        
    if refine:
        if verbose:
            print("Refining...")
        res = refine_regression(x, y, rs)
        if verbose:
            print("Number of segments: {}".format(len(res[0])))
        
        return res
        
    return rs, xs, ys
    
def optimal_piecewise_linear(x, y, grid_size, granularity_penalty=0.1, smoothing_min=None, 
                             smoothing_max=None, min_points_per_segment=3, verbose=0, fast=False):
    if smoothing_max is None:
        smoothing_max = np.sum((y - np.mean(y))**2)
    
    if smoothing_min is None:
        smoothing_min = 200

    i = 0
    while True:
        spl = interpolate.UnivariateSpline(x, y, s=smoothing_min, k=3)
        breaking_points = estimate_breaking_points(spl, x)
        if np.diff(breaking_points).min() + 1 >= min_points_per_segment:
            break
            
        smoothing_min *= 10
        i += 1
        if verbose and i % 2 == 0:
            print("Estimating minimal smoothing. Step {}, smoothing {}".format(i, smoothing_min))
            
    while True:
        spl = interpolate.UnivariateSpline(x, y, s=smoothing_min / 1.1, k=3)
        breaking_points = estimate_breaking_points(spl, x)
        if np.diff(breaking_points).min() + 1 < min_points_per_segment:
            break
            
        smoothing_min /= 1.5
        i += 1
        if verbose and i % 2 == 0:
            print("Estimating minimal smoothing. Step {}, smoothing {}".format(i, smoothing_min))
        
    spl = interpolate.UnivariateSpline(x, y, s=smoothing_max, k=3)
    breaking_points = estimate_breaking_points(spl, x)
    min_point_num = len(breaking_points)

    while True:
        spl = interpolate.UnivariateSpline(x, y, s=smoothing_max / 1.2, k=3)
        breaking_points = estimate_breaking_points(spl, x)
        if len(breaking_points) != min_point_num:
            break
        
        smoothing_max /= 1.2
        
    max_granularity = len(piecewise_linear(x, y, smoothing=smoothing_min)[0])
    rs = piecewise_linear(x, y, smoothing=smoothing_max)[0]
    max_mad = mad(rs, x, y)
    
    if verbose:
        print("Minimal smoothing: {}, maximal smoothing: {:.2f}".format(smoothing_min, smoothing_max))
        print("Maximal granularity: {}, maximal MAD: {:.2f}".format(max_granularity, max_mad))
    
    def penalty(rs):
        return (1 - granularity_penalty) * mad(rs, x, y) / max_mad + \
            granularity_penalty * len(rs) / max_granularity

    smoothing_by_bp_num = dict()
    for smoothing in np.linspace(smoothing_min, smoothing_max, grid_size):
        breaking_points_num = len(piecewise_linear(x, y, smoothing=smoothing)[0]) - 1
        if breaking_points_num in smoothing_by_bp_num:
            continue
            
        if breaking_points_num == 1:
            break
            
        smoothing_by_bp_num[breaking_points_num] = smoothing
        
    min_penalty = penalty(rs)
    for smoothing in smoothing_by_bp_num.values():
        rs, xs, ys = piecewise_linear(x, y, smoothing=smoothing)
        if not fast:
            breaking_points = [r[1] for r in rs] + [rs[-1][2]]
            breaking_points = refine_optimize(x, y, breaking_points)
            rs = piecewise_linear(x, y, breaking_points=breaking_points)[0]
        p = penalty(rs)
        if p < min_penalty:
            min_penalty = p
            optimal_smoothing = smoothing
            
            if verbose:
                print("Smoothing {}. Min penalty {}".format(smoothing, min_penalty))
            
    return (smoothing_min, optimal_smoothing, smoothing_max), \
        piecewise_linear(x, y, smoothing=optimal_smoothing)