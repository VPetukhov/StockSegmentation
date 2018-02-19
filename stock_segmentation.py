from scipy import optimize, interpolate
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

# Metrics
def r2(rs, x, y):
    total_var = np.std(y)**2 * y.size
    res_var = np.sum(np.concatenate([get_residuals(r, x, y) for r in rs])**2)
    return 1 - res_var / total_var

def mse(rs, x, y):
    return np.mean(np.concatenate([get_residuals(r, x, y) for r in rs])**2)

# Utils
def get_extremums(arr, lag=1):
    maximums = np.where((arr[lag:-lag] > arr[2*lag:]) & (arr[lag:-lag] > arr[:-2*lag]))[0] + lag
    minimums = np.where((arr[lag:-lag] < arr[2*lag:]) & (arr[lag:-lag] < arr[:-2*lag]))[0] + lag

    return np.sort(np.concatenate([[0], minimums, maximums, [arr.size - 1]]))

def get_residuals(r, x, y):
    r, start, end = r
    return y[start:end] - (r.slope * x[start:end] + r.intercept)

def plot_regression(x, y, smoothing):
    rs, xs, ys = piecewise_linear(x, y, min_smoothing=smoothing, refine=True, verbose=False)
    plt.plot(x, y, color="green", alpha=0.5);
    plt.plot(xs, [y[0] for y in ys] + [ys[-1][1]]);
    plt.scatter(xs, [y[0] for y in ys] + [ys[-1][1]]);
    plt.title("R^2: {:.3f}. Mean squared error: {:.2f}".format(r2(rs, x, y), mse(rs, x, y)))

# Refine
def merge_regressions(x, y, rs, id):
    rs = list(rs)
    r1, s1, e1 = rs[id]
    r2, s2, e2 = rs[id + 1]
    res_before = np.abs(np.r_[get_residuals(rs[id], x, y), get_residuals(rs[id + 1], x, y)])
    
    rs[id + 1] = sp.stats.linregress(x[s1:e2], y[s1:e2]), s1, e2
    res_after = np.abs(get_residuals(rs[id + 1], x, y))

    pvalue = sp.stats.mannwhitneyu(res_before, res_after, alternative="less")[1]

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

# Piecewice linear estimation
def estimate_breaking_points(spl, x):
    breaking_points = get_extremums(spl(x))
    breaking_points2 = get_extremums(spl.derivative(2)(x))

    removed_points = [np.argmin(np.abs(breaking_points2 - p)) for p in breaking_points]
    inflection_points = breaking_points2[list(set(range(breaking_points2.size)) - set(removed_points))]

    breaking_points = np.sort(np.concatenate([breaking_points, inflection_points]))
    
    return breaking_points

def piecewise_linear(x, y, breaking_points=None, min_smoothing=1e5, refine=True, verbose=False):
    if breaking_points is None:
        spl = interpolate.UnivariateSpline(x, y, s=min_smoothing, k=3)
        breaking_points = estimate_breaking_points(spl, x)

    if verbose:
        print("Smoothing: {:.2f}".format(smoothing))
    
    rs = [(sp.stats.linregress(x[s:e], y[s:e]), s, e) for s, e in zip(breaking_points[:-1], breaking_points[1:])]
    
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
                             smoothing_max=None, min_points_per_segment=5, verbose=0):
    if smoothing_max is None:
        smoothing_max = np.sum((y - np.mean(y))**2)
    
    if smoothing_min is None:
        smoothing_min = 2

    points_per_segment = 1
    while True:
        spl = interpolate.UnivariateSpline(x, y, s=smoothing_min, k=3)
        breaking_points = estimate_breaking_points(spl, x)
        points_per_segment = np.diff(breaking_points).min()
        if points_per_segment >= min_points_per_segment:
            break
        smoothing_min *= 2
        
    max_granularity = len(piecewise_linear(x, y, min_smoothing=smoothing_min)[0])
    rs = piecewise_linear(x, y, min_smoothing=smoothing_max)[0]
    max_mse = mse(rs, x, y)
    
    if verbose:
        print("Minimal smoothing: {}, maximal smoothing: {:.2f}".format(smoothing_min, smoothing_max))
    
    def penalty(rs):
        return (1 - granularity_penalty) * mse(rs, x, y) / max_mse + \
            granularity_penalty * len(rs) / max_granularity
    
    min_penalty = penalty(rs)
    optimal_smoothing = smoothing_max
    
    for i, smoothing in enumerate(np.linspace(smoothing_min, smoothing_max, grid_size)):
        rs, xs, ys = piecewise_linear(x, y, min_smoothing=smoothing)
        p = penalty(rs)
        if p < min_penalty:
            min_penalty = p
            optimal_smoothing = smoothing
            
            if verbose:
                print("Step {}. Min penalty {}".format(i, min_penalty))
            
    return (smoothing_min, optimal_smoothing, smoothing_max), \
        piecewise_linear(x, y, min_smoothing=optimal_smoothing)