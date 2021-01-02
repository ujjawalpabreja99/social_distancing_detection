import numpy as np
import constants
from scipy import stats


def decode_data(data, roi):
    # decode the raw data w.r.t. the defined roi
    x_min, x_max, y_min, y_max = roi
    area = (x_max - x_min) * (y_max - y_min)

    density = []
    ts_inference = []
    points_roi_all_frame = []
    inds_frame = []
    nums_ped = []

    for i_frame, t_inference, points in data:
        count_in = 0
        count_out = 0
        points_roi = []
        for pt in points:
            if x_min < pt[0] < x_max and y_min < pt[1] < y_max:
                count_in += 1
                points_roi.append(pt)
            else:
                count_out += 1
        points_roi_all_frame.append(np.array(points_roi))
        density.append(count_in / area)
        ts_inference.append(t_inference)
        inds_frame.append(i_frame)
        nums_ped.append(count_in)

    return np.array(inds_frame), np.array(ts_inference), points_roi_all_frame, np.array(density), nums_ped


def count_violation_pairs(points_all_frames, dist=2.0):
    counts = []
    for points in points_all_frames:
        pairs = find_violation(points, dist)
        counts.append(len(pairs))
    return np.array(counts)


def find_violation(points, dist=2.0):
    # a list of index pairs indicating two pedestrians who are violating social distancing
    n = len(points)
    pairs = []
    for i in np.arange(0, n, 1):
        for j in np.arange(i+1, n, 1):
            if np.linalg.norm(points[i] - points[j]) < dist:
                pairs.append((i, j))
    return pairs


def cal_min_dists_all_frame(points_all_frame):
    all_min_dists = []
    avg_min_dists = []
    min_min_dists = []
    for points in points_all_frame:
        min_dists = cal_min_dists(points)
        all_min_dists.append(min_dists)
        min_min_dists.append(min(min_dists) if len(min_dists) > 0 else None)
        avg_min_dists.append(sum(min_dists) / len(min_dists) if len(min_dists) > 0 else None)

    all_min_dists = sum(all_min_dists, [])

    return all_min_dists, np.array(min_min_dists), np.array(avg_min_dists)


def cal_min_dists(points):
    # a list of each pedestrian's min distances to other pedestrians
    n = len(points)
    ds_min = []
    for i in range(n):
        d_min = np.inf
        for j in range(n):
            if i != j and np.linalg.norm(points[i] - points[j]) < d_min:
                d_min = np.linalg.norm(points[i] - points[j])
        if d_min is not np.inf:
            ds_min.append(d_min)
    return ds_min


def custom_simple_linear_regression(xs, ys, x_select):

    def pred_interval(x_star, prob=0.95):
        se_pred = np.sqrt(variance * (1 + 1 / n + (x_star - xs.mean()) ** 2 / s_xx))
        t_dist = stats.t.ppf([1 - (1 - prob)/2], len(xs) - 2)
        y_hat = x_star * b + a
        return float(y_hat - se_pred * t_dist), float(y_hat + se_pred * t_dist)

    b, a, r_value, p_value, std_err = stats.linregress(xs, ys)

    print('slope b = %.6f' % b)
    print('intercept a = %.6f' % a)
    print('r_value = %.6f' % r_value)
    print('p_value = %.6f' % p_value)
    print('se_a (from package) = %.6f' % std_err)

    residuals = ys - (xs * b + a)
    n = len(residuals)
    variance = np.sum(residuals ** 2) / (n - 2)
    s_xx = np.sum((xs - xs.mean()) ** 2)
    s_x = s_xx / (n - 1)

    se_a = np.sqrt(variance / s_xx)
    print('se_a_ = %.6f' % se_a)

    preds, lbs, ubs = [], [], [] 
    for x in np.arange(0, np.max(xs), 0.001):
        lb, ub = pred_interval(x)
        preds.append(x)
        lbs.append(lb)
        ubs.append(ub)
    if x_select == 'x_intercept':
        x_select = - a / b
    elif x_select == 'y_intercept':
        x_select = 0.0
    else:
        raise Exception
    lb_select, ub_select = pred_interval(x_select)

    return a, b, preds, lbs, ubs, x_select, lb_select, ub_select

