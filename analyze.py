import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import LogNorm
from scipy import stats
import constants
import os
import pickle
from utilities import count_violation_pairs, decode_data, cal_min_dists_all_frame, custom_simple_linear_regression

def analyze_statistics(dataset, file_name):
    print('=======================')
    print('Processing %s ...' % dataset)

    file_base_name = file_name.split('.')[0]
    path_result = os.path.join('results', file_base_name)
    path_analysis = os.path.join(path_result, 'analysis')
    images_path = os.path.join('static', 'images')
    images_dir = os.path.join(images_path, file_base_name)

    os.makedirs(path_result, exist_ok=True)
    os.makedirs(path_analysis, exist_ok=True)

    data = pickle.load(
        open(os.path.join(path_result, 'statistic_data.p'), 'rb'))
    roi = constants.ROIs[dataset]
    x_min, x_max, y_min, y_max = roi
    area = (x_max - x_min) * (y_max - y_min)
    indexs_frame, ts_inference, points_roi_all_frame, density, nums_ped = decode_data(
        data=data, roi=roi)
    print('Mean inference time = %.6f' % np.mean(ts_inference))

    all_min_dists, min_min_dists, avg_min_dists = cal_min_dists_all_frame(
        points_roi_all_frame)
    violations = count_violation_pairs(points_all_frames=points_roi_all_frame)

    none_indexes = np.where(avg_min_dists == None)[0]
    indexs_frame = np.delete(indexs_frame, none_indexes, 0)
    density = np.delete(density, none_indexes, 0)
    min_min_dists = np.delete(min_min_dists, none_indexes, 0)
    avg_min_dists = np.delete(avg_min_dists, none_indexes, 0)
    nums_ped = np.delete(nums_ped, none_indexes, 0)
    violations = np.delete(violations, none_indexes, 0)

    # figure 1 - closest_dists_path
    fig = plt.figure(figsize=(5., 3.))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.9, bottom=0.15)
    fig.suptitle(constants.DICT_DATASET_NAMES[dataset])

    ax = fig.add_subplot(111)
    ax.hist(all_min_dists, bins=100, color='lightseagreen')
    ax.grid()
    ax.set_xlabel(r'Closest Distance ($m$)')
    ax.set_ylabel('Count')
    ax.set(xlim=(0, 10))
    closest_dists_path = os.path.join(
        path_analysis, constants.CLOSEST_DIST)
    fig.savefig(closest_dists_path)
    closest_dists_path = os.path.join(
        images_dir, constants.CLOSEST_DIST)
    fig.savefig(closest_dists_path)
    plt.close(fig)

    # figure 2 - min_closest_dists_path
    fig = plt.figure(figsize=(5., 3.))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.9, bottom=0.15)
    fig.suptitle(constants.DICT_DATASET_NAMES[dataset])

    ax = fig.add_subplot(111)
    ax.hist(min_min_dists, bins=100, color='lightseagreen')
    ax.grid()
    ax.set_xlabel(r'Minimal Closest Distance ($m$)')
    ax.set_ylabel('Count')
    min_closest_dists_path = os.path.join(
        path_analysis, constants.MIN_CLOSEST_DIST)
    fig.savefig(min_closest_dists_path)
    min_closest_dists_path = os.path.join(
        images_dir, constants.MIN_CLOSEST_DIST)
    fig.savefig(min_closest_dists_path)
    plt.close(fig)

    # figure 3 - density, closest_dists, min_closest_dists v/s time
    t_max = len(indexs_frame)

    if dataset == 'oxford_town':
        ts = indexs_frame / 10
        t_max = t_max / 10
    elif dataset == 'mall':
        ts = indexs_frame / 1
    elif dataset == 'grand_central':
        ts = indexs_frame / 25
        t_max = t_max / 25
    else:
        raise Exception('invalid dataset')
    extracted = ts <= t_max

    fig = plt.figure(figsize=(11.47, 4.33))
    fig.subplots_adjust(left=0.07, bottom=0.12,
                        right=0.98, top=0.91, hspace=0.36)
    fig.suptitle(constants.DICT_DATASET_NAMES[dataset])

    ax = fig.add_subplot(2, 1, 1)
    ax.plot(ts[extracted], avg_min_dists[extracted], '.-',
            label=r'avg. closest physical distance $d_{avg}$ ($m$)', color='g')
    ax.plot(ts[extracted], min_min_dists[extracted], '.-',
            label=r'min. closest physical distance $d_{min}$ ($m$)', color='r')

    ax.grid()
    ax.set_xlabel(r'Time [$sec$]')
    ax.set(xlim=(0, t_max))
    ax.legend(loc=1)

    ax = fig.add_subplot(2, 1, 2)
    ax.plot(ts[extracted], density[extracted], '.-',
            label=r'social density $\rho$ (ped./$m^2$)', color='navy')
    ax.grid()
    ax.set_xlabel(r'Time [$sec$]')
    ax.set(xlim=(0, t_max))
    ax.legend(loc=1)
    stats_vs_time_path = os.path.join(
        path_analysis, constants.STATS_VS_TIME)
    fig.savefig(stats_vs_time_path)
    stats_vs_time_path = os.path.join(
        images_dir, constants.STATS_VS_TIME)
    fig.savefig(stats_vs_time_path)
    plt.close(fig)

    # 2d histograms 

    bin_size = 15

    # figure 4 - density_vs_avg_dists_path
    fig = plt.figure(figsize=(4.15, 3.27))
    fig.subplots_adjust(left=0.18, right=1.0, top=0.9, bottom=0.14)
    fig.suptitle(constants.DICT_DATASET_NAMES[dataset])
    ax = fig.add_subplot(1, 1, 1)
    plt.hist2d(avg_min_dists, density, bins=(bin_size, bin_size), cmap='Blues')
    plt.colorbar()
    ax.set_ylabel(r'Social Density $\rho$ (ped./$m^2$)')
    ax.set_xlabel('Avg. Closest Physical Distance $d_{avg}$ ($m$)')
    two_d_hist_density_vs_avg_dists_path = os.path.join(
        path_analysis, constants.TWO_D_HIST_DENSITY_VS_AVG_DISTS)
    fig.savefig(two_d_hist_density_vs_avg_dists_path)
    two_d_hist_density_vs_avg_dists_path = os.path.join(
        images_dir, constants.TWO_D_HIST_DENSITY_VS_AVG_DISTS)
    fig.savefig(two_d_hist_density_vs_avg_dists_path)
    plt.close(fig)

    # figure 5 - density_vs_min_dists_path
    fig = plt.figure(figsize=(4.15, 3.27))
    fig.subplots_adjust(left=0.18, right=1.0, top=0.9, bottom=0.14)
    fig.suptitle(constants.DICT_DATASET_NAMES[dataset])
    ax = fig.add_subplot(1, 1, 1)
    plt.hist2d(min_min_dists, density, bins=(bin_size, bin_size), cmap='Blues')
    plt.colorbar()
    ax.set_ylabel(r'Social Density $\rho$ (ped./$m^2$)')
    ax.set_xlabel(r'Min. Closest Physical Distance $d_{min}$ ($m$)')
    two_d_hist_density_vs_min_dists_path = os.path.join(
        path_analysis, constants.TWO_D_HIST_DENSITY_VS_MIN_DISTS)
    fig.savefig(two_d_hist_density_vs_min_dists_path)
    two_d_hist_density_vs_min_dists_path = os.path.join(
        images_dir, constants.TWO_D_HIST_DENSITY_VS_MIN_DISTS)
    fig.savefig(two_d_hist_density_vs_min_dists_path)
    plt.close(fig)

    # figure 6 - density_vs_violation_path
    fig = plt.figure(figsize=(4.15, 3.27))
    fig.subplots_adjust(left=0.18, right=1.0, top=0.9, bottom=0.14)
    fig.suptitle(constants.DICT_DATASET_NAMES[dataset])
    ax = fig.add_subplot(1, 1, 1)
    plt.hist2d(violations, density, bins=(bin_size, bin_size), cmap='Blues')
    plt.colorbar()
    ax.set_ylabel(r'Social Density $\rho$ (ped./$m^2$)')
    ax.set_xlabel(r'Num. of Social Distancing Violations $v$')
    two_d_hist_density_vs_violation_path = os.path.join(
        path_analysis, constants.TWO_D_HIST_DENSITY_VS_VIOLATIONS)
    fig.savefig(two_d_hist_density_vs_violation_path)
    two_d_hist_density_vs_violation_path = os.path.join(
        images_dir, constants.TWO_D_HIST_DENSITY_VS_VIOLATIONS)
    fig.savefig(two_d_hist_density_vs_violation_path)
    plt.close(fig)

    # ---- figure 7 ----
    fig = plt.figure(figsize=(3.28, 3.14))
    fig.subplots_adjust(left=0.21, bottom=0.15, right=0.97, top=0.90)
    fig.suptitle(constants.DICT_DATASET_NAMES[dataset])

    ax = fig.add_subplot(1, 1, 1)
    ax.grid()
    ax.plot(violations, density, '.b', alpha=0.2)
    ax.set_ylabel(r'Social Density $\rho$ (ped./$m^2$)')
    ax.set_xlabel(r'Num. of Social Distancing Violations $v$')

    print('Average Closest Physical Distance = %.6f' %
          np.mean(avg_min_dists))
    print('Skewness = %.6f' % stats.skew(density))

    intercept, slope, preds, lbs, ubs, x_select, lb_select, ub_select = custom_simple_linear_regression(
        xs=violations, ys=density, x_select='y_intercept')

    print('x_select = %.4f' % x_select)
    print('x_select lower bound = %.4f' % lb_select)
    print('x_select upper bound = %.4f' % ub_select)

    line = [0, max(violations)]

    ax.plot(line, [intercept + slope * line[0],
                   intercept + slope * line[1]], color='deeppink')
    ax.plot(preds, lbs, color='navy')
    ax.plot(preds, ubs, color='navy')
    plt.text(0.0 + 0.5, lb_select - 0.005, r'$\rho_c$',
             fontsize=15, color='deeppink')
    ax.set(xlim=(0.0, np.max(violations)))
    ax.set(ylim=(0.0, np.max(density)))
    

    regression_density_vs_violations_path = os.path.join(
        path_analysis, constants.REGRESSION_DENSITY_VS_VIOLATION)
    fig.savefig(regression_density_vs_violations_path)
    regression_density_vs_violations_path = os.path.join(
        images_dir, constants.REGRESSION_DENSITY_VS_VIOLATION)
    fig.savefig(regression_density_vs_violations_path)
    plt.close(fig)
    return (closest_dists_path,
            min_closest_dists_path,
            stats_vs_time_path,
            two_d_hist_density_vs_avg_dists_path,
            two_d_hist_density_vs_min_dists_path,
            two_d_hist_density_vs_violation_path,
            regression_density_vs_violations_path)

