import matplotlib.pyplot as plt
import numpy as np
import cv2
import constants

def get_roi_points(dataset, roi_raw, matrix_c2w):

    if dataset == 'oxford_town':
        y1, y2, x1, x2 = roi_raw
    elif dataset == 'mall':
        x1, x2, y1, y2 = roi_raw
    elif dataset == 'grand_central':
        x1, x2, y1, y2 = roi_raw
    else:
        raise Exception('Invalid dataset.')

    points_world = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1], [x1, y1]])
    points_cam = []
    for pt_world in points_world:
        pt_cam = np.linalg.inv(
            matrix_c2w) @ np.array([[pt_world[0]], [pt_world[1]], [1]]).reshape(3)
        points_cam.append(pt_cam / pt_cam[-1])
    points_cam = np.array(points_cam)
    return points_world, points_cam[:, :2]


def plot_frame_one_row(dataset, img_raw, points_roi_cam, points_roi_world, points_w, pairs):
    b, g, r = cv2.split(img_raw)  # get b,g,r
    img_raw = cv2.merge([r, g, b])  # switch it to rgb

    if dataset == 'oxford_town':
        sub_3_lim = (20, -10, 0, 30)
        points_roi_world[:, [0, 1]] = points_roi_world[:, [1, 0]]
    elif dataset == 'mall':
        sub_3_lim = (-10, 10, 10, -10)
    elif dataset == 'grand_central':
        sub_3_lim = (-10, 30, 36, -4)
    else:
        raise Exception('Invalid dataset.')

    # plot
    fig = plt.figure(figsize=(8.77, 3.06))
    fig.subplots_adjust(left=0.08, bottom=0.15,
                        right=0.98, top=0.90, wspace=0.3)
    fig.suptitle('%s (%s)' %
                 (constants.DICT_DATASET_PLACES[dataset], constants.DICT_DATASET_NAMES[dataset]))

    # subplot 1 - camera view
    a = fig.add_subplot(1, 3, (1, 2))
    plt.imshow(img_raw)
    a.plot(points_roi_cam[:, 0], points_roi_cam[:, 1], '--b')
    a.set_xlabel('x position (pixel)')
    a.set_ylabel('y position (pixel)')

    # subplot 2 - bird eye view social distancing
    a = fig.add_subplot(1, 3, 3)
    a.plot(points_roi_world[:, 0], points_roi_world[:, 1], '--b')
    a.plot(points_w[:, 0], points_w[:, 1], 'og', alpha=0.5)

    for pair in pairs:
        data = np.array([points_w[pair[0]], points_w[pair[1]]])
        a.plot(data[:, 0], data[:, 1], '-r')

    a.axis('equal')
    a.grid()
    a.set_xlabel('x position (meter)')
    a.set_ylabel('y position (meter)')
    a.set(xlim=(sub_3_lim[0], sub_3_lim[1]), ylim=(sub_3_lim[2], sub_3_lim[3]))

    return fig


def plot_frame(dataset, img_raw, img_bev_bkgd_10x, points_roi_cam, points_roi_world, points_w, pairs):
    b, g, r = cv2.split(img_raw)  # get b,g,r
    img_raw = cv2.merge([r, g, b])  # switch it to rgb

    b, g, r = cv2.split(img_bev_bkgd_10x)  # get b,g,r
    img_bev_bkgd_10x = cv2.merge([r, g, b])  # switch it to rgb

    if dataset == 'oxford_town':
        sub_2_lim = (300, -100, 0, 400)
        sub_3_lim = (20, -20, 0, 40)
        points_roi_world[:, [0, 1]] = points_roi_world[:, [1, 0]]
    elif dataset == 'mall':
        sub_2_lim = (-150, 150, 100, -200)
        sub_3_lim = (-15, 15, 10, -20)
    elif dataset == 'grand_central':
        sub_2_lim = (-100, 300, 400, -100)
        sub_3_lim = (-15, 35, 40, -10)
    else:
        raise Exception('Invalid dataset.')
    ps_w_10x = points_w * 10

    # plot
    fig = plt.figure(figsize=(10, 10))

    # subplot 1 - camera view
    a = fig.add_subplot(2, 1, 1)
    plt.imshow(img_raw)
    a.plot(points_roi_cam[:, 0], points_roi_cam[:, 1], '--b')
    a.set_title('Video')

    # subplot 2 - bird eye view background
    a = fig.add_subplot(2, 2, 3)
    plt.imshow(img_bev_bkgd_10x)
    a.set_title('BEV')
    a.plot(ps_w_10x[:, 0], ps_w_10x[:, 1], 'or', alpha=0.5)

    a.axis('equal')
    a.grid()
    a.set_xlabel('0.1m / pixel')
    a.set_ylabel('0.1m / pixel')
    a.set(xlim=(sub_2_lim[0], sub_2_lim[1]), ylim=(sub_2_lim[2], sub_2_lim[3]))

    # subplot 3 - bird eye view social distancing
    a = fig.add_subplot(2, 2, 4)
    a.set_title('BEV - social distancing')
    a.plot(points_w[:, 0], points_w[:, 1], 'or', alpha=0.5)

    for pair in pairs:
        data = np.array([points_w[pair[0]], points_w[pair[1]]])
        a.plot(data[:, 0], data[:, 1], '-g')

    a.plot(points_roi_world[:, 0], points_roi_world[:, 1], '--b')
    a.axis('equal')
    a.grid()
    a.set_xlabel('meters')
    a.set_ylabel('meters')
    a.set(xlim=(sub_3_lim[0], sub_3_lim[1]), ylim=(sub_3_lim[2], sub_3_lim[3]))

    return fig
