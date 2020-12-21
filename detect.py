import sys
import torch
import torchvision
import os
import glob
import numpy as np
import time
import pickle
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from plotting import plot_frame, plot_frame_one_row, get_roi_pts
from utils1 import ROIs, find_violation

from utils1 import COCO_INSTANCE_CATEGORY_NAMES as LABELS
import cv2
np.set_printoptions(precision=4)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

videos_path = os.path.join('static', 'videos')
output_format = '.mp4'


def main(file_name='mall.mp4', dataset='mall', modelName='YOLO'):
    print('=========== %s ===========' % dataset)
    file_base_name = file_name.split('.')[0]
    path_result = os.path.join('results', file_base_name)
    videos_dir = os.path.join(videos_path, file_base_name)
    dataset_path = os.path.join(videos_dir, file_name)

    os.makedirs(path_result, exist_ok=True)
    matplotlib.use('agg')

    frameCount = 10
    personId = 0
    thr_score = 0

    if modelName == 'YOLO':
        # Model YOLOv5
        modelYolo = torch.hub.load(
            'ultralytics/yolov5', 'yolov5s', pretrained=True).autoshape()
        # modelYolo = torch.hub.load('ultralytics/yolov5', 'yolov5s',
        #                            pretrained=True, force_reload=True).autoshape()  # force reload
        frameCount = 10
    else:
        # Faster RCNN
        # initialize detector
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        modelRcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True)
        modelRcnn.to(device=device)
        modelRcnn.eval()
        frameCount = 5
        personId = 1

    # load transformation matrix
    transform_cam2world = np.loadtxt(os.path.join(
        'calibration', dataset + '_matrix_cam2world.txt'))

    # open video of dataset
    if dataset == 'oxford_town':
        cap = cv2.VideoCapture(dataset_path)
        frame_skip = 10
        if modelName == 'YOLO':
            thr_score = 0.5
        else:
            thr_score = 0.9
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
    elif dataset == 'mall':
        cap = cv2.VideoCapture(dataset_path)
        frame_skip = 1
        if modelName == 'YOLO':
            thr_score = 0.5
        else:
            thr_score = 0.9
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
    elif dataset == 'grand_central':
        cap = cv2.VideoCapture(dataset_path)
        frame_skip = 25
        if modelName == 'YOLO':
            thr_score = 0.25
        else:
            thr_score = 0.5
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
    else:
        raise Exception('Invalid Dataset')

    output_file_name = 'output_{}'.format(
        file_base_name) + output_format

    output_file_path = os.path.join(videos_dir, output_file_name)

    out = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(
        *'vp09'), 1, (frame_width, frame_height))

    statistic_data = []
    i_frame = 0
    avg_inference_time = 0
    # while cap.isOpened() and i_frame < 5000:
    while cap.isOpened():
        ret, img = cap.read()
        # print('Frame %d - ' % i_frame)
        if ret is False:
            break

        if i_frame > frameCount:
            break

        if i_frame / frame_skip < 20:
            vis = True
        else:
            vis = False

        # counting process time
        t0 = time.time()
        boxes = []
        classIds = []
        scroes = []

        if modelName == 'YOLO':
            # YOLO Model
            img_rgb = img[:, :, ::-1]  # OpenCV image (BGR to RGB)
            results = modelYolo(img_rgb, size=640)  # includes NMS
            arr = np.array(results.xyxy[0])
            boxes = arr[:, 0:4]
            classIDs = arr[:, 5]
            scores = arr[:, 4]

        else:
            # RCNN
            # convert image from OpenCV format to PyTorch tensor format
            img_t = np.moveaxis(img, -1, 0) / 255
            img_t = torch.tensor(img_t, device=device).float()

            # pedestrian detection
            predictions = modelRcnn([img_t])
            boxes = predictions[0]['boxes'].cpu().data.numpy()
            classIDs = predictions[0]['labels'].cpu().data.numpy()
            scores = predictions[0]['scores'].cpu().data.numpy()

        # get positions and plot on raw image
        pts_world = []
        for i in range(len(boxes)):
            if classIDs[i] == personId and scores[i] > thr_score:
                # extract the bounding box coordinates
                (x1, y1) = (boxes[i][0], boxes[i][1])
                (x2, y2) = (boxes[i][2], boxes[i][3])

                if vis:
                    # draw a bounding box rectangle and label on the image
                    cv2.rectangle(img, (x1, y1), (x2, y2), [0, 255, 0], 2)
                    text = "{}: {:.2f}".format('person', scores[i])
                    cv2.putText(img, text, (int(x1), int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 2)

                # find the bottom center position and convert it to world coordinate
                p_c = np.array([[(x1 + x2)/2], [y2], [1]])
                p_w = transform_cam2world @ p_c
                p_w = p_w / p_w[2]
                pts_world.append([p_w[0][0], p_w[1][0]])

        t1 = time.time()

        pts_world = np.array(pts_world)
        if dataset == 'oxford_town':
            pts_world[:, [0, 1]] = pts_world[:, [1, 0]]
            pass
        elif dataset == 'mall':
            # pts_world[:, [0, 1]] = pts_world[:, [1, 0]]
            pass
        elif dataset == 'grand_central':
            # pts_world[:, [0, 1]] = pts_world[:, [1, 0]]
            pass
        statistic_data.append((i_frame, t1 - t0, pts_world))

        # visualize
        if vis:
            violation_pairs = find_violation(pts_world)

            for i in range(len(violation_pairs)):
                box1 = violation_pairs[i][0]
                (x1, y1) = (boxes[box1][0], boxes[box1][1])
                (x2, y2) = (boxes[box1][2], boxes[box1][3])
                cv2.rectangle(img, (x1, y1), (x2, y2), [0, 0, 255], 2)

                box2 = violation_pairs[i][1]
                (x1, y1) = (boxes[box2][0], boxes[box2][1])
                (x2, y2) = (boxes[box2][2], boxes[box2][3])
                cv2.rectangle(img, (x1, y1), (x2, y2), [0, 0, 255], 2)

            pts_roi_world, pts_roi_cam = get_roi_pts(
                dataset=dataset, roi_raw=ROIs[dataset], matrix_c2w=transform_cam2world)

            fig = plot_frame_one_row(
                dataset=dataset,
                img_raw=img,
                pts_roi_cam=pts_roi_cam,
                pts_roi_world=pts_roi_world,
                pts_w=pts_world,
                pairs=violation_pairs
            )

            fig.savefig(os.path.join(path_result, 'frame%04d.png' % i_frame))
            plt.close(fig)

        # update loop info
        print('Frame %d - Inference Time: %.2f' % (i_frame, t1 - t0))
        print('=======================')
        i_frame += 1
        avg_inference_time = avg_inference_time + (t1 - t0)

        out.write(img)

    out.release()
    avg_inference_time = avg_inference_time / i_frame
    print(modelName+': Average Inferece Time = %.2f' % avg_inference_time)

    # save statistics
    # f.close()
    pickle.dump(statistic_data, open(os.path.join(
        path_result, 'statistic_data.p'), 'wb'))
