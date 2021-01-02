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
import cv2
import constants
from datetime import datetime
from plotting import plot_frame, plot_frame_one_row, get_roi_points
from utilities import find_violation

def draw_bounding_box(img, x1, y1, x2, y2, score, color):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    text = "person: %.2f" % score
    text_size = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    cv2.rectangle(
        img, (x1, y1), (int(x1 + text_size[0] + 3), int(y1 + text_size[1] + 4)), color, -1)
    cv2.putText(img, text,
                (x1, int(y1 + text_size[1] + 4)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

def main(file_name='mall.mp4', dataset='mall', model_name='YOLO'):
    print('=========== %s ===========' % dataset)
    file_base_name = file_name.split('.')[0]
    results_path = os.path.join('results', file_base_name)
    videos_path = os.path.join('static', 'videos')
    videos_dir = os.path.join(videos_path, file_base_name)
    dataset_path = os.path.join(videos_dir, file_name)

    os.makedirs(results_path, exist_ok=True)
    matplotlib.use('agg')

    person_id = 0
    threshold_score = 0

    if model_name == 'YOLO':
        # Model YOLOv5
        model_yolo_v5 = torch.hub.load(
            'ultralytics/yolov5', 'yolov5s', pretrained=True).autoshape()
        person_id = 0
    else:
        # Faster RCNN
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True)
        model_faster_rcnn.to(device=device)
        model_faster_rcnn.eval()
        person_id = 1

    # load transformation matrix
    transform_cam2world = np.loadtxt(os.path.join(
        'calibration', dataset + '_matrix_cam2world.txt'))

    # open video of dataset
    cap = cv2.VideoCapture(dataset_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    if dataset == 'oxford_town':
        if model_name == 'YOLO':
            threshold_score = 0.5
        else:
            threshold_score = 0.9
    elif dataset == 'mall':
        if model_name == 'YOLO':
            threshold_score = 0.5
        else:
            threshold_score = 0.9
    elif dataset == 'grand_central':
        if model_name == 'YOLO':
            threshold_score = 0.25
        else:
            threshold_score = 0.5
    else:
        raise Exception('Invalid Dataset')

    output_file_name = 'output_%s%s' % (file_base_name, constants.OUTPUT_FORMAT)

    output_file_path = os.path.join(videos_dir, output_file_name)

    out = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(
        *'vp09'), 1, (frame_width, frame_height))

    statistic_data = []
    i_frame = 0
    total_inference_time = 0

    frame_count = 5

    while cap.isOpened():
        # comment line 99, 100 to run on full video
        if i_frame > frame_count:
            break

        ret, img = cap.read()
        if ret is False:
            break

        # counting process time
        t0 = time.time()

        boxes = []
        classIds = []
        scroes = []

        if model_name == 'YOLO':
            # YOLO Model
            img_rgb = img[:, :, ::-1] 
            results = model_yolo_v5(img_rgb, size=640)
            arr = np.array(results.xyxy[0])
            boxes = arr[:, 0:4]
            classIDs = arr[:, 5]
            scores = arr[:, 4]

        else:
            # Faster RCNN
            img_t = np.moveaxis(img, -1, 0) / 255
            img_t = torch.tensor(img_t, device=device).float()
            predictions = model_faster_rcnn([img_t])
            boxes = predictions[0]['boxes'].cpu().data.numpy()
            classIDs = predictions[0]['labels'].cpu().data.numpy()
            scores = predictions[0]['scores'].cpu().data.numpy()

        # get positions and plot on raw image
        points_world = []
        for i in range(len(boxes)):
            if classIDs[i] == person_id and scores[i] > threshold_score:
                # extract the bounding box coordinates
                (x1, y1) = (boxes[i][0], boxes[i][1])
                (x2, y2) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the image
                draw_bounding_box(img, x1, y1, x2, y2, scores[i], constants.GREEN)

                # find the bottom center position and convert it to world coordinate
                p_c = np.array([[(x1 + x2)/2], [y2], [1]])
                p_w = transform_cam2world @ p_c
                p_w = p_w / p_w[2]
                points_world.append([p_w[0][0], p_w[1][0]])

        t1 = time.time()

        points_world = np.array(points_world)
        if dataset == 'oxford_town':
            points_world[:, [0, 1]] = points_world[:, [1, 0]]
            pass
        elif dataset == 'mall':
            pass
        elif dataset == 'grand_central':
            pass

        # store current frame info
        statistic_data.append((i_frame, t1 - t0, points_world))

        # visualize
        violation_pairs = find_violation(points_world)

        for i in range(len(violation_pairs)):
            # draw bounding box rectangle on the violating pair of pedestrians
            # box1 denotes the first pedestrian
            box1 = violation_pairs[i][0]
            (x1, y1) = (boxes[box1][0], boxes[box1][1])
            (x2, y2) = (boxes[box1][2], boxes[box1][3])
            draw_bounding_box(img, x1, y1, x2, y2, scores[box1], constants.RED)
            # box2 denotes the second pedestrian
            box2 = violation_pairs[i][1]
            (x1, y1) = (boxes[box2][0], boxes[box2][1])
            (x2, y2) = (boxes[box2][2], boxes[box2][3])
            draw_bounding_box(img, x1, y1, x2, y2, scores[box2], constants.RED)

        text = 'Social Distancing Violations = %d' % len(violation_pairs)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        cv2.rectangle(img, (0, 60), (frame_width, 0), (0, 0, 0), -1)
        cv2.putText(img, text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA, False)

        points_roi_world, points_roi_cam = get_roi_points(
            dataset=dataset, roi_raw=constants.ROIs[dataset], matrix_c2w=transform_cam2world)

        fig = plot_frame_one_row(
            dataset=dataset,
            img_raw=img,
            points_roi_cam=points_roi_cam,
            points_roi_world=points_roi_world,
            points_w=points_world,
            pairs=violation_pairs
        )

        fig.savefig(os.path.join(results_path, 'frame%04d.png' % i_frame))
        plt.close(fig)

        # update loop info
        print('Frame %d - Inference Time: %.2f' % (i_frame, t1 - t0))
        print('=======================')
        i_frame += 1
        total_inference_time += (t1 - t0)

        out.write(img)

    out.release()
    avg_inference_time = total_inference_time / i_frame
    print('%s: Average Inferece Time = %.2f' %
          (model_name, avg_inference_time))

    # save statistics
    pickle.dump(statistic_data, open(os.path.join(
        results_path, 'statistic_data.p'), 'wb'))
