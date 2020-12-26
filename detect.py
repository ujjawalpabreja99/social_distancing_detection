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
from datetime import datetime
from plotting import plot_frame, plot_frame_one_row, get_roi_pts
from utilities import ROIs, find_violation

videos_path = os.path.join('static', 'videos')
output_format = '.mp4'


def main(file_name='mall.mp4', dataset='mall', model_name='YOLO'):
    print('=========== %s ===========' % dataset)
    file_base_name = file_name.split('.')[0]
    results_path = os.path.join('results', file_base_name)
    videos_dir = os.path.join(videos_path, file_base_name)
    dataset_path = os.path.join(videos_dir, file_name)

    os.makedirs(results_path, exist_ok=True)
    matplotlib.use('agg')

    frame_count = 10
    person_id = 0
    threshold_score = 0

    if model_name == 'YOLO':
        # Model YOLOv5
        model_yolo_v5 = torch.hub.load(
            'ultralytics/yolov5', 'yolov5s', pretrained=True).autoshape()
        # model_yolo_v5 = torch.hub.load('ultralytics/yolov5', 'yolov5s',
        #                            pretrained=True, force_reload=True).autoshape()  # force reload
        frame_count = 10
        person_id = 0
    else:
        # Faster RCNN
        # initialize detector
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True)
        model_faster_rcnn.to(device=device)
        model_faster_rcnn.eval()
        frame_count = 5
        person_id = 1

    # load transformation matrix
    transform_cam2world = np.loadtxt(os.path.join(
        'calibration', dataset + '_matrix_cam2world.txt'))

    # open video of dataset
    cap = cv2.VideoCapture(dataset_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    if dataset == 'oxford_town':
        frame_skip = 10
        if model_name == 'YOLO':
            threshold_score = 0.5
        else:
            threshold_score = 0.9
    elif dataset == 'mall':
        frame_skip = 1
        if model_name == 'YOLO':
            threshold_score = 0.5
        else:
            threshold_score = 0.9
    elif dataset == 'grand_central':
        frame_skip = 25
        if model_name == 'YOLO':
            threshold_score = 0.25
        else:
            threshold_score = 0.5
    else:
        raise Exception('Invalid Dataset')

    output_file_name = 'output_%s%s' % (file_base_name, output_format)

    output_file_path = os.path.join(videos_dir, output_file_name)

    out = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(
        *'vp09'), 1, (frame_width, frame_height))

    statistic_data = []
    i_frame = 0
    total_inference_time = 0

    while cap.isOpened():
        ret, img = cap.read()
        if ret is False:
            break

        if i_frame > frame_count:
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

        if model_name == 'YOLO':
            # YOLO Model
            img_rgb = img[:, :, ::-1]  # OpenCV image (BGR to RGB)
            results = model_yolo_v5(img_rgb, size=640)
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
            predictions = model_faster_rcnn([img_t])
            boxes = predictions[0]['boxes'].cpu().data.numpy()
            classIDs = predictions[0]['labels'].cpu().data.numpy()
            scores = predictions[0]['scores'].cpu().data.numpy()

        # get positions and plot on raw image
        pts_world = []
        for i in range(len(boxes)):
            if classIDs[i] == person_id and scores[i] > threshold_score:
                # extract the bounding box coordinates
                (x1, y1) = (boxes[i][0], boxes[i][1])
                (x2, y2) = (boxes[i][2], boxes[i][3])

                if vis:
                    # draw a bounding box rectangle and label on the image
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 128, 0), 1)
                    text = "{}: {:.2f}".format('person', scores[i])
                    text_size = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                    cv2.rectangle(
                        img, (x1, y1), (int(x1 + text_size[0] + 3), int(y1 + text_size[1] + 4)), (0, 128, 0), -1)
                    cv2.putText(img, text,
                                (x1, int(y1 + text_size[1] + 4)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

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
                # draw bounding box rectangle on the violating pair of pedestrians
                # box1 denotes the first pedestrian
                box1 = violation_pairs[i][0]
                (x1, y1) = (boxes[box1][0], boxes[box1][1])
                (x2, y2) = (boxes[box1][2], boxes[box1][3])
                cv2.rectangle(img, (x1, y1), (x2, y2), [0, 0, 255], 1)
                text = "{}: {:.2f}".format('person', scores[box1])
                text_size = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                cv2.rectangle(
                    img, (x1, y1), (int(x1 + text_size[0] + 3), int(y1 + text_size[1] + 4)), (0, 0, 255), -1)
                cv2.putText(img, text,
                            (x1, int(y1 + text_size[1] + 4)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                # box2 denotes the second pedestrian
                box2 = violation_pairs[i][1]
                (x1, y1) = (boxes[box2][0], boxes[box2][1])
                (x2, y2) = (boxes[box2][2], boxes[box2][3])
                cv2.rectangle(img, (x1, y1), (x2, y2), [0, 0, 255], 1)
                text = "{}: {:.2f}".format('person', scores[box2])
                text_size = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                cv2.rectangle(
                    img, (x1, y1), (int(x1 + text_size[0] + 3), int(y1 + text_size[1] + 4)), (0, 0, 255), -1)
                cv2.putText(img, text,
                            (x1, int(y1 + text_size[1] + 4)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

            text = 'Social Distancing Violations = %d' % len(violation_pairs)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            cv2.rectangle(img, (0, 60), (frame_width, 0), (0, 0, 0), -1)
            cv2.putText(img, text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2, cv2.LINE_AA, False)

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
    # f.close()
    pickle.dump(statistic_data, open(os.path.join(
        results_path, 'statistic_data.p'), 'wb'))
