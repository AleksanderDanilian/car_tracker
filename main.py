import os
from random import randint
import numpy as np
import pandas as pd
from imutils.video import VideoStream
import argparse
import time
import cv2
from typing import Tuple

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, default="test_video/test_video_1.mkv",
                help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="medianflow",
                help="OpenCV object tracker type")
ap.add_argument("-vr", "--video_res", type=Tuple[int, int], default=(416, 416),
                help="video_resolution (height, width)")
ap.add_argument("-tr_1", "--track_one", type=bool, default=True,
                help="track only middle object")
ap.add_argument("-p", "--path_save_pic", type=str, default='test_video/images',
                help="folder to save only unique pictures from video")
args = vars(ap.parse_args())

# OpenCV object tracker implementations
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.legacy.TrackerCSRT_create,
    "kcf": cv2.legacy.TrackerKCF_create,
    "mil": cv2.legacy.TrackerMIL_create,
    "tld": cv2.legacy.TrackerTLD_create,
    "medianflow": cv2.legacy.TrackerMedianFlow_create,
    "mosse": cv2.legacy.TrackerMOSSE_create
}
# initialize OpenCV's special multi-object tracker
trackers = cv2.legacy.MultiTracker_create()

# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)
# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])

frame = vs.read()[1]
cv2.destroyAllWindows()

df_bboxes = pd.read_csv(args["video"][:-3] + 'csv')

bboxes = []
colors = []
nr_cars_first_frame = len(df_bboxes[df_bboxes['frame'] == 0])
orig_v_height = frame.shape[0]
orig_v_width = frame.shape[1]

for i in range(nr_cars_first_frame):
    bboxes.append(df_bboxes.loc[i, ['x0', 'y0', 'x1', 'y1']].values)
    colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))

# to absolute values
bboxes = np.array(bboxes)
bboxes[:, 0] = bboxes[:, 0] * orig_v_width
bboxes[:, 1] = bboxes[:, 1] * orig_v_height
bboxes[:, 2] = bboxes[:, 2] * orig_v_width
bboxes[:, 3] = bboxes[:, 3] * orig_v_height

new_v_height = args['video_res'][0]
new_v_width = args['video_res'][1]
width_v_scale = new_v_width / orig_v_width
height_v_scale = new_v_height / orig_v_height

# to new scale
bboxes[:, 0] = bboxes[:, 0] * width_v_scale
bboxes[:, 1] = bboxes[:, 1] * height_v_scale
bboxes[:, 2] = bboxes[:, 2] * width_v_scale
bboxes[:, 3] = bboxes[:, 3] * height_v_scale

bboxes_centroids = []
for box in bboxes:
    x = int(box[0] + (box[2] - box[0])/2)
    y = int(box[1] + (box[3] - box[1])/2)
    bboxes_centroids.append(np.array([x, y]))

frame = cv2.resize(frame, (416, 416))

# turn to [x, y, width, height]
bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]  # width of rect
bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]  # height of rect

if not args['track_one']:
    for box in bboxes:
        tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
        trackers.add(tracker, frame, box)
else:
    center_point_x, center_point_y = int(new_v_height / 2), int(new_v_width / 2)
    image_centroid = np.array([center_point_x, center_point_y])
    euclidian_distance_arr = []
    for i, box in enumerate(bboxes):
        euclidian_distance = np.sqrt(np.sum((image_centroid - bboxes_centroids[i]) ** 2))
        euclidian_distance_arr.append(euclidian_distance)
    idx_center_frame = np.argmin(euclidian_distance_arr)
    tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

    trackers.add(tracker, frame, bboxes[idx_center_frame])

# loop over frames from the video stream

frame_counter = 0
previous_frame = np.zeros(shape = frame.shape)

while True:
    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    # check to see if we have reached the end of the stream
    if frame is None:
        break
    # resize the frame (so we can process it faster) and grab the
    # frame dimensions
    frame = cv2.resize(frame, args['video_res'])

    # grab the updated bounding box coordinates (if any) for each
    # object that is being tracked
    (success, boxes) = trackers.update(frame)
    # loop over the bounding boxes and draw then on the frame

    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if args['path_save_pic']:
        if not os.path.exists(args['path_save_pic']):
            os.makedirs(args['path_save_pic'])
        frame_counter += 1
        if frame_counter > 0:
            euclidian_distance = np.sqrt(np.sum((frame - previous_frame) ** 2))
            previous_frame = frame
            if euclidian_distance > 100:
                cv2.imwrite(f'{args["path_save_pic"]}/{frame_counter}.jpg', frame)

    length = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
    print(length)

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# if we are using a webcam, release the pointer
if not args.get("video", False):
    vs.stop()
# otherwise, release the file pointer
else:
    vs.release()
# close all windows

cv2.destroyAllWindows()
