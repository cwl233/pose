import logging
import multiprocessing
import asyncio
import platform
import sys
import logging
import os
import time
import socket
import grpc
import cv2
import numpy as np

import linke_vision_pb2
import linke_vision_pb2_grpc

queue1 = multiprocessing.Queue()
queue2 = multiprocessing.Queue()
queue3 = multiprocessing.Queue()
# return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
# pb_file = "./yolov3.pb"
# num_classes = 80
# input_size = 416
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" #gpu 物理上1号机器


# def tf_detect(image, mipp, sess, return_tensors):
#     frame_size = image.shape[:2]
#     image_data = utils.image_preporcess(np.copy(image), [input_size, input_size])
#     image_data = image_data[np.newaxis, ...]
#     # prev_time = time.time()
#
#     pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
#         [return_tensors[1], return_tensors[2], return_tensors[3]],
#         feed_dict={return_tensors[0]: image_data})
#
#     pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
#                                 np.reshape(pred_mbbox, (-1, 5 + num_classes)),
#                                 np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)
#
#     bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.6)
#     bboxes = utils.nms(bboxes, mipp, method='soft-nms')
#     image = utils.draw_bbox(image, bboxes)
#     return image, bboxes


grpc_timeout = 1
grpc_check_interval = 5
grpc_connecting = False
channel = None
stub = None

save_ext = ".jpg"
last_time = time.time()

already_sent = True
temp_frame = None
output_bytes = None
output_dets = None

flag1 = 0
flag2 = 0
is_container = False
container_id = ""
output = socket.gethostname()
if output:
    is_container = True
    container_id = output


def init_openpose():
    global opWrapper,datum
    try:
        # Import Openpose (Windows/Ubuntu/OSX)
        try:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
        except ImportError as e:
            print(
                'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
            raise e

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        params["model_folder"] = "../../../models/"

        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        # opWrapper.start()
        datum = op.Datum()
    except Exception as e:
        print(e)
        sys.exit(-1)

# Thread-1
def get_video():
    env_list = os.environ
    if "VIDEO_STREAM_ADDRESS" not in env_list:
        print("Can't find env VIDEO_STREAM_ADDRESS, please set it.")
        exit(0)
    # video_stream_address = "rtsp://admin:FHRGLD@202.38.86.60:10554/H.264"
    video_stream_address = env_list["VIDEO_STREAM_ADDRESS"]
    count = 0
    while True:
        count += 1
        if count == 3:
            raise Exception('network error')
        cap = cv2.VideoCapture(video_stream_address)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.resize(frame, (640, 360))
            if not ret:
                cap = cv2.VideoCapture(video_stream_address)
                continue

            queue1.put(frame)
        time.sleep(5)

# Thread-2
def detect():
    try:
        # Import Openpose (Windows/Ubuntu/OSX)
        try:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
        except ImportError as e:
            print(
                'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
            raise e

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        params["model_folder"] = "../../../models/"

        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        t = time.time()
        while True:
            t = time.time()
            frame_to_detect = queue1.get()
            while True:
                if not queue1.empty():
                    frame_to_detect = queue1.get()
                else:
                    break
            datum = op.Datum()
            # imageToProcess = cv2.imread(frame_to_detect)
            datum.cvInputData = frame_to_detect
            opWrapper.emplaceAndPop([datum])

            queue2.put(datum.cvOutputData)
            print('detect:' + str(time.time() - t))

    except Exception as e:
        print(e)
        sys.exit(-1)




def encode():
    t = time.time()
    while True:
        t = time.time()
        sent_bytes = queue2.get()
        print('encode1:' + str(time.time() - t))
        sent_bytes = cv2.imencode('.jpg', sent_bytes)[1].tobytes()
        print('encode2:' + str(time.time() - t))
        queue3.put(sent_bytes)


# Thread-3
def grpc_send_request():
    global grpc_connecting, last_time
    env_list = os.environ
    if "GRPC_SERVER_ADDRESS" not in env_list:
        print("Can't find env GRPC_SERVER_ADDRESS, please set it.")
        exit(0)
    # grpc_server_address = "localhost:50051"
    # grpc_server_address = "202.38.86.69:50051"
    grpc_server_address = env_list["GRPC_SERVER_ADDRESS"]
    while True:
        channel = grpc.insecure_channel(grpc_server_address)
        try:
            grpc.channel_ready_future(channel).result(timeout=grpc_timeout)
        except grpc.FutureTimeoutError:
            print("grpc connect failed:", grpc_server_address)
            grpc_connecting = False
        else:
            if not grpc_connecting:
                print("grpc connect successed", grpc_server_address)
                grpc_connecting = True
                stub = linke_vision_pb2_grpc.LinkeVisionStub(channel)
                while True:
                    if not grpc_connecting:
                        break
                    # if time.time() - last_time < spf:
                    #     continue

                    # if queue2.empty():
                    #     continue
                    # else:
                    last_time = time.time()
                    sent_bytes = queue3.get()
                    print('send1:' + str(time.time() - last_time))

                    stub.ObjectDetection(linke_vision_pb2.DetectObjects4Image(
                        image=linke_vision_pb2.File(extension=save_ext, size=len(sent_bytes), bytes=sent_bytes),
                        dets=output_dets,
                        is_container=is_container,
                        container_id=container_id))
                    print('send2:' + str(time.time() - last_time))
        finally:
            time.sleep(grpc_check_interval)


# Main Thread
if __name__ == '__main__':
    time.sleep(10)
    logging.basicConfig()
    p_list = []
    function_list = [detect, grpc_send_request, encode]
    for function in function_list:
        p = multiprocessing.Process(target=function, daemon=True)
        p.start()
        p_list.append(p)
    get_video()
    for pp in p_list:
        pp.join()
