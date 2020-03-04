"""trt_yolov3.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLOv3 engine.
"""

import os
import sys
import time
import random
import numpy as np
import subprocess
import argparse

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolov3_classes import get_cls_dict
from utils.yolov3 import TrtYOLOv3
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization


WINDOW_NAME = 'TrtYOLOv3Demo'


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLOv3 model on Jetson Nano')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument('--model', type=str, default='yolov3-416',
                        choices=['yolov3-288', 'yolov3-416', 'yolov3-608',
                                 'yolov3-tiny-288', 'yolov3-tiny-416'])
    args = parser.parse_args()
    return args


def loop_and_detect(cam, trt_yolov3, conf_th, vis):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolov3: the TRT YOLOv3 object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    full_scrn = False
    fps = 0.0
    detectandspeech = np.zeros((6,2), dtype=float)
    tic = time.time()
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break

        img = cam.read()
        if img is not None:
            boxes, confs, clss = trt_yolov3.detect(img, conf_th)
            ###########################
            ## DETECTED CASE - SSMIN ## 
            ###########################
            if len(clss) > 0:
                print("Detected : ",clss, ", ", confs)
                detectandspeech[clss, 0] = tic
                if detectandspeech[clss, 0] - detectandspeech[clss, 1] > 60:
                    detectandspeech[clss, 1] = tic
                    print("Played : ",clss)
                    MusicPlayCheck(clss)

                    if clss == 1:
                        os.system('mpg321 voice/jason.mp3 & > /dev/null')
                    elif clss == 2:
                        os.system('mpg321 voice/jessica.mp3 & > /dev/null')
                    elif clss == 3:
                        os.system('mpg321 voice/erica.mp3 & > /dev/null')
                    elif clss == 4:
                        os.system('mpg321 voice/woo.mp3 & > /dev/null')
                    elif clss == 5:
                        os.system('mpg321 voice/woong.mp3 & > /dev/null')

                

            ###########################
            draw = 1
            if draw is not 0:
                img = vis.draw_bboxes(img, boxes, confs, clss)
                img = show_fps(img, fps)
                cv2.imshow(WINDOW_NAME, img)

            toc = time.time()
            curr_fps = 1.0 / (toc - tic)
            # calculate an exponentially decaying average of fps number
            fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
            tic = toc
        key = cv2.waitKey(1)
        if key == 27:  # ESC key: quit program
            break
        elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            full_scrn = not full_scrn
            set_display(WINDOW_NAME, full_scrn)
        time.sleep(0.05)

def MusicPlayCheck(clss):
    ps = subprocess.Popen(['ps', '-A'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0]
    if str(ps).find('mpg321') > 0:
        print("Background Music is Playing now")
        return

    jason_music_list   = ["sleepingsun.mp3", "Nemo.mp3", "Over The Hills And Far Away.mp3", "Wishmaster.mp3"]
    jessica_music_list = ["vivaldi-1.mp3","vivaldi-2.mp3","vivaldi-3.mp3","vivaldi-4.mp3"]
    erica_music_list   = ["Mozart_Symphony_41.mp3", "Saint_Saens_Carnival_of_the_Animals_Finale.mp3"]

    music_list = []
    if clss == 1:
        music_list = jason_music_list
    elif clss == 2:
        music_list = jessica_music_list
    elif clss == 3:
        music_list = erica_music_list
    else:
        return

    music_list_count = len(music_list)
    randselect = random.randint(0, music_list_count-1)

    print("BG Song is ",music_list[randselect])
    os.system("mpg321 voice/"+music_list[randselect]+" & > /dev/null")                
         
    return 

# python3 trt_yolov3_deepfamily1.py --model yolov3-416 --vid 0 --width 1280 --height 720
def main():
    args = parse_args()


    #YOLO INIT
    #cls_dict = get_cls_dict('coco')
    cls_dict = get_cls_dict('deepfamily')
    print("classes count : ", len(cls_dict))
    yolo_dim = int(args.model.split('-')[-1])  # 416 or 608
    print("yolo_dim : ", yolo_dim)
    trt_yolov3 = TrtYOLOv3(args.model, (yolo_dim, yolo_dim))


    #CAMERA
    cam = Camera(args)
    cam.open()
    if not cam.is_opened:
        sys.exit('Failed to open camera!')
    cam.start()

    #CAM-WINDOW
    open_window(WINDOW_NAME, args.image_width, args.image_height, 'DEEPFAMILY PROJECT - TensorRT YOLOv3')
    vis = BBoxVisualization(cls_dict)

    #DETECT-LOOP
    loop_and_detect(cam, trt_yolov3, conf_th=0.95, vis=vis)
    #loop_and_detect(cam, trt_yolov3, conf_th=0.95)

    cam.stop()
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
