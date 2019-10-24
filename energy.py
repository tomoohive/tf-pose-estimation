import argparse
import logging
import sys
import time

from tf_pose import common
import cv2
import numpy as np
import pandas as pd
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimatorRun')
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def get_judgement_human(humans):
    #判定する人を一人に絞る(画像上で一番大きい人)
    judgement_human_idx = 0
    judgement_human_size = 0

    #複数人検出されている時
    if len(humans) > 1:
        for idx, human in enumerate(humans):
            d=0
            #首の有無
            if 0 not in human.body_parts.keys():
                continue
            for i in human.body_parts.keys():
                if i == 0:
                    continue
                d += abs(human.body_parts[i].x - human.body_parts[0].x) \
                    + abs(human.body_parts[i].y - human.body_parts[0].y)
                if d > judgement_human_size:
                    judgement_human_idx = idx
                    judgement_human_size = d

    return humans[judgement_human_idx]

def get_position_list(judgement_human):
    judgement_human_pos = {}

    for i in judgement_human.body_parts.keys():
        judgement_human_pos[str(i)] = (judgement_human.body_parts[i].x, judgement_human.body_parts[i].y)
    
    return judgement_human_pos

def calculatePoseSimiler(judgement_human_pos, target):
    d = 0
    for i in best_human_pos.keys() & target.keys():
        d += (abs(target[str(i)][0] - best_human_pos[str(i)][0]) \
            + abs(target[str(i)][1] - best_human_pos[str(i)][1]))
    d += len(set(best_human_pos.keys()).symmetric_difference(target.keys()))
    return d/18

def calculateHumanPoseFrame(image_path, model, resize, resize_out_ratio=4.0):
    w, h = model_wh(resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))


    image = common.read_imgfile(image_path, None, None)
    if image is None:
        logger.error('Image can not be read, path=%s' % image_path)
        sys.exit(-1)

    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
    if len(humans) != 0:
        # 人物特定
        judgement_human = get_best_human(humans)
        # 座標取得
        judgement_human_pos = pos_list(judgement_human)
    else:
        human = None
        judgement_human_pos = {}
    return humans, judgement_human, judgement_human_pos

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('--image', type=str, default='./images/p1.jpg')
    parser.add_argument('--model', type=str, default='cmu',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. '
                             'default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    args = parser.parse_args()

    calculateHumanPoseFrame(image_path = args.image, model = args.model, resize = args.resize)