#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# openpose_utils.py

import os
import json
import re
import numpy as np
from collections import Counter

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_openpose_json(openpose_output_dir, idx, is_debug=False):
    if is_debug == True:
        logger.setLevel(logging.DEBUG)

    # openpose output format:
    # [x1,y1,c1,x2,y2,c2,...]
    # ignore confidence score, take x and y [x1,y1,x2,y2,...]

    logger.info("start reading data: %s", openpose_output_dir)
    #load json files
    json_files = os.listdir(openpose_output_dir)
    # check for other file types
    json_files = sorted([filename for filename in json_files if filename.endswith(".json")])
    cache = {}
    cache_confidence = {}
    smoothed = {}
    _past_tmp_points = []
    _past_tmp_data = []
    _tmp_data = []
    ### extract x,y and ignore confidence score
    is_started = False
    start_frame_index = 0
    end_frame_index = 0
    # 実際の処理用フレームINDEX
    frame_idx = 0
    for file_name in json_files:
        logger.debug("reading {0}".format(file_name))
        _file = os.path.join(openpose_output_dir, file_name)
        if not os.path.isfile(_file): raise Exception("No file found!!, {0}".format(_file))
        data = json.load(open(_file))

        # 12桁の数字文字列から、フレームINDEX取得
        frame_idx = int(re.findall("(\d{12})", file_name)[0])
        
        if frame_idx <= 0 or is_started == False:
            # 最初のフレームはそのまま登録するため、INDEXをそのまま指定
            _tmp_data = data["people"][idx]["pose_keypoints_2d"]
            # 開始したらフラグを立てる
            is_started = True
            # 開始フレームインデックス保持
            start_frame_index = frame_idx
        else:
            # 前フレームと一番近い人物データを採用する
            past_xy = cache[frame_idx - 1]

            # データが取れていたら、そのINDEX数分配列を生成。取れてなかったら、とりあえずINDEX分確保
            target_num = len(data["people"]) if len(data["people"]) >= idx + 1 else idx + 1
            # 同一フレーム内の全人物データを一旦保持する
            _tmp_points = [[0 for i in range(target_num)] for j in range(36)]
            
            # logger.debug("_past_tmp_points")
            # logger.debug(_past_tmp_points)

            for _data_idx in range(idx + 1):
                if len(data["people"]) - 1 < _data_idx:
                    for o in range(len(_past_tmp_points)):
                        # 人物データが取れていない場合、とりあえず前回のをコピっとく
                        # logger.debug("o={0}, _data_idx={1}".format(o, _data_idx))
                        # logger.debug(_tmp_points)
                        # logger.debug(_tmp_points[o][_data_idx])
                        # logger.debug(_past_tmp_points[o][_data_idx])
                        _tmp_points[o][_data_idx] = _past_tmp_points[o][_data_idx]
                    
                    # データも前回のを引き継ぐ
                    _tmp_data = _past_tmp_data
                else:
                    # ちゃんと取れている場合、データ展開
                    _tmp_data = data["people"][_data_idx]["pose_keypoints_2d"]

                    n = 0
                    for o in range(0,len(_tmp_data),3):
                        # logger.debug("o: {0}".format(o))
                        # logger.debug("len(_tmp_points): {0}".format(len(_tmp_points)))
                        # logger.debug("len(_tmp_points[o]): {0}".format(len(_tmp_points[n])))
                        # logger.debug("_tmp_data[o]")
                        # logger.debug(_tmp_data[o])
                        _tmp_points[n][_data_idx] = _tmp_data[o]
                        n += 1
                        _tmp_points[n][_data_idx] = _tmp_data[o+1]
                        n += 1            

                    # とりあえず前回のを保持
                    _past_tmp_data = _tmp_data            
                    _past_tmp_points = _tmp_points

            # logger.debug("_tmp_points")
            # logger.debug(_tmp_points)

            # 各INDEXの前回と最も近い値を持つINDEXを取得
            nearest_idx_list = []
            for n, plist in enumerate(_tmp_points):
                nearest_idx_list.append(get_nearest_idx(plist, past_xy[n]))

            most_common_idx = Counter(nearest_idx_list).most_common(1)
            
            # 最も多くヒットしたINDEXを処理対象とする
            target_idx = most_common_idx[0][0]
            logger.debug("target_idx={0}".format(target_idx))

        _data = _tmp_data

        xy = []
        confidence = []
        for o in range(0,len(_data),3):
            xy.append(_data[o])
            xy.append(_data[o+1])
            confidence.append(_data[o+2])
        
        logger.debug("found {0} for frame {1}".format(xy, str(frame_idx)))
        #add xy to frame
        cache[frame_idx] = xy
        cache_confidence[frame_idx] = confidence
        end_frame_index = frame_idx

    # plt.figure(1)
    # drop_curves_plot = show_anim_curves(cache, plt)
    # pngName = '{0}/dirty_plot.png'.format(subdir)
    # drop_curves_plot.savefig(pngName)

    # # exit if no smoothing
    # if not smooth:
    #     # return frames cache incl. 18 joints (x,y)
    #     return cache

    if len(json_files) == 1:
        logger.info("found single json file")
        # return frames cache incl. 18 joints (x,y) on single image\json
        return cache

    if len(json_files) <= 8:
        raise Exception("need more frames, min 9 frames/json files for smoothing!!!")

    logger.info("start smoothing")

    # last frame of data
    last_frame = [start_frame_index-1 for i in range(18)]

    # threshold of confidence
    confidence_th = 0.3

    ### smooth by median value, n frames 
    for frame, xy in cache.items():
        confidence = cache_confidence[frame]

        # joints x,y array
        _len = len(xy) # 36

        # frame range
        smooth_n = 7 # odd number
        one_side_n = int((smooth_n - 1)/2)
        one_side_n = min([one_side_n, frame-start_frame_index, end_frame_index-frame])

        smooth_start_frame = frame - one_side_n
        smooth_end_frame = frame + one_side_n

        # build frame range vector 
        frames_joint_median = [0 for i in range(_len)]
        # more info about mapping in src/data_utils.py
        # for each 18joints*x,y  (x1,y1,x2,y2,...)~36 
        for x in range(0,_len,2):
            # set x and y
            y = x+1
            joint_no = int(x / 2)

            x_v = []
            y_v = []
            for neighbor in range(smooth_start_frame, smooth_end_frame + 1):
                if cache_confidence[neighbor][joint_no] >= confidence_th:
                    x_v.append(cache[neighbor][x])
                    y_v.append(cache[neighbor][y])

            if len(x_v) >= 1:
                # 配列の長さを奇数にする
                if len(x_v) % 2 == 0:
                    x_v.append(cache[frame][x])
                    y_v.append(cache[frame][y])

                # get median of vector
                x_med = np.median(sorted(x_v))
                y_med = np.median(sorted(y_v))

                # 前のフレームが欠損している場合、今回のフレームのデータと前回最後に取得できたデータで線形補間する
                if last_frame[joint_no] != frame -1:
                    if last_frame[joint_no] < start_frame_index:
                        # 過去に一度もデータを取得できなかった場合
                        last_value_x = x_med
                        last_value_y = y_med
                    else:
                        last_value_x = smoothed[last_frame[joint_no] - start_frame_index][x]
                        last_value_y = smoothed[last_frame[joint_no] - start_frame_index][y]

                    for frame_linear_in in range(last_frame[joint_no] + 1, frame):
                        logger.debug("frame_linear_in: %s, start_frame_index: %s, x: %s, last_value_x: %s, x_med: %s, joint_no: %s, frame: %s, last_frame[joint_no]: %s", frame_linear_in, start_frame_index, x, last_value_x, x_med, joint_no, frame, last_frame[joint_no])
                        smoothed[frame_linear_in - start_frame_index][x] = last_value_x + (x_med - last_value_x) * (frame_linear_in - last_frame[joint_no]) / (frame - last_frame[joint_no])
                        smoothed[frame_linear_in - start_frame_index][y] = last_value_y + (y_med - last_value_y) * (frame_linear_in - last_frame[joint_no]) / (frame - last_frame[joint_no])

                last_frame[joint_no] = frame

            else:
                # holding frame drops for joint
                # allow fix from first frame
                if frame > start_frame_index:
                    # get x from last frame
                    # logger.info("frame %s, x %s", frame, x)
                    x_med = smoothed[frame - start_frame_index -1][x]
                    # get y from last frame
                    y_med = smoothed[frame - start_frame_index -1][y]
                else:
                    x_med = 0
                    y_med = 0

            # logger.debug("old X {0} sorted neighbor {1} new X {2}".format(xy[x],sorted(x_v), x_med))
            # logger.debug("old Y {0} sorted neighbor {1} new Y {2}".format(xy[y],sorted(y_v), y_med))

            # build new array of joint x and y value
            frames_joint_median[x] = x_med 
            frames_joint_median[x+1] = y_med 


        smoothed[frame - start_frame_index] = frames_joint_median

    # return frames cache incl. smooth 18 joints (x,y)
    return start_frame_index, smoothed

def get_nearest_idx(target_list, num):
    """
    概要: リストからある値に最も近い値のINDEXを返却する関数
    @param target_list: データ配列
    @param num: 対象値
    @return 対象値に最も近い値のINDEX
    """

    # logger.debug(target_list)
    # logger.debug(num)

    # リスト要素と対象値の差分を計算し最小値のインデックスを取得
    idx = np.abs(np.asarray(target_list) - num).argmin()
    return idx

