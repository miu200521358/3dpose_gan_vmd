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
    smoothed = {}
    _past_tmp_points = []
    _past_tmp_data = []
    _tmp_data = []
    ### extract x,y and ignore confidence score
    is_started = False
    start_frame_index = 0
    for file_name in json_files:
        logger.debug("reading {0}".format(file_name))
        _file = os.path.join(openpose_output_dir, file_name)
        if not os.path.isfile(_file): raise Exception("No file found!!, {0}".format(_file))
        data = json.load(open(_file))

        # 12桁の数字文字列から、フレームINDEX取得
        frame_indx = int(re.findall("(\d{12})", file_name)[0])
        
        if frame_indx <= 0 or is_started == False:
            # 最初のフレームはそのまま登録するため、INDEXをそのまま指定
            _tmp_data = data["people"][idx]["pose_keypoints_2d"]
            # 開始したらフラグを立てる
            is_started = True
            # 開始フレームインデックス保持
            start_frame_index = frame_indx
        else:
            # 前フレームと一番近い人物データを採用する
            past_xy = cache[frame_indx - 1]

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
        #ignore confidence score
        for o in range(0,len(_data),3):
            xy.append(_data[o])
            xy.append(_data[o+1])
        
        logger.debug("found {0} for frame {1}".format(xy, str(frame_indx)))
        #add xy to frame
        cache[frame_indx] = xy

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

    # create frame blocks
    first_frame_block = [int(re.findall("(\d{12})", o)[0]) for o in json_files[:4]]
    last_frame_block = [int(re.findall("(\d{12})", o)[0]) for o in json_files[-4:]]

    ### smooth by median value, n frames 
    for frame, xy in cache.items():

        # create neighbor array based on frame index
        forward, back = ([] for _ in range(2))

        # joints x,y array
        _len = len(xy) # 36

        # create array of parallel frames (-3<n>3)
        for neighbor in range(1,4):
            # first n frames, get value of xy in postive lookahead frames(current frame + 3)
            if frame in first_frame_block:
                # logger.debug ("first_frame_block: len(cache)={0}, frame={1}, neighbor={2}".format(len(cache), frame, neighbor))
                forward += cache[frame+neighbor]
            # last n frames, get value of xy in negative lookahead frames(current frame - 3)
            elif frame in last_frame_block:
                # logger.debug ("last_frame_block: len(cache)={0}, frame={1}, neighbor={2}".format(len(cache), frame, neighbor))
                back += cache[frame-neighbor]
            else:
                # between frames, get value of xy in bi-directional frames(current frame -+ 3)     
                forward += cache[frame+neighbor]
                back += cache[frame-neighbor]

        # build frame range vector 
        frames_joint_median = [0 for i in range(_len)]
        # more info about mapping in src/data_utils.py
        # for each 18joints*x,y  (x1,y1,x2,y2,...)~36 
        for x in range(0,_len,2):
            # set x and y
            y = x+1
            if frame in first_frame_block:
                # get vector of n frames forward for x and y, incl. current frame
                x_v = [xy[x], forward[x], forward[x+_len], forward[x+_len*2]]
                y_v = [xy[y], forward[y], forward[y+_len], forward[y+_len*2]]
            elif frame in last_frame_block:
                # get vector of n frames back for x and y, incl. current frame
                x_v =[xy[x], back[x], back[x+_len], back[x+_len*2]]
                y_v =[xy[y], back[y], back[y+_len], back[y+_len*2]]
            else:
                # get vector of n frames forward/back for x and y, incl. current frame
                # median value calc: find neighbor frames joint value and sorted them, use numpy median module
                # frame[x1,y1,[x2,y2],..]frame[x1,y1,[x2,y2],...], frame[x1,y1,[x2,y2],..]
                #                 ^---------------------|-------------------------^
                x_v =[xy[x], forward[x], forward[x+_len], forward[x+_len*2],
                        back[x], back[x+_len], back[x+_len*2]]
                y_v =[xy[y], forward[y], forward[y+_len], forward[y+_len*2],
                        back[y], back[y+_len], back[y+_len*2]]

            # get median of vector
            x_med = np.median(sorted(x_v))
            y_med = np.median(sorted(y_v))

            # holding frame drops for joint
            if not x_med:
                # allow fix from first frame
                if frame:
                    # get x from last frame
                    x_med = smoothed[frame-1][x]
            # if joint is hidden y
            if not y_med:
                # allow fix from first frame
                if frame:
                    # get y from last frame
                    y_med = smoothed[frame-1][y]

            # logger.debug("old X {0} sorted neighbor {1} new X {2}".format(xy[x],sorted(x_v), x_med))
            # logger.debug("old Y {0} sorted neighbor {1} new Y {2}".format(xy[y],sorted(y_v), y_med))

            # build new array of joint x and y value
            frames_joint_median[x] = x_med 
            frames_joint_median[x+1] = y_med 


        smoothed[frame] = frames_joint_median

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

