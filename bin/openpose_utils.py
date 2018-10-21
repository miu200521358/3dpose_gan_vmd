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
    for file_name in json_files:
        logger.debug("reading {0}".format(file_name))
        _file = os.path.join(openpose_output_dir, file_name)
        if not os.path.isfile(_file): raise Exception("No file found!!, {0}".format(_file))
        data = json.load(open(_file))

        # 12桁の数字文字列から、フレームINDEX取得
        frame_indx = int(re.findall("(\d{12})", file_name)[0])
        
        if frame_indx <= 0:
            # 最初のフレームはそのまま登録するため、INDEXをそのまま指定
            _tmp_data = data["people"][idx]["pose_keypoints_2d"]
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
        
        if frame_indx > 0:
            # 1F前のxy座標
            past_xy = cache[frame_indx - 1]

            # # 手 -------------------------------------------------------

            # # 右肩 <--> 右肩 の距離差分
            # diff_RRShoulder_x = abs(np.diff([xy[4], past_xy[4]]))
            # diff_RRShoulder_y = abs(np.diff([xy[5], past_xy[5]]))
            # # 右ひじ <--> 右ひじ の距離差分
            # diff_RRElbow_x = abs(np.diff([xy[6], past_xy[6]]))
            # diff_RRElbow_y = abs(np.diff([xy[7], past_xy[7]]))
            # # 右手首 <--> 右手首 の距離差分
            # diff_RRWrist_x = abs(np.diff([xy[8], past_xy[8]]))
            # diff_RRWrist_y = abs(np.diff([xy[9], past_xy[9]]))

            # # 右肩 <--> 左肩 の距離差分
            # diff_RLShoulder_x = abs(np.diff([xy[4], past_xy[10]])[0])
            # diff_RLShoulder_y = abs(np.diff([xy[5], past_xy[11]])[0])
            # # 右ひじ <--> 左ひじ の距離差分
            # diff_RLElbow_x = abs(np.diff([xy[6], past_xy[12]])[0])
            # diff_RLElbow_y = abs(np.diff([xy[7], past_xy[13]])[0])
            # # 右手首 <--> 左手首 の距離差分
            # diff_RLWrist_x = abs(np.diff([xy[8], past_xy[14]])[0])
            # diff_RLWrist_y = abs(np.diff([xy[9], past_xy[15]])[0])

            # logger.debug("diff_RRShoulder_x: %d <--> %s", diff_RRShoulder_x, diff_RLShoulder_x)
            # logger.debug("diff_RRShoulder_y: %d <--> %s", diff_RRShoulder_y, diff_RLShoulder_y)
            # logger.debug("diff_RRElbow_x: %d <--> %s", diff_RRElbow_x, diff_RLElbow_x)
            # logger.debug("diff_RRElbow_y: %d <--> %s", diff_RRElbow_y, diff_RLElbow_y)
            # logger.debug("diff_RRWrist_x: %d <--> %s", diff_RRWrist_x, diff_RLWrist_x)
            # logger.debug("diff_RRWrist_y: %d <--> %s", diff_RRWrist_y, diff_RLWrist_x)

            # # 差分ポイント制で判定
            # diff_point = 0
            # diff_point += 1 if diff_RRShoulder_x > diff_RLShoulder_x else 0
            # diff_point += 1 if diff_RRShoulder_y > diff_RLShoulder_y else 0
            # diff_point += 1 if diff_RRElbow_x > diff_RLElbow_x else 0
            # diff_point += 1 if diff_RRElbow_y > diff_RLElbow_y else 0
            # diff_point += 1 if diff_RRWrist_x > diff_RLWrist_x else 0
            # diff_point += 1 if diff_RRWrist_y > diff_RLWrist_y else 0

            # logger.debug("diff_point: %d", diff_point)

            # # 半分以上が差分で、右同士の方が、左右よりも差が大きい場合、反転とみなす
            # if diff_point == 6:
            #     logger.info("＊手左右反転 {0}".format(frame_indx))
            #     logger.info("diff_RRShoulder_x: %d <--> %s", diff_RRShoulder_x, diff_RLShoulder_x)
            #     logger.info("diff_RRShoulder_y: %d <--> %s", diff_RRShoulder_y, diff_RLShoulder_y)
            #     logger.info("diff_RRElbow_x: %d <--> %s", diff_RRElbow_x, diff_RLElbow_x)
            #     logger.info("diff_RRElbow_y: %d <--> %s", diff_RRElbow_y, diff_RLElbow_y)
            #     logger.info("diff_RRWrist_x: %d <--> %s", diff_RRWrist_x, diff_RLWrist_x)
            #     logger.info("diff_RRWrist_y: %d <--> %s", diff_RRWrist_y, diff_RLWrist_x)

            #     # 右手系に左手の値を設定
            #     RShoulder_x = xy[10]
            #     RShoulder_y = xy[11]
            #     RElbow_x = xy[12]
            #     RElbow_y = xy[13]
            #     RWrist_x = xy[14]
            #     RWrist_y = xy[15]

            #     # 左手系に右手の値を設定
            #     LShoulder_x = xy[4]
            #     LShoulder_y = xy[5]
            #     LElbow_x = xy[6]
            #     LElbow_y = xy[7]
            #     LWrist_x = xy[8]
            #     LWrist_y = xy[9]

            #     # 取得し直した値を再設定
            #     xy[4] = RShoulder_x
            #     xy[5] = RShoulder_y
            #     xy[6] = RElbow_x
            #     xy[7] = RElbow_y
            #     xy[8] = RWrist_x
            #     xy[9] = RWrist_y
            #     xy[10] = LShoulder_x
            #     xy[11] = LShoulder_y
            #     xy[12] = LElbow_x
            #     xy[13] = LElbow_y
            #     xy[14] = LWrist_x
            #     xy[15] = LWrist_y

            # # 足 -------------------------------------------------------

            # # 右尻 <--> 右尻 の距離差分
            # diff_RRHip_x = abs(np.diff([xy[16], past_xy[16]]))
            # diff_RRHip_y = abs(np.diff([xy[17], past_xy[17]]))
            # # 右ひざ <--> 右ひざ の距離差分
            # diff_RRKnee_x = abs(np.diff([xy[18], past_xy[18]]))
            # diff_RRKnee_y = abs(np.diff([xy[19], past_xy[19]]))
            # # 右足首 <--> 右足首 の距離差分
            # diff_RRAnkle_x = abs(np.diff([xy[20], past_xy[20]]))
            # diff_RRAnkle_y = abs(np.diff([xy[21], past_xy[21]]))

            # # 右尻 <--> 左尻 の距離差分
            # diff_RLHip_x = abs(np.diff([xy[16], past_xy[22]])[0])
            # diff_RLHip_y = abs(np.diff([xy[17], past_xy[23]])[0])
            # # 右ひざ <--> 左ひざ の距離差分
            # diff_RLKnee_x = abs(np.diff([xy[18], past_xy[24]])[0])
            # diff_RLKnee_y = abs(np.diff([xy[19], past_xy[25]])[0])
            # # 右足首 <--> 左足首 の距離差分
            # diff_RLAnkle_x = abs(np.diff([xy[20], past_xy[26]])[0])
            # diff_RLAnkle_y = abs(np.diff([xy[21], past_xy[27]])[0])

            # logger.debug("diff_RRHip_x: %d <--> %s", diff_RRHip_x, diff_RLHip_x)
            # logger.debug("diff_RRHip_y: %d <--> %s", diff_RRHip_y, diff_RLHip_y)
            # logger.debug("diff_RRKnee_x: %d <--> %s", diff_RRKnee_x, diff_RLKnee_x)
            # logger.debug("diff_RRKnee_y: %d <--> %s", diff_RRKnee_y, diff_RLKnee_y)
            # logger.debug("diff_RRAnkle_x: %d <--> %s", diff_RRAnkle_x, diff_RLAnkle_x)
            # logger.debug("diff_RRAnkle_y: %d <--> %s", diff_RRAnkle_y, diff_RLAnkle_x)

            # # 差分ポイント制で判定
            # diff_point = 0
            # diff_point += 1 if diff_RRHip_x > diff_RLHip_x else 0
            # diff_point += 1 if diff_RRHip_y > diff_RLHip_y else 0
            # diff_point += 1 if diff_RRKnee_x > diff_RLKnee_x else 0
            # diff_point += 1 if diff_RRKnee_y > diff_RLKnee_y else 0
            # diff_point += 1 if diff_RRAnkle_x > diff_RLAnkle_x else 0
            # diff_point += 1 if diff_RRAnkle_y > diff_RLAnkle_y else 0

            # logger.debug("diff_point: %d", diff_point)

            # # 半分以上が差分で、右同士の方が、左右よりも差が大きい場合、反転とみなす
            # if diff_point == 6:
            #     logger.info("＊足左右反転 {0}".format(frame_indx))
            #     logger.info("diff_RRHip_x: %d <--> %s", diff_RRHip_x, diff_RLHip_x)
            #     logger.info("diff_RRHip_y: %d <--> %s", diff_RRHip_y, diff_RLHip_y)
            #     logger.info("diff_RRKnee_x: %d <--> %s", diff_RRKnee_x, diff_RLKnee_x)
            #     logger.info("diff_RRKnee_y: %d <--> %s", diff_RRKnee_y, diff_RLKnee_y)
            #     logger.info("diff_RRAnkle_x: %d <--> %s", diff_RRAnkle_x, diff_RLAnkle_x)
            #     logger.info("diff_RRAnkle_y: %d <--> %s", diff_RRAnkle_y, diff_RLAnkle_x)

            #     # 右足系に左足の値を設定
            #     RHip_x = xy[22]
            #     RHip_y = xy[23]
            #     RKnee_x = xy[24]
            #     RKnee_y = xy[25]
            #     RAnkle_x = xy[26]
            #     RAnkle_y = xy[27]

            #     # 左足系に右足の値を設定
            #     LHip_x = xy[16]
            #     LHip_y = xy[17]
            #     LKnee_x = xy[18]
            #     LKnee_y = xy[19]
            #     LAnkle_x = xy[20]
            #     LAnkle_y = xy[21]

            #     # 取得し直した値を再設定
            #     xy[16] = RHip_x
            #     xy[17] = RHip_y
            #     xy[18] = RKnee_x
            #     xy[19] = RKnee_y
            #     xy[20] = RAnkle_x
            #     xy[21] = RAnkle_y
            #     xy[22] = LHip_x
            #     xy[23] = LHip_y
            #     xy[24] = LKnee_x
            #     xy[25] = LKnee_y
            #     xy[26] = LAnkle_x
            #     xy[27] = LAnkle_y

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
    return smoothed

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

