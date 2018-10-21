import chainer
import cv2 as cv
import numpy as np
import argparse

import sys
import os
import json

import re
import logging
import datetime
from collections import Counter

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

import evaluation_util

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import projection_gan

def to36M(bones, body_parts):
    H36M_JOINTS_17 = [
        'Hip',
        'RHip',
        'RKnee',
        'RFoot',
        'LHip',
        'LKnee',
        'LFoot',
        'Spine',
        'Thorax',
        'Neck/Nose',
        'Head',
        'LShoulder',
        'LElbow',
        'LWrist',
        'RShoulder',
        'RElbow',
        'RWrist',
    ]

    re_bones = []
    for b in bones:
        if b.dtype == 'object':
            # オブジェクト（None）の場合、0を与える
            re_bones.append(np.zeros(2))
        else:
            re_bones.append(b)

    adjusted_bones = []
    for name in H36M_JOINTS_17:
        if not name in body_parts:
            if name == 'Hip':
                adjusted_bones.append((re_bones[body_parts['RHip']] + re_bones[body_parts['LHip']]) / 2)
            elif name == 'RFoot':
                adjusted_bones.append(re_bones[body_parts['RAnkle']])
            elif name == 'LFoot':
                adjusted_bones.append(re_bones[body_parts['LAnkle']])
            elif name == 'Spine':
                adjusted_bones.append(
                    (
                            re_bones[body_parts['RHip']] + re_bones[body_parts['LHip']]
                            + re_bones[body_parts['RShoulder']] + re_bones[body_parts['LShoulder']]
                    ) / 4
                )
            elif name == 'Thorax':
                adjusted_bones.append(
                    (
                            + re_bones[body_parts['RShoulder']] + re_bones[body_parts['LShoulder']]
                    ) / 2
                )
            elif name == 'Head':
                thorax = (
                                 + re_bones[body_parts['RShoulder']] + re_bones[body_parts['LShoulder']]
                         ) / 2
                adjusted_bones.append(
                    thorax + (
                            re_bones[body_parts['Nose']] - thorax
                    ) * 2
                )
            elif name == 'Neck/Nose':
                adjusted_bones.append(re_bones[body_parts['Nose']])
            else:
                raise Exception(name)
        else:
            adjusted_bones.append(re_bones[body_parts[name]])

    return adjusted_bones


def parts(args):
    if args.dataset == 'COCO':
        BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                      "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                      "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                      "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

        POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                      ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                      ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                      ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                      ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]
    else:
        assert (args.dataset == 'MPI')
        BODY_PARTS = {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                      "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                      "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                      "Background": 15}

        POSE_PAIRS = [["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                      ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                      ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                      ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"]]
    return BODY_PARTS, POSE_PAIRS


class OpenPose(object):
    """
    This implementation is based on https://github.com/opencv/opencv/blob/master/samples/dnn/openpose.py
    """

    def __init__(self, args):
        self.net = cv.dnn.readNetFromCaffe(args.proto2d, args.model2d)
        if args.inf_engine:
            self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_INFERENCE_ENGINE)

    def predict(self, args, frame):

        inWidth = args.width
        inHeight = args.height

        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        inp = cv.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                   (0, 0, 0), swapRB=False, crop=False)
        self.net.setInput(inp)
        out = self.net.forward()

        BODY_PARTS, POSE_PAIRS = parts(args)

        points = []
        for i in range(len(BODY_PARTS)):
            # Slice heatmap of corresponging body's part.
            heatMap = out[0, i, :, :]

            # Originally, we try to find all the local maximums. To simplify a sample
            # we just find a global one. However only a single pose at the same time
            # could be detected this way.
            _, conf, _, point = cv.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]

            # Add a point if it's confidence is higher than threshold.
            points.append((x, y) if conf > args.thr else None)
        return points


def create_pose(model, points):
    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        x = points[:, 0::2]
        y = points[:, 1::2]
        z_pred = model(points).data

        pose = np.stack((x, y, z_pred), axis=-1)
        pose = np.reshape(pose, (len(points), -1))

        return pose


def main(args):
    model = evaluation_util.load_model(vars(args))
    chainer.serializers.load_npz(args.lift_model, model)
#    cap = cv.VideoCapture(args.input if args.input else 0)

    out_directory = "demo_out"

    out_vmd_directory = "vmd_out"
    os.makedirs(out_vmd_directory, exist_ok=True)

    #Openpose位置情報ファイル
    smoothedf = open(os.path.join(out_vmd_directory, 'smoothed.txt'), 'w')

    #関節位置情報ファイル
    posf = open(os.path.join(out_vmd_directory, 'pos.txt'), 'w')

    smoothed = read_openpose_json(args.openpose_output_dir, 0)

    for n, (frame, xy) in enumerate(smoothed.items()):
        logger.info("calc idx {0}, frame {1}".format(0, frame))

        print("xy")
        print(xy)

        points = []
        for o in range(0,len(xy),2):
            points.append(np.array( [xy[o], xy[o+1]] ))

        print("points pre 36m")
        print(points)

        BODY_PARTS, POSE_PAIRS = parts(args)

        # Openpose位置情報をとりあえず出力する
        for poi in points:
#            print(poi)
#            print(poi.dtype)
#            print(poi.dtype == 'object')
            if poi.dtype == 'object':
                # print('poi is None')
                pass
            else:
#                print(' ' + str(poi[0]) + ' ' + str(poi[1]))
                smoothedf.write(' ' + str(poi[0]) + ' ' + str(poi[1]))

        smoothedf.write("\n")

        # 2d→3dに変換
        points = to36M(points, BODY_PARTS)
        print("points after 36m")
        print(points)
        points = np.reshape(points, [1, -1]).astype('f')
        print("points reshape 36m")
        print(points)
        points_norm = projection_gan.pose.dataset.pose_dataset.pose_dataset_base.Normalization.normalize_2d(points)
        print("points_norm")
        print(points_norm)
        pose = create_pose(model, points_norm)
        print("pose")
        print(pose)

        # 3D関節位置情報を出力する
        for o in range(0,len(pose[0]),3):
            print(str(o) + " "+ str(pose[0][o]) +" "+ str(pose[0][o+2]) +" "+ str(pose[0][o+1] * -1) + ", ")
            posf.write(str(o) + " "+ str(pose[0][o]) +" "+ str(pose[0][o+2]) +" "+ str(pose[0][o+1] * -1) + ", ")
            
        posf.write("\n")

        out_sub_dir = os.path.join(out_directory, "{0:012d}".format(n))
        os.makedirs(out_sub_dir, exist_ok=True)

        out_img = evaluation_util.create_img(points[0], frame)
        cv.imwrite(os.path.join(out_sub_dir, 'openpose_detect.jpg'), out_img)
        deg = 15
        for d in range(0, 360 + deg, deg):
            img = evaluation_util.create_projection_img(pose, np.pi * d / 180.)
            cv.imwrite(os.path.join(out_sub_dir, "rot_{0:03d}_degree.png".format(d)), img)

        n += 1

    smoothedf.close()
    posf.close()

def read_openpose_json(openpose_output_dir, idx, smooth=True, *args):
    # openpose output format:
    # [x1,y1,c1,x2,y2,c2,...]
    # ignore confidence score, take x and y [x1,y1,x2,y2,...]

    logger.info("start reading data")
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
        frame_indx = re.findall("(\d{12})", file_name)
        
        if int(frame_indx[0]) <= 0:
            # 最初のフレームはそのまま登録するため、INDEXをそのまま指定
            _tmp_data = data["people"][idx]["pose_keypoints_2d"]
        else:
            # 前フレームと一番近い人物データを採用する
            past_xy = cache[int(frame_indx[0]) - 1]

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

        logger.debug("found {0} for frame {1}".format(xy, str(int(frame_indx[0]))))
        #add xy to frame
        cache[int(frame_indx[0])] = xy

    # plt.figure(1)
    # drop_curves_plot = show_anim_curves(cache, plt)
    # pngName = '{0}/dirty_plot.png'.format(subdir)
    # drop_curves_plot.savefig(pngName)

    # exit if no smoothing
    if not smooth:
        # return frames cache incl. 18 joints (x,y)
        return cache

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
                # print ("first_frame_block: len(cache)={0}, frame={1}, neighbor={2}".format(len(cache), frame, neighbor))
                forward += cache[frame+neighbor]
            # last n frames, get value of xy in negative lookahead frames(current frame - 3)
            elif frame in last_frame_block:
                # print ("last_frame_block: len(cache)={0}, frame={1}, neighbor={2}".format(len(cache), frame, neighbor))
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Path to image or video. Skip to capture frames from camera')
    parser.add_argument('--proto2d', help='Path to .prototxt', required=True)
    parser.add_argument('--model2d', help='Path to .caffemodel', required=True)
    parser.add_argument('--thr', default=0.1, type=float, help='Threshold value for pose parts heat map')
    parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
    parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')
    parser.add_argument('--inf_engine', action='store_true',
                        help='Enable Intel Inference Engine computational backend. '
                             'Check that plugins folder is in LD_LIBRARY_PATH environment variable')
    parser.add_argument('--lift_model', type=str, required=True)
    parser.add_argument('--openpose_output_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, default="COCO")

    parser.add_argument('--activate_func', type=str, default='leaky_relu')
    parser.add_argument('--use_bn', action="store_true")
    args = parser.parse_args()
    main(args)
