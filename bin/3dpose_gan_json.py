import chainer
import cv2 as cv
import numpy as np
import argparse

import sys
import os
import json

import viz
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import imageio

import logging
import datetime
import openpose_utils
import copy

import evaluation_util

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import projection_gan

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

level = {0:logging.ERROR,
            1:logging.WARNING,
            2:logging.INFO,
            3:logging.DEBUG}

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

    # 3d-pose-baseline の日付+indexディレクトリを基準とする
    subdir = args.base_target

    frame3d_dir = "{0}/frame3d_gan".format(subdir)
    os.makedirs(frame3d_dir)

    #関節位置情報ファイル
    posf = open(subdir +'/pos_gan.txt', 'w')

    #正規化済みOpenpose位置情報ファイル
    smoothedf = open(subdir +'/smoothed_gan.txt', 'w')

    start_frame_index, smoothed = openpose_utils.read_openpose_json(args.input, 0)

    before_pose = None
    png_lib = []
    for n, (frame, xy) in enumerate(smoothed.items()):
        logger.info("calc idx {0}, frame {1}".format(0, frame))

        logger.debug("xy")
        logger.debug(xy)

        points = []
        for o in range(0,len(xy),2):
            points.append(np.array( [xy[o], xy[o+1]] ))

        logger.debug("points pre 36m")
        logger.debug(points)

        BODY_PARTS, POSE_PAIRS = parts(args)

        # Openpose位置情報をとりあえず出力する
        for poi in points:
#            logger.debug(poi)
#            logger.debug(poi.dtype)
#            logger.debug(poi.dtype == 'object')
            if poi.dtype == 'object':
                # logger.debug('poi is None')
                pass
            else:
#                logger.debug(' ' + str(poi[0]) + ' ' + str(poi[1]))
                smoothedf.write(' ' + str(poi[0]) + ' ' + str(poi[1]))

        smoothedf.write("\n")

        # 2d→3dに変換
        points = to36M(points, BODY_PARTS)
        logger.debug("points after 36m")
        logger.debug(points)
        points = np.reshape(points, [1, -1]).astype('f')
        logger.debug("points reshape 36m")
        logger.debug(points)
        points_norm = projection_gan.pose.dataset.pose_dataset.pose_dataset_base.Normalization.normalize_2d(points)
        logger.debug("points_norm")
        logger.debug(points_norm)
        poses3d = create_pose(model, points_norm)
        logger.debug("poses3d")
        logger.debug(poses3d)

        # Plot 3d predictions
        subplot_idx, exidx = 1, 1
        gs1 = gridspec.GridSpec(1, 1)
        gs1.update(wspace=-0.00, hspace=0.05)  # set the spacing between axes.
        ax = plt.subplot(gs1[subplot_idx - 1], projection='3d')
        ax.view_init(18, 280)

        logger.debug(np.min(poses3d))

        if np.min(poses3d) < -1000 and before_pose is not None:
            poses3d = before_pose

        p3d = poses3d

        xs = p3d[:, 0::3]
        ys = p3d[:, 1::3]
        zs = p3d[:, 2::3]

        # 拡大する
        xs *= 600
        ys *= 600
        zs *= 600

        # 画像の出力だけYZ反転させる
        p3d_copy = copy.deepcopy(p3d)
        p3d_copy[:, 1::3] *= -1
        p3d_copy[:, 2::3] *= -1

        # 3D画像を出力する
        if level[args.verbose] <= logging.INFO:
            # d = 30
            # img = evaluation_util.create_img_xyz(xs, ys, zs, np.pi * d / 180.)
            # cv.imwrite(os.path.join(frame3d_dir, "out_{0:012d}_{0:03d}_degree.png".format(n, d)), img)
            
            viz.show3Dpose(p3d_copy, ax, lcolor="#9b59b6", rcolor="#2ecc71", add_labels=True)

            # 各フレームの単一視点からのはINFO時のみ
            pngName = os.path.join(frame3d_dir, "3d_gan_{0:012d}.png".format(n))
            plt.savefig(pngName)
            png_lib.append(imageio.imread(pngName))            
            before_pose = poses3d

        # 各フレームの角度別出力はデバッグ時のみ
        if level[args.verbose] == logging.DEBUG:

            for azim in [0, 45, 90, 135, 180, 225, 270, 315, 360]:
                ax2 = plt.subplot(gs1[subplot_idx - 1], projection='3d')
                ax2.view_init(18, azim)
                viz.show3Dpose(p3d, ax2, lcolor="#FF0000", rcolor="#0000FF", add_labels=True)

                pngName2 = os.path.join(frame3d_dir, "debug_{0:012d}_{1:03d}.png".format(n, azim))
                plt.savefig(pngName2)

        # 3D関節位置情報を出力する
        for o in range(0,len(p3d[0]),3):
            logger.debug(str(o) + " "+ str(p3d[0][o]) +" "+ str(p3d[0][o+2]) +" "+ str(p3d[0][o+1] * -1) + ", ")
            posf.write(str(o) + " "+ str(p3d[0][o]) +" "+ str(p3d[0][o+2]) +" "+ str(p3d[0][o+1] * -1) + ", ")
            
        posf.write("\n")

        # # 各角度の出力はデバッグ時のみ
        # if level[args.verbose] == logging.DEBUG:
        #     deg = 15
        #     for d in range(0, 360 + deg, deg):
        #         img = evaluation_util.create_projection_img(pose, np.pi * d / 180.)
        #         cv.imwrite(os.path.join(out_sub_dir, "rot_{0:03d}_degree.png".format(d)), img)

        n += 1

    smoothedf.close()
    posf.close()

    # INFO時は、アニメーションGIF生成
    if level[args.verbose] <= logging.INFO:
        logger.info("creating Gif {0}/movie_smoothing_gan.gif, please Wait!".format(subdir))
        imageio.mimsave('{0}/movie_smoothing_gan.gif'.format(subdir), png_lib, fps=30)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Openpose result json (COCO)')
    parser.add_argument('--proto2d', help='Path to .prototxt', required=True)
    parser.add_argument('--model2d', help='Path to .caffemodel', required=True)
    parser.add_argument('--thr', default=0.1, type=float, help='Threshold value for pose parts heat map')
    parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
    parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')
    parser.add_argument('--inf_engine', action='store_true',
                        help='Enable Intel Inference Engine computational backend. '
                             'Check that plugins folder is in LD_LIBRARY_PATH environment variable')
    parser.add_argument('--lift_model', type=str, required=True)
    parser.add_argument('--dataset', type=str, default="COCO")
    parser.add_argument('--person_idx', type=int, default=1)
    parser.add_argument('--base-target', dest='base_target', type=str,
                        help='target directory (3d-pose-baseline-vmd)')

    parser.add_argument('--activate_func', type=str, default='leaky_relu')
    parser.add_argument('--use_bn', action="store_true")

    parser.add_argument('--verbose', dest='verbose', type=int,
                        default=2,
                        help='logging level')

    args = parser.parse_args()

    # ログレベル設定
    logger.setLevel(level[args.verbose])

    main(args)
