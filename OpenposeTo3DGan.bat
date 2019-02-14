@echo off
rem --- 
rem ---  OpenPose の jsonデータから 3Dデータに変換
rem ---  https://nico-opendata.jp/ja/casestudy/3dpose_gan/index.htmlを使用
rem --- 

rem ---  カレントディレクトリを実行先に変更
cd /d %~dp0

rem ---  INDEX別ディレクトリパス
echo INDEX別ディレクトリのフルパスを入力して下さい。({動画名}_json_{実行日時}_idx00)
echo pos.txtなどのあるディレクトリです。
echo この設定は半角英数字のみ設定可能で、必須項目です。
echo ,(カンマ)で5件まで設定可能です。
set TARGET_DIR=
set /P TARGET_DIR=■INDEX別ディレクトリパス: 
rem echo TARGET_DIR：%TARGET_DIR%

IF /I "%TARGET_DIR%" EQU "" (
    ECHO INDEX別ディレクトリパスが設定されていないため、処理を中断します。
    EXIT /B
)

rem ---  詳細ログ有無

echo --------------
echo 詳細なログを出すか、yes か no を入力して下さい。
echo 何も入力せず、ENTERを押下した場合、通常ログとモーションのアニメーションGIFを出力します。
echo 詳細ログの場合、各フレームごとのデバッグ画像も追加出力されます。（その分時間がかかります）
echo warn と指定すると、アニメーションGIFも出力しません。（その分早いです）
set VERBOSE=2
set IS_DEBUG=no
set /P IS_DEBUG="■詳細ログ[yes/no/warn]: "

IF /I "%IS_DEBUG%" EQU "yes" (
    set VERBOSE=3
)

IF /I "%IS_DEBUG%" EQU "warn" (
    set VERBOSE=1
)

rem ---  python 実行
python bin/3dpose_gan_json.py --lift_model train/gen_epoch_500.npz --model2d openpose/pose_iter_440000.caffemodel --proto2d openpose/openpose_pose_coco.prototxt --person_idx 1 --base-target %TARGET_DIR% --verbose %VERBOSE%


