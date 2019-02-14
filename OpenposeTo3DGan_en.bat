@echo off
rem --- 
rem ---  Convert json data of OpenPose to 3D data
rem ---  https://nico-opendata.jp/ja/casestudy/3dpose_gan/index.html
rem --- 

rem ---  Change the current directory to the execution destination
cd /d %~dp0

rem ---  Individual directory path
echo Please enter the full path of individual index directory.({Movie name}_json_{Execution date and time}_idx00)
echo There is a directory such as pos.txt.
echo This setting is available only in Half - width alphanumeric characters and is mandatory.
set TARGET_DIR=
set /P TARGET_DIR=** Individual directory path: 
rem echo TARGET_DIRÅF%TARGET_DIR%

IF /I "%TARGET_DIR%" EQU "" (
    ECHO Since the individual directory path of the index is not set, processing is interrupted.
    EXIT /B
)

rem ---  Presence of detailed log

echo --------------
echo Please output detailed logs or enter yes or no.
echo If nothing is entered and ENTER is pressed, normal animation GIF of log and motion is output.
echo For detailed logs, debug images for each frame are additionally output. (It will take time for that)
echo If warn is specified, animation GIF is not output. (That is earlier)
set VERBOSE=2
set IS_DEBUG=no
set /P IS_DEBUG="** Detailed log[yes/no/warn]: "

IF /I "%IS_DEBUG%" EQU "yes" (
    set VERBOSE=3
)

IF /I "%IS_DEBUG%" EQU "warn" (
    set VERBOSE=1
)

rem ---  python é¿çs
python bin/3dpose_gan_json.py --lift_model train/gen_epoch_500.npz --model2d openpose/pose_iter_440000.caffemodel --proto2d openpose/openpose_pose_coco.prototxt --person_idx 1 --base-target %TARGET_DIR% --verbose %VERBOSE%


