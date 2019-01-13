@echo off
rem --- 
rem ---  Convert json data of OpenPose to 3D data
rem ---  https://nico-opendata.jp/ja/casestudy/3dpose_gan/index.html
rem --- 

rem ---  Change the current directory to the execution destination
cd /d %~dp0

rem ---  Analysis result JSON directory path
echo Please enter the full path of JSON directory of analysis result of Openpose.
echo This setting is available only for half size alphanumeric characters, it is a required item.
set OPENPOSE_JSON=
set /P OPENPOSE_JSON=** Analysis result JSON directory path: 
rem echo OPENPOSE_JSONÅF%OPENPOSE_JSON%

IF /I "%OPENPOSE_JSON%" EQU "" (
    ECHO Analysis result Since JSON directory path is not set, processing is interrupted.
    EXIT /B
)

rem ---  3d-pose-baseline-vmd Analysis result JSON directory path
echo Please enter the absolute path of the analysis result directory of 3d-pose-baseline-vmd.(3d_{yyyymmdd_hhmmss}_idx00)
echo This setting is available only for half size alphanumeric characters, it is a required item.
set TARGET_DIR=
set /P TARGET_DIR=** 3d-pose-baseline-vmd Analysis result directory path: 
rem echo TARGET_DIRÅF%TARGET_DIR%

IF /I "%TARGET_DIR%" EQU "" (
    ECHO 3D analysis result directory path is not set, processing will be interrupted.
    EXIT /B
)

rem ---  Maximum number of people in the movie

echo --------------
echo In the analysis result of the video, what number of people will be analyzed 1 Please input at the beginning.
echo If you do not enter anything and press ENTER, you will be analyzing the first person.
set PERSON_IDX=1
set /P PERSON_IDX="** Analysis target person INDEX: "

rem --echo PERSON_IDX: %PERSON_IDX%


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

rem ---  run python 
python bin/3dpose_gan_json.py --lift_model train/gen_epoch_500.npz --model2d openpose/pose_iter_440000.caffemodel --proto2d openpose/openpose_pose_coco.prototxt --input %OPENPOSE_JSON%  --person_idx %PERSON_IDX% --base-target %TARGET_DIR% --verbose %VERBOSE%


