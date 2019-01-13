@echo off
rem --- 
rem ---  OpenPose の jsonデータから 3Dデータに変換
rem ---  https://nico-opendata.jp/ja/casestudy/3dpose_gan/index.htmlを使用
rem --- 

rem ---  カレントディレクトリを実行先に変更
cd /d %~dp0

rem ---  解析結果JSONディレクトリパス
echo Openposeの解析結果のJSONディレクトリのフルパスを入力して下さい。
echo この設定は半角英数字のみ設定可能で、必須項目です。
set OPENPOSE_JSON=
set /P OPENPOSE_JSON=■解析結果JSONディレクトリパス: 
rem echo OPENPOSE_JSON：%OPENPOSE_JSON%

IF /I "%OPENPOSE_JSON%" EQU "" (
    ECHO 解析結果JSONディレクトリパスが設定されていないため、処理を中断します。
    EXIT /B
)

rem ---  3d-pose-baseline-vmd解析結果JSONディレクトリパス
echo 3d-pose-baseline-vmdの解析結果ディレクトリの絶対パスを入力して下さい。(3d_{実行日時}_idx00)
echo この設定は半角英数字のみ設定可能で、必須項目です。
set TARGET_DIR=
set /P TARGET_DIR=■3d-pose-baseline-vmd 解析結果ディレクトリパス: 
rem echo TARGET_DIR：%TARGET_DIR%

IF /I "%TARGET_DIR%" EQU "" (
    ECHO 3D解析結果ディレクトリパスが設定されていないため、処理を中断します。
    EXIT /B
)

rem ---  映像に映っている最大人数

echo --------------
echo 映像の解析結果のうち、何番目の人物を解析するか1始まりで入力して下さい。
echo 何も入力せず、ENTERを押下した場合、1人目の解析になります。
set PERSON_IDX=1
set /P PERSON_IDX="■解析対象人物INDEX: "

rem --echo PERSON_IDX: %PERSON_IDX%


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
python bin/3dpose_gan_json.py --lift_model train/gen_epoch_500.npz --model2d openpose/pose_iter_440000.caffemodel --proto2d openpose/openpose_pose_coco.prototxt --input %OPENPOSE_JSON%  --person_idx %PERSON_IDX% --base-target %TARGET_DIR% --verbose %VERBOSE%


