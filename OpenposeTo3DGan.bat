@echo off
rem --- 
rem ---  OpenPose �� json�f�[�^���� 3D�f�[�^�ɕϊ�
rem ---  https://nico-opendata.jp/ja/casestudy/3dpose_gan/index.html���g�p
rem --- 

rem ---  �J�����g�f�B���N�g�������s��ɕύX
cd /d %~dp0

rem ---  INDEX�ʃf�B���N�g���p�X
echo INDEX�ʃf�B���N�g���̃t���p�X����͂��ĉ������B({���於}_json_{���s����}_idx00)
echo pos.txt�Ȃǂ̂���f�B���N�g���ł��B
echo ���̐ݒ�͔��p�p�����̂ݐݒ�\�ŁA�K�{���ڂł��B
echo ,(�J���})��5���܂Őݒ�\�ł��B
set TARGET_DIR=
set /P TARGET_DIR=��INDEX�ʃf�B���N�g���p�X: 
rem echo TARGET_DIR�F%TARGET_DIR%

IF /I "%TARGET_DIR%" EQU "" (
    ECHO INDEX�ʃf�B���N�g���p�X���ݒ肳��Ă��Ȃ����߁A�����𒆒f���܂��B
    EXIT /B
)

rem ---  �ڍ׃��O�L��

echo --------------
echo �ڍׂȃ��O���o�����Ayes �� no ����͂��ĉ������B
echo �������͂����AENTER�����������ꍇ�A�ʏ탍�O�ƃ��[�V�����̃A�j���[�V����GIF���o�͂��܂��B
echo �ڍ׃��O�̏ꍇ�A�e�t���[�����Ƃ̃f�o�b�O�摜���ǉ��o�͂���܂��B�i���̕����Ԃ�������܂��j
echo warn �Ǝw�肷��ƁA�A�j���[�V����GIF���o�͂��܂���B�i���̕������ł��j
set VERBOSE=2
set IS_DEBUG=no
set /P IS_DEBUG="���ڍ׃��O[yes/no/warn]: "

IF /I "%IS_DEBUG%" EQU "yes" (
    set VERBOSE=3
)

IF /I "%IS_DEBUG%" EQU "warn" (
    set VERBOSE=1
)

rem ---  python ���s
python bin/3dpose_gan_json.py --lift_model train/gen_epoch_500.npz --model2d openpose/pose_iter_440000.caffemodel --proto2d openpose/openpose_pose_coco.prototxt --person_idx 1 --base-target %TARGET_DIR% --verbose %VERBOSE%


