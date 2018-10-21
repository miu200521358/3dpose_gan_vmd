# 3dpose_gan_vmd

このプログラムは、[3dpose_gan](https://github.com/DwangoMediaVillage/3dpose_gan) \(DwangoMediaVillage様\) を miu(miu200521358) がfork して、改造しました。

動作詳細等は上記URL、または [README-original.md](README-original.md) をご確認ください。

## 機能概要

- [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) で検出された人体の骨格構造から、3Dの人体モデルを生成します。
- 3Dの人体モデルを生成する際に、関節データを出力します
    - 関節データを [VMD-3d-pose-baseline-multi](https://github.com/miu200521358/VMD-3d-pose-baseline-multi) で読み込む事で、vmd(MMDモーションデータ)ファイルを生成できます
- 複数人数のOpenPoseデータを解析（正確には、解析対象人物INDEXを指定）できます。
    - 2018/05/07 時点では正確に解析できません。1人のみの解析を試してください。

## 準備

詳細は、[Qiita](https://qiita.com/miu200521358/items/d826e9d70853728abc51)を参照して下さい。

### 依存関係

python3系 で以下をインストールして下さい

* [h5py](http://www.h5py.org/)
* [chainer](https://chainer.org/)

### 学習データ

`openpose`ディレクトリを作成し、以下の学習データをダウンロードして配置して下さい。

- [openpose_pose_coco.prototxt](https://github.com/opencv/opencv_extra/blob/3.4.1/testdata/dnn/openpose_pose_coco.prototxt)
- [pose_iter_440000.caffemodel](http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel)

`train`ディレクトリを作成し、以下の学習データをダウンロードして配置して下さい。

- [gen_epoch_500.npz](https://github.com/DwangoMediaVillage/3dpose_gan/blob/master/sample/gen_epoch_500.npz?raw=true)

## 実行方法

1. [Openpose簡易起動バッチ](https://github.com/miu200521358/openpose-simple) で データを解析する
1. [miu200521358/3d-pose-baseline-vmd](https://github.com/miu200521358/3d-pose-baseline-vmd) で生成された2D関節データ (smoothed.txt) を用意する
1. [OpenposeTo3DGan.bat](OpenposeTo3DGan.bat) を実行する
1. `解析結果JSONディレクトリパス` が聞かれるので、1.の`json出力ディレクトリパス`のフルパスを指定する
1. `3D解析結果ディレクトリパス` が聞かれるので、2.の結果ディレクトリパスを指定する
1. `出力対象人物INDEX` が聞かれるので、Openposeで読み取った人物のうち、何番目の人物を出力したいか、1始まりで指定する。
	- 未指定の場合、デフォルトで1が設定される(１人目の解析)
1. `詳細なログを出すか` 聞かれるので、出す場合、`yes` を入力する
    - 未指定 もしくは `no` の場合、通常ログ（各パラメータファイルと3D化アニメーションGIF）
    - `warn` の場合、3D化アニメーションGIFも生成しない（その分早い）
    - `yes`の場合、詳細ログを出力し、ログメッセージの他、デバッグ用画像も出力される（その分遅い）
1. 処理開始
1. 処理が終了すると、2.の結果ディレクトリパスに、以下の結果が出力される。
    - pos_gan.txt … 全フレームの関節データ([VMD-3d-pose-baseline-multi](https://github.com/miu200521358/VMD-3d-pose-baseline-multi) に必要) 詳細：[Output](https://github.com/miu200521358/3d-pose-baseline-vmd/blob/master/doc/Output.md)
    - smoothed_gan.txt … 全フレームの2D位置データ([VMD-3d-pose-baseline-multi](https://github.com/miu200521358/VMD-3d-pose-baseline-multi) に必要) 詳細：[Output](https://github.com/miu200521358/3d-pose-baseline-vmd/blob/master/doc/Output.md)
    - movie_smoothing_gan.gif … フレームごとの姿勢を結合したアニメーションGIF
    - frame3d_gan/gan_0000000000xx.png … 各フレームの3D姿勢
    - frame3d_gan/gan_0000000000xx_xxx.png … 各フレームの角度別3D姿勢(詳細ログyes時のみ)

## 注意点

- Openpose のjson任意ファイル名に12桁の数字列は使わないで下さい。
    - `short02_000000000000_keypoints.json` のように、`{任意ファイル名}_{フレーム番号}_keypoints.json` というファイル名のうち、12桁の数字をフレーム番号として抽出するため

## ライセンス
MIT

以下の行為は自由に行って下さい

- モーションの調整・改変
- ニコニコ動画やTwitter等へのモーション使用動画投稿
- モーションの不特定多数への配布
    - **必ず踊り手様や各権利者様に失礼のない形に調整してください**

以下の行為は必ず行って下さい。ご協力よろしくお願いいたします。

- クレジットへの記載を、テキストで検索できる形で記載お願いします。

```
ツール名：モーショントレース自動化キット
権利者名：miu200521358
```

- モーションを配布する場合、以下ライセンスを同梱してください。 (記載場所不問)

```
LICENCE

モーショントレース自動化キット
【Openpose】：CMU　…　https://github.com/CMU-Perceptual-Computing-Lab/openpose
【Openpose起動バッチ】：miu200521358　…　https://github.com/miu200521358/openpose-simple
【Openpose→3D変換】：una-dinosauria, ArashHosseini, miu200521358　…　https://github.com/miu200521358/3d-pose-baseline-vmd
【Openpose→3D変換その2】：Dwango Media Village, miu200521358：MIT　…　https://github.com/miu200521358/3dpose_gan_vmd
【深度推定】：Iro Laina, miu200521358　…　https://github.com/miu200521358/FCRN-DepthPrediction-vmd
【3D→VMD変換】： errno-mmd, miu200521358 　…　https://github.com/miu200521358/VMD-3d-pose-baseline-multi
```

- ニコニコ動画の場合、コンテンツツリーへ [トレース自動化マイリスト](https://www.nicovideo.jp/mylist/61943776) の最新版動画を登録してください。
    - コンテンツツリーに登録していただける場合、テキストでのクレジット有無は問いません。

以下の行為はご遠慮願います

- 自作発言
- 権利者様のご迷惑になるような行為
- 営利目的の利用
- 他者の誹謗中傷目的の利用（二次元・三次元不問）
- 過度な暴力・猥褻・恋愛・猟奇的・政治的・宗教的表現を含む（R-15相当）作品への利用
- その他、公序良俗に反する作品への利用

## 免責事項

- 自己責任でご利用ください
- ツール使用によって生じたいかなる問題に関して、作者は一切の責任を負いかねます
