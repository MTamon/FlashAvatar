# SMIRK — オプションの FLAME 特徴抽出器

FlashAvatar の既定トラッカーは [metrical-tracker](https://github.com/Zielon/metrical-tracker)
（MTamon の `claude0420` フォーク経由）で、フレームごとに最適化ベースで
FLAME フィットを行います。Obama の例のような「スタジオ品質」のシーケンス
では正確ですが、**頭部の大きな回転やモーションブラーを含む動画では
失敗します** — フレーム単位の最適化は初期値に敏感で、頭が素早く動くと
誤差が蓄積するためです。

[SMIRK](https://github.com/MTamon/smirk) はフィードフォワードの FLAME
エンコーダです。1 フレームあたり 1 回のネットワーク順伝播で FLAME
パラメータを回帰するため、速い動きに頑健です。MTamon フォークの
`release/cuda128` ブランチは FlashAvatar（Python 3.11 / PyTorch
2.9.1 / CUDA 12.8）とピンが揃っているので、FlashAvatar と同じ環境で
動作します — 追加の環境は不要です。

**SMIRK はオプトインです。** `install_128.sh` ではインストールされません。
必要になったときだけ `scripts/setup_smirk.sh` で個別にインストールしてください。

## クイックスタート

```bash
source .venv/bin/activate
bash scripts/setup_smirk.sh                      # 初回のみ

# 次のコマンドの代わりに:
#   bash scripts/run_tracker.sh <idname>
# こちらを実行:
bash scripts/run_tracker.sh <idname> --smirk     # ディスパッチャ
# または直接:
bash scripts/run_smirk_tracker.sh <idname>
# あるいは CLI 経由:
python scripts/preprocess.py smirk --idname <idname>

# finalize はどのトラッカーでも同じ:
python scripts/preprocess.py finalize --idname <idname>

# あとは通常どおり学習:
python train.py --idname <idname>
```

既定は `--eye-mode zero` で、全フレームで恒等の目ポーズを書き出すため、
学習されたアバターの目は正面を向いたまま固定されます。
**視線トラッキングを有効化**するには `--eye-mode blendshapes` を指定してください。
各 SMIRK 検出時に同時取得される MediaPipe ARKit ブレンドシェイプ
(`eyeLookIn/Out/Up/Down*`) からフレームごとの目の回転が導出され、
学習済みアバターが被写体の視線を追うようになります。
厳密なマッピングは注意事項 #1 を参照してください。

## 機能互換性マトリクス

FlashAvatar の `.frame` ファイル形式（仕様全体は
[preprocessing.md](preprocessing.md) 参照）は、以下のキーを想定しています。
この表は、それぞれが SMIRK のエンコーダ出力からどのように導出されるかを示します。

| `.frame` キー       | 形状      | SMIRK 由来                       | 変換 |
|---------------------|-----------|----------------------------------|------|
| `flame.shape`       | (1, 300)  | フレームごとの `shape_params`    | 先頭から検出できた `--shape-frames` フレームの**中央値**を全 `.frame` で共有（metrical-tracker の「単一アイデンティティ」方針と一致） |
| `flame.exp`         | (1, 100)  | `expression_params` (1, 50)      | 末尾 50 次元を**ゼロパディング**。FlashAvatar の FLAME 基底は 100 次元だが、追加の 50 は SMIRK では励起されない。 |
| `flame.jaw`         | (1, 6)    | `jaw_params`（軸角・3 次元）     | `matrix_to_rotation_6d(axis_angle_to_matrix(aa))` |
| `flame.eyes`        | (1, 12)   | 既定では恒等、**または** `--eye-mode blendshapes` 指定時は **MediaPipe Face Landmarker ブレンドシェイプ** | 既定 `--eye-mode zero` は恒等 6D × 2（固定目）を書き込む。`--eye-mode blendshapes` で視線追従を有効化：左右それぞれ `axis_angle = [(down - up) * 0.6, ±(in - out) * 0.6, 0]` → 3×3 回転 → pytorch3d rot6d。被写体が内側を見たとき両目が寄るよう、右目はヨー符号を反転する。検出なしのフレームでは恒等にフォールバック。 |
| `flame.eyelids`     | (1, 2)    | `eyelid_params`                  | `[0, 1]` にクランプ |
| `opencv.R`          | (1, 3, 3) | `pose_params`（軸角）            | `Rodrigues(aa)` のあと `diag(1,-1,-1) @ R` で OpenGL（y 上向き）→ OpenCV（y 下向き、+z 前方）に変換。 |
| `opencv.t`          | (1, 3)    | `cam=[s, tx, ty]` + クロップ `tform` | 後述「カメラ合成」参照 |
| `opencv.K`          | (1, 3, 3) | 合成                             | `f_px = --focal-px`（既定 5000）、主点はフルフレーム中心。 |
| `img_size`          | (W, H)    | フルの生フレーム解像度           | 後段の `preprocess finalize` が K と `img_size` を最終の 512×512 頭部クロップに書き換える — metrical-tracker と同じ契約。 |

### カメラ合成（細かいが重要）

SMIRK は 224×224 の内部クロップに対して**弱透視／正射影**カメラを使います。
一方 FlashAvatar は完全な**透視投影**カメラ（OpenCV `K/R/t`）を使います。
変換は 2 つの条件を満たさなければなりません。

1. 投影された FLAME メッシュが、SMIRK の 224 クロップではなくフルの生フレームの
   正しいピクセルに落ちること — そうすれば `preprocess finalize` が切り出す
   最終 512 頭部クロップで頭位置が正しくなる。
2. 投影メッシュが画面上で正しい**サイズ**になること。

正射影は焦点距離の大きい透視投影で近似します。数式は
`preprocess/smirk_convert.py::_build_t` にありますが、要約すると次のとおりです。

- SMIRK の内部クロップはフルフレームピクセル → 224 クロップピクセルへの
  **相似変換** `tform` です。等方であり、スケール係数
  `s_ff = |tform[0,0]|` がフルフレーム px をクロップ px に変換します。
- SMIRK の `cam = [s, tx, ty]` は FLAME 原点をクロップピクセル
  `(112 + 112*s*tx, 112 − 112*s*ty)` に配置します（SMIRK のレンダラ内部で
  y が反転される）。
- `tform` を反転すれば、FLAME 原点のフルフレームピクセルが得られます。
- 見かけサイズの一致：1 FLAME 単位は `s * 112` クロップピクセル
  ⇒ `s * 112 / s_ff` フルフレームピクセルに投影されます。焦点 `f_px`・
  奥行き `Z` の透視投影カメラでは 1 単位が `f_px / Z` フルフレームピクセルに
  投影されるため、等式より `Z = f_px * s_ff / (s * 112)`。
- 最後に `t = [(u_full − cx) * Z / f_px, (v_full − cy) * Z / f_px, Z]`、
  ここで `(cx, cy) = (W/2, H/2)`。

`--focal-px` フラグが透視投影近似を制御します。`f_px` が大きいほど
奥行きが遠くなり、正射影に近づきます。既定値 `5000` は ~1k〜4k 解像度の
頭部のみのシーケンスで安全ですが、フレーム端で「近距離カメラ」的な歪みが
目立つ場合は `10000` に上げてください。

### なぜ FlashAvatar の 512 クロップから切り離せるのか

`preprocess finalize --crop` はフルの生フレームから**まったく別の**正方
クロップ（シーケンス全体の `*_neckhead.png` マスクの和集合に
`--crop-pad` でパディング）を計算します。そのあと各 `.frame` の `opencv.K`
と `img_size` を最終 512×512 ビューに書き換えます。SMIRK の `.frame`
ファイルを**フル生解像度**で書き出しているのは、まさにこの下流処理を
変更なしで動かすためです — SMIRK 内部の 224 クロップは FlashAvatar から
見えません。

## 注意事項

1. **SMIRK は眼球の回転を回帰しません。** 既定の `--eye-mode zero` は全
   フレームで恒等の目ポーズを書き込むため、学習後のアバターは固定目で
   レンダリングされます。視線追従アバターにするには
   `--eye-mode blendshapes` を指定してください。同じフレーム単位の検出
   パスで取得される MediaPipe Face Landmarker の ARKit ブレンドシェイプ
   から目の回転が合成されます。マッピングは ARKit → FLAME 規約に従います。

   - `left_pitch = (eyeLookDownLeft  - eyeLookUpLeft)  * 0.6 rad`
   - `left_yaw   = (eyeLookInLeft    - eyeLookOutLeft) * 0.6 rad`
   - `right_pitch = (eyeLookDownRight - eyeLookUpRight) * 0.6 rad`
   - `right_yaw  = (eyeLookOutRight  - eyeLookInRight) * 0.6 rad`（反転）
   - 左右それぞれ `axis_angle = [pitch, yaw, 0]` → `axis_angle_to_matrix`
     → `matrix_to_rotation_6d`（pytorch3d の行規約、
     `flame/lbs.py::rotation_6d_to_matrix` と一致）。

   `--eye-mode blendshapes` には SMIRK の `quick_install.sh` が
   `external/smirk/assets/` にダウンロードする `face_landmarker.task`
   ファイルが必要です。このファイルが無い場合、ランタイムは
   旧来の `mp.solutions.face_mesh.FaceMesh` API（ブレンドシェイプ非対応）
   にフォールバックし、黙って恒等の目ポーズに劣化します。
   `bash external/smirk/quick_install.sh` を再実行して直してください。
   なお `--eye-mode` に関わらずまぶた (eyelids) は SMIRK 自身の
   `eyelid_params` 出力から取得され、MediaPipe の瞬きブレンドシェイプは
   使いません。
2. **SMIRK の弱透視カメラは近似です。** 頭部が画像に対して小さく焦点が
   長い場合、正射影→透視投影の変換誤差は無視できる程度ですが、短焦点・
   近接撮影の動画では metrical-tracker と比べて微妙なスケールバイアスが
   入ることがあります。`--verify-dir` でオーバーレイを出力して確認して
   ください。
3. **フレームごとの形状ジッタ。** SMIRK は 300 次元形状をフレームごとに
   独立して回帰します。metrical-tracker に合わせてシーケンス単位の中央値に
   畳み込んでいます。変装や表情変化でアイデンティティが大きく変わる
   シーケンスでは、`--shape-frames` を増やすか、先頭フレームを事前に
   フィルタリングしてください。
4. **表情次元の不一致。** FlashAvatar は上流トラッカーの 100 次元表情で
   学習されています。SMIRK は先頭 50 成分しか回帰せず、残り 50 は厳密に
   ゼロです。実際には先頭 50 成分が支配的ですが、metrical-tracker 出力と
   比べて収束が少し遅くなることがあります — 数千イテレーション追加学習を
   検討してください。
5. **最初のフレームで MediaPipe の顔検出が成功する必要があります。**
   SMIRK にはバウンディングボックス検出器が無いので、こちらで MediaPipe
   検出を行い、検出失敗フレームには直前の有効ランドマークを伝播します。
   最初のフレームに顔が無いとランナーはエラー終了するため、動画を
   トリミングしてください。

## SMIRK + FlashAvatar エンドツーエンドデモ

生の動画から学習済みアバターまで、FLAME トラッカーとして SMIRK を使い
端から端まで通す手順です。`<idname>` は安定した識別子（例：被写体名）に、
`path/to/clip.mp4` は入力動画に置き換えてください。

前提（マシンごとに 1 回だけ）：

```bash
bash install_128.sh                              # FlashAvatar 環境
bash scripts/setup_metrical_tracker.sh           # 任意 — 既定パスも使う
                                                 # 場合だけ
bash scripts/setup_smirk.sh                      # SMIRK + FLAME + task ファイル
source .venv/bin/activate                        # ここ以降はアクティベート
```

ステップ 1 — 入力動画をフレームごとの画像、parsing マスク、matting アルファに
デコードします。この段階は後でどのトラッカーを選んでも同じで、生の `.mp4`
を触る唯一のタイミングです。ステップ 3 以降、SMIRK はここで書き出された
**画像ファイル**を走査するだけで、動画そのものは触りません。

```bash
python scripts/preprocess.py prepare \
    --idname <idname> --video path/to/clip.mp4
```

内部では `prepare` が 3 つのサブステージを実行し、動画の**ネイティブ解像度**で
3 つの並列ディレクトリを書き出します（後段の `finalize` で 512×512 に再クロップ）。

| サブステージ | ツール | 出力 | 初回要件 |
|---|---|---|---|
| `extract` | `ffmpeg`（`PATH` 上に必要） | `dataset/<idname>/raw/imgs/00001.jpg, 00002.jpg, …` | ffmpeg のインストール |
| `parsing` | BiSeNet 顔パーシング | `dataset/<idname>/raw/parsing/*.png`（画素単位のクラスラベル。`*_neckhead.png` = 頭部+首マスク） | インターネット接続（`gdown` による重み自動ダウンロード）または `--bisenet-weights PATH` |
| `matting` | RobustVideoMatting | `dataset/<idname>/raw/alpha/*.jpg`（ソフトな前景アルファ） | インターネット接続（初回の `torch.hub.load`） |

前段のサブステージがキャッシュされたら、`--skip-extract`・`--skip-parsing`・
`--skip-matting` で選択的に再実行できます。SMIRK のランタイム（ステップ 3）
は `raw/imgs/*.jpg` を辞書順に読み取るため、ここで書き込まれる 5 桁
ゼロ埋めのファイル名がそのままフレーム ID となり、下流の
`preprocess finalize` で再キーイングされます。

### ffmpeg でクリップを事前にトリム／正規化する（任意だが有用）

`prepare` はネイティブ fps・解像度で全フレームを抽出するため、無駄な
フレームは 3 回（extract → parsing → matting → SMIRK エンコード）
コストを払うことになります。`prepare` を呼ぶ前に ffmpeg でクリップを
正規化する価値があります。SMIRK 特有の制約として、
**MediaPipe が最初のフレームで顔を検出できる必要があります**（SMIRK の
クロップを初期化するため）。したがって、先頭の空白フレームや後ろを向いた
フレームは除去する必要があります。

```bash
# 必要な範囲にトリム（再エンコードなしのロスレス）:
ffmpeg -ss 00:00:10 -to 00:00:25 -i raw.mp4 -c copy trimmed.mp4

# 60 fps → 30 fps にダウンサンプル。トラッキング品質を実質損なわずに
# パイプライン時間をおおよそ半減できる:
ffmpeg -i raw.mp4 -vf fps=30 -c:v libx264 -pix_fmt yuv420p -an out.mp4

# 4K → 1080p にダウンスケール。SMIRK の 224×224 内部クロップに対して
# ~1080p 超の解像度は無駄:
ffmpeg -i raw.mp4 -vf "scale=-2:1080" -c:v libx264 -pix_fmt yuv420p -an out.mp4

# prepare 内部の ffmpeg が苦手とする変なコンテナ／ピクセル形式を修正:
ffmpeg -i weird.mov -c:v libx264 -pix_fmt yuv420p -an normalised.mp4
```

`prepare` のフラグ一覧（`--bisenet-weights`・`--rvm-variant`・各ステージの
スキップ、正確なファイル命名契約）は
[`preprocessing.md`](preprocessing.md) を参照してください。

ステップ 2 *（任意。手持ち撮影のブレが多いクリップで推奨）* — 最もブレの
大きいフレームにタグを付け、データを削除せずに `train.py` / `test.py`
でスキップできるようにします。

```bash
python scripts/preprocess.py filter-blur --idname <idname> --percentile 15
# dataset/<idname>/raw/keep_list.txt と blur_preview.jpg を書き出す
```

ステップ 3 — SMIRK を実行して `.frame` ファイルを生成します。これが
既定パイプラインとの唯一の相違点です。フレームごとの metrical-tracker
最適化ではなく、フレームごとに 1 回のフィードフォワード推論を行います。
出力先は `metrical-tracker/output/<idname>/checkpoint_raw/*.frame` です。

```bash
bash scripts/run_tracker.sh <idname> --smirk \
    --verify-dir dataset/<idname>/smirk_verify
# 等価な書き方:
#   bash scripts/run_smirk_tracker.sh <idname> --verify-dir ...
#   python scripts/preprocess.py smirk --idname <idname> --verify-dir ...
```

**視線追従を有効化するには `--eye-mode blendshapes` を追加します。**
これが学習済みアバターを「固定目」から「被写体の視線を追う」へ切り替える
ステップです。

```bash
bash scripts/run_tracker.sh <idname> --smirk --eye-mode blendshapes \
    --verify-dir dataset/<idname>/smirk_verify
```

`--eye-mode blendshapes` を指定すると、SMIRK ランタイムは各フレームで
MediaPipe Face Landmarker の ARKit ブレンドシェイプ係数
(`eyeLookInLeft`・`eyeLookOutLeft`・`eyeLookUpLeft`・`eyeLookDownLeft`
および右目対応物) も追加取得し、左右の軸角回転に変換してから
`(1, 12)` の pytorch3d rot6d にパックし、各 `.frame` の `flame.eyes`
に書き込みます。FlashAvatar のデフォーム MLP (`src/deform_model.py`) と
FLAME LBS (`flame/flame_mica.py`) はいずれもこのテンソルを消費するため、
最終レンダリング結果は入力クリップと同期して目が動きます。

このフラグを付けない（既定の `--eye-mode zero`）と `flame.eyes` は
恒等 rot6d × 2 となり、アバターの目は正面を向いたまま固定されます。
視線モーションが不要な場合や、ブレンドシェイプ信号を切って比較したい
場合に使います。

`--eye-mode blendshapes` は `external/smirk/assets/face_landmarker.task`
に依存します（SMIRK の `quick_install.sh` がダウンロードし、
`scripts/setup_smirk.sh` 内で自動実行されます）。SMIRK ランタイムの
起動ログ行がどの MediaPipe バックエンドがアクティブかを示します。
`face_landmarker.task not found` が出ていたら、トラッカーステップの前に
`bash external/smirk/quick_install.sh` を再実行してください。

`dataset/<idname>/smirk_verify/stats.csv` と `overlay_*.jpg` を
確認してください。1080p クリップでランドマーク再投影誤差の中央値が
10 px 程度であれば、SMIRK のカメラが FlashAvatar とよく揃っています。
中央値が 20 px を超える場合は `--focal-px` を増やして（例：
`--focal-px 8000`）再実行してください。

`--eye-mode` を変えてこのステップを再実行するのは安価です。
`.frame` ファイルはその場で書き換えられ、以降のステップ 4〜6 は
同一です。強制再生成するには `--overwrite` を指定します。

```bash
bash scripts/run_tracker.sh <idname> --smirk \
    --eye-mode blendshapes --overwrite
```

ステップ 4 — finalize：頭部中心 512×512 クロップ + K / `img_size`
書き換え。どのトラッカーでも同じです。

```bash
python scripts/preprocess.py finalize --idname <idname>
# dataset/<idname>/{imgs,parsing,alpha}/ と
# metrical-tracker/output/<idname>/checkpoint/ (再キーイング済み .frame) を書き出す
```

ステップ 5 — 学習。FlashAvatar の `train.py` は metrical-tracker 由来の
場合とまったく同じように `.frame` を消費します。

```bash
python train.py --idname <idname>
# チェックポイントは logs/<idname>/ 配下に出力。
```

ステップ 6 — 学習済みチェックポイントでテスト分割をレンダリング。

```bash
python test.py --idname <idname>
# logs/<idname>/test.avi に書き出す（既定では全フレーム）。
```

以上がフルループです。視線追従のサニティチェック：
`--eye-mode blendshapes` で学習した結果、レンダリングされたアバターの目が
被写体の視線に追従していれば、ブレンドシェイプから導出した `eyes_pose` が
デフォーム MLP と FLAME LBS を通って意図どおり伝わっています。入力クリップ
では視線が動いているのに目が正面固定で出力される場合は、既定の
`--eye-mode zero` で学習したか（ステップ 3 を
`--eye-mode blendshapes --overwrite` で再実行し、ステップ 4〜6 を
やり直す）、`external/smirk/assets/` に `face_landmarker.task` が無い
かのどちらかです（`[smirk/runtime]` の起動ログを確認し、
`bash external/smirk/quick_install.sh` を再実行してください）。

### 最小ワンライナー（バッチジョブ用）

視線追従なし（固定目アバター）：

```bash
source .venv/bin/activate && \
  python scripts/preprocess.py prepare  --idname $ID --video $VIDEO && \
  bash   scripts/run_tracker.sh         $ID --smirk && \
  python scripts/preprocess.py finalize --idname $ID && \
  python train.py --idname $ID && \
  python test.py  --idname $ID
```

視線追従あり（トラッカーステップに `--eye-mode blendshapes` を追加）：

```bash
source .venv/bin/activate && \
  python scripts/preprocess.py prepare  --idname $ID --video $VIDEO && \
  bash   scripts/run_tracker.sh         $ID --smirk --eye-mode blendshapes && \
  python scripts/preprocess.py finalize --idname $ID && \
  python train.py --idname $ID && \
  python test.py  --idname $ID
```

## SMIRK 自体のデモを動かす（任意のスモークテスト）

`scripts/setup_smirk.sh` は SMIRK を `external/smirk/` にクローンします。
完了後は、FlashAvatar のパイプラインとは独立に、そのチェックアウトから
直接 SMIRK 上流のデモを実行してインストールの健全性を確認できます。
これらのデモはオーバーレイや動画をレンダリングするので、FlashAvatar の
フル実行に入る前にトラッキング品質を目視チェックできます。

すべてのデモは*同じ*アクティブな FlashAvatar venv で動作します — 別環境
への切り替えは不要です（後述「環境互換性」を参照）。

```bash
source .venv/bin/activate
cd external/smirk

# （初回のみ）デモ用のサンプル動画バンドルを取得
bash prepare_demos.sh

# 単一画像 -> FLAME オーバーレイ + レンダリング画像
bash demos/run_demo.sh --input_path samples/test_image2.png --crop
#   external/smirk/output/ ... に書き出し（フラグは SMIRK の README 参照）

# 動画 -> オーバーレイ動画
bash demos/run_demo_video.sh --input_path samples/dafoe.mp4 --crop
#   external/smirk/output/dafoe/dafoe.mp4 に書き出し

# 動画 -> 生の FLAME パラメータ（.pt 辞書、レンダリングなし）
bash demos/run_demo_save_flame.sh --input_path samples/dafoe.mp4 --crop
#   external/smirk/output/dafoe/dafoe.pt に書き出し
```

よく使うフラグ（`.sh` ラッパが内部の `demos/demo*.py` に転送）：

| フラグ | 用途 |
|---|---|
| `--crop` | MediaPipe で顔を自動クロップ（FlashAvatar の SMIRK パスと同じ挙動）。入力が既に顔クロップ済みの場合だけ省略。 |
| `--mp_delegate {cpu,gpu}` | MediaPipe 顔検出の実行場所。`gpu` は CUDA マシンで高速だが MediaPipe の GPU デリゲートが必要（既定のフォールバックは CPU）。 |
| `--with_eye_pose` | （save_flame 専用）MediaPipe ブレンドシェイプから目の rot6d とまぶたも導出し、出力 .pt に含める。FlashAvatar とは無関係。 |
| `--batch_size N` | エンコーダ推論のバッチサイズ。 |
| `--benchmark` | 段階ごとのタイミングを表示。 |

**注意**：これらのデモは SMIRK 自身のツール類で、その出力形式
（フレームごとの `shape/exp/pose/cam/...` を持つ `.pt` 辞書）は
FlashAvatar の `.frame` 形式とは**異なります**。SMIRK を FlashAvatar に
流し込むには、`scripts/run_smirk_tracker.sh`（または `preprocess smirk`）
を使ってください。これが SMIRK エンコーダをプログラム的に呼び出し、
上述の `.frame` 変換を行います。SMIRK のデモ実行はあくまでサニティ
チェックであり、統合スクリプトの代わりにはなりません。

トラブルシューティング：

- デモ実行時に `ModuleNotFoundError: src.smirk_encoder` → 実行前に
  `external/smirk/` ディレクトリを出ています。`cd` で戻ってください。
- `FileNotFoundError: .../SMIRK_em1.pt` → `quick_install.sh` が未完了。
  `bash external/smirk/quick_install.sh` を再実行してください。
- 特定の顔でトラッキングが悪い → `demo_video.py` で `--scale 1.6` にして
  クロップを緩めるか、独自のバウンディングボックスを与えてください。

フラグの全リファレンスは `external/smirk/README.md` を参照してください。

## FlashAvatar との環境互換性

SMIRK の cuda128 ブランチは、FlashAvatar の install_128.sh と意図的に
ピンを揃えています。

| | FlashAvatar `install_128.sh` | SMIRK `external/smirk/install_128.sh` |
|---|---|---|
| Python | 3.11 | 3.11 |
| CUDA   | 12.8 | 12.8 |
| PyTorch | 2.9.1 | 2.9.1 |
| numpy  | 2.2.6 | 2.2.6 |
| chumpy | `git+mattloper/chumpy@main`（numpy 2 対応） | 同じ |

`setup_smirk.sh` が追加で入れる SMIRK 専用の依存：
`timm`・`albumentations`・`mediapipe`・`scikit-image`・
`pytorch_lightning`（任意。推論コードはこれが無くても動作）。
いずれも FlashAvatar のクリティカルパスにはありません。これらの
インストールは `install_128.sh` がビルドした torch / pytorch3d /
diff_gaussian_rasterization / simple_knn を変更しません。

この統合では SMIRK と FlashAvatar を**単一の共有 venv** で動かします — 
もう一つ別の環境をアクティベートする必要はありません。SMIRK のコード
自体はインポート時に `sys.path.insert(0, external)` 経由で読み込まれ
（`preprocess/smirk_tracker._ensure_on_pythonpath` 参照）、
`smirk.src.smirk_encoder` としてインポートされるため、クローンしたリポジトリを
パッケージとして pip インストールする必要はありません。`smirk.*`
ドット付きパス経由にしているのは、SMIRK 内部の `src/` が
FlashAvatar 自身のトップレベル `src/` 名前空間をシャドーしないようにする
ためです。

`setup_smirk.sh` はインストール後のサニティチェックを実行し、主要な
インポート（`torch`・`pytorch3d`・`diff_gaussian_rasterization`・
`simple_knn`、および SMIRK の `smirk.src.smirk_encoder`）が無傷である
ことを検証します。ここで失敗する場合、SMIRK 側のインストール連鎖で
共有パッケージがダウングレードされています — 差分を報告してもらえれば
ピンを修正します。

既知の非衝突（念のためリスト）：

- `mediapipe` は metrical-tracker の `setup_metrical_tracker.sh` と
  SMIRK のインストーラの両方からインストールされます。どちらも同じ
  系列（0.10.x）をピンしており、後から走った方が勝ちますが、どちらでも
  動作します。
- `chumpy` は FlashAvatar と SMIRK の両方が GitHub main から
  インストールします。どちらを再実行してもべき等です（pip が同一
  コミットを検出）。
- `pytorch_lightning` は SMIRK 側の依存のみで、別の torch を引き込み
  ません。FlashAvatar は一切インポートしません。

## 出力の検証

`preprocess smirk`（またはシェルラッパ経由の `--verify-dir ...`）に
`--verify-dir PATH` を渡すと次を書き出します。

- `verify/stats.csv` — フレームごとの検出フラグ、ランドマーク再投影
  誤差 (px)、クロップ bbox サイズ。
- `verify/overlay_XXXXX.jpg` — 合成した `K/R/t` で FLAME メッシュ点を
  投影した数フレームのサンプル。確認ポイント：
  - 投影点が顔の中心にあること（大きくずれている場合は R の y 反転か
    t の符号が誤り）。
  - 投影メッシュのサイズが顔のサイズに合っていること（`Z` が誤ると
    メッシュが拡大/縮小されて現れる）。

1080p 入力で再投影誤差の中央値が ~10 px 以下であれば「正しそう」、
~20 px を超えると警告が出て、通常は正射影→透視投影の深度推定に
もっと大きな `--focal-px` が必要であることを意味します。

## 既存のインストール／パイプラインとの関係

- `install_128.sh` → 変更なし。FlashAvatar 自身の環境には手を入れません。
- `scripts/setup_metrical_tracker.sh` → 変更なし。依然として既定。
- `scripts/setup_smirk.sh` → 新規、オプション。アクティブ環境の中で
  SMIRK 自身の `install_128.sh` + `quick_install.sh`
  （FlashAvatar のピンセットを共有）を実行します。
- `scripts/run_tracker.sh` → `--smirk` を受け付け、`run_smirk_tracker.sh`
  にディスパッチするようになりました。フラグなしの既定動作は従来どおり
  metrical-tracker です。
- `preprocess finalize` と学習コード → 変更なし。同じ `.frame` 形式を
  消費します。

SMIRK を使わない既存の FlashAvatar ツリーへの外部影響はゼロです：
新規依存ゼロ、挙動変化ゼロ。
