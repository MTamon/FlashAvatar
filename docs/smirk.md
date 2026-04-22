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

> **新規データには `--bbox-mode offline` を推奨します。** 口の開閉・瞬きの
> たびに再投影メッシュが揺れる問題（詳細は「BBox 安定化」節）を抑えます。
> 既存学習済みモデルとのビット互換が必要な場合のみ既定（`legacy`）を
> 使ってください。
>
> ```bash
> bash scripts/run_tracker.sh <idname> --smirk \
>     --bbox-mode offline --bbox-fps 30 \
>     --verify-dir dataset/<idname>/smirk_verify
> ```

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

bbox 安定化あり（新規データ用の推奨設定。`--bbox-fps` は原動画の FPS）：

```bash
source .venv/bin/activate && \
  python scripts/preprocess.py prepare  --idname $ID --video $VIDEO && \
  bash   scripts/run_tracker.sh         $ID --smirk \
           --bbox-mode offline --bbox-fps 30 \
           --eye-mode blendshapes && \
  python scripts/preprocess.py finalize --idname $ID && \
  python train.py --idname $ID && \
  python test.py  --idname $ID
```

## 実行方法まとめ（前処理／デモ／学習／テスト）

ここまでの内容を、「どのコマンドを何の目的で打つか」で一覧化します。
`<idname>` は安定した識別子、`<video>` は入力動画、`<fps>` は原動画の
FPS（例：30）。`--bbox-mode offline` を推奨しますが、既存 `.frame` との
互換性が必要な場合は `legacy`（既定）のままでも構いません。

### 1. 前処理

前処理は「フレーム展開＋マスク生成（`prepare`）」→「SMIRK トラッキング
（`run_tracker.sh --smirk`）」→「512×512 クロップ／K 調整（`finalize`）」の
3 段です。SMIRK ステップが `.frame` を生成する中核で、ここに `--bbox-mode`
が効きます。

```bash
source .venv/bin/activate

# (a) 動画 → フレーム/parsing/alpha（どのトラッカーでも同じ）
python scripts/preprocess.py prepare --idname <idname> --video <video>

# (b) SMIRK で `.frame` 生成（bbox 安定化 + 視線追従つき推奨）
bash scripts/run_tracker.sh <idname> --smirk \
    --bbox-mode offline --bbox-fps <fps> \
    --eye-mode blendshapes \
    --verify-dir dataset/<idname>/smirk_verify

# (c) 最終 512×512 クロップ + K / img_size 書き換え（どのトラッカーでも同じ）
python scripts/preprocess.py finalize --idname <idname>
```

**変形例：**

- 既存 `.frame` とのビット互換が必要な場合は `--bbox-mode` を省略（=`legacy`）。
- リアルタイム相当の平滑（causal）を使う場合は
  `--bbox-mode online --bbox-fps <fps>`。
- 速度・加速度特徴の教師データ用途には FLAME 側の LPF も併用：
  `--lpf-cutoff 2.0 --lpf-fps <fps> --lpf-channels cam,pose,exp,jaw,eyelids`。
- 手ブレの大きい素材では `finalize` の前に
  `python scripts/preprocess.py filter-blur --idname <idname> --percentile 15`。
- `--eye-mode` は学習済みアバターを固定目にするなら `zero`（既定）、
  視線追従にするなら `blendshapes`。

### 2. デモ（トラッキング品質の確認）

SMIRK ステップに `--demo-video` を追加すると、元フレームに **bbox・
MediaPipe ランドマーク・FLAME メッシュ再投影**を重ね書きした mp4 と、
フレームごとの jitter 指標（`*_stats.csv`）が出ます。
`.frame` 出力には影響しません（`--demo-*` 系は描画側のみ）。

**レンダリング convention は LBS が既定**になりました（`pose_params` を
FLAME の root joint を中心に LBS で適用）。旧挙動（`pose` を外部 R に
畳み込む「原点中心回転」）は `--demo-ext-pose` で opt-in できます。
`--demo-lbs-pose` は deprecated ですが、**既定が LBS なので no-op として
受け付けられます**（既存スクリプトはそのまま動きます）。

本タスクで共有されたコマンド（LBS 既定化 + bbox 安定化を有効化した最終形）：

```bash
# LBS convention（既定）+ bbox 安定化
python scripts/preprocess.py smirk --idname Mikawa3 \
    --bbox-mode offline --bbox-fps 30 \
    --demo-video dataset/Mikawa3/smirk_demo/demo_lbs.mp4 --demo-fps 30
# `--demo-lbs-pose` はもう不要です（書いても no-op として受理されます）。
# 既存 `.frame` を再生成したい場合は `--overwrite` を追加。
```

比較用の追加パターン（必要なら並べて見比べる）：

```bash
# baseline: bbox 安定化なし・LBS 既定
python scripts/preprocess.py smirk --idname <idname> \
    --demo-video dataset/<idname>/smirk_demo/demo_baseline.mp4 --demo-fps <fps>

# bbox 安定化あり・LBS 既定（推奨）
python scripts/preprocess.py smirk --idname <idname> \
    --bbox-mode offline --bbox-fps <fps> \
    --demo-video dataset/<idname>/smirk_demo/demo_lbs.mp4 --demo-fps <fps> \
    --overwrite

# 旧 ext-R convention の再現（診断目的。通常は使わない）
python scripts/preprocess.py smirk --idname <idname> \
    --bbox-mode offline --bbox-fps <fps> \
    --demo-video dataset/<idname>/smirk_demo/demo_ext.mp4 --demo-fps <fps> \
    --demo-ext-pose --overwrite
```

**補助フラグ（いずれも demo 描画専用で `.frame` に影響なし）：**

- `--demo-lock-bbox` — 先頭フレームの bbox を全フレームに固定。残存 jitter が
  encoder 由来か bbox 由来かを切り分ける。
- `--demo-smooth-bbox N` — `(2N+1)` フレーム中心平均。lock と baseline の中間。
- `--demo-ext-pose` — 旧「`pose` を外部 R に畳み込む」convention で再投影。
  既存学習済みモデルの描画を再現したい／A/B 比較したい時のみ使います。
- `--demo-lbs-pose` — **deprecated no-op**（LBS が既定化済み）。後方互換で
  受理されますが何もしません。

### 3. 学習

`finalize` まで済んだら `train.py` を走らせます。SMIRK 由来か
metrical-tracker 由来かで学習コード側の分岐はありません（同じ `.frame`
形式を消費）。

```bash
python train.py --idname <idname>
# チェックポイントは logs/<idname>/ 配下に出力
```

主なオプションは `arguments/` 配下のパーサ定義を参照。SMIRK + bbox 安定化
で作った `.frame` でも、metrical-tracker の `.frame` と同じ設定で学習できます。

### 4. テスト（レンダリング）

学習済みチェックポイントで test 分割をレンダリングします。

```bash
python test.py --idname <idname>
# logs/<idname>/test.avi に書き出す（既定では全フレーム）
```

### 5. 再実行のコツ

- **SMIRK ステップの再実行は安価**です。`.frame` ファイルは同じディレクトリに
  上書き書き込まれ、以降の `finalize` / `train.py` / `test.py` は再キックで
  同じ結果になります。`--bbox-mode` を `legacy` と `offline` で往復したい時は
  `--overwrite` をつけます。
- `--bbox-mode` を切り替えた後、**`finalize` を再実行する必要はありません**
  （finalize が読むのは `checkpoint_raw/*.frame` で、その内容が丸ごと差し替わる
  だけなので）。ただし学習済みモデルは `.frame` の内容に依存するので、モードを
  切り替えたら**再学習が必要**な点に注意してください。

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

## Demo オーバーレイ（`--demo-video`）

`--verify-dir` は数枚だけのスポット確認ですが、**連続再生で jitter や
アライメントを目視・時系列比較したい**場合は `--demo-video` を使います。

```bash
python scripts/preprocess.py smirk --idname <idname> \
    --demo-video demo.mp4 --demo-fps 30
```

出力:

- `demo.mp4` — 元フレームに以下をオーバーレイした mp4:
  - **MediaPipe bbox**（水色＝検出成功、アンバー＝前フレーム再利用、
    中心にクロスマーカー）
  - **MediaPipe landmarks**（赤の点、468 点）
  - **FLAME メッシュ再投影**（緑の点、stride=8 で ~630 頂点）
  - 左上 HUD: `f=フレーム idx  det=検出フラグ  bbox=サイズ  Z=深度
    pose=ext|lbs`
- `demo_stats.csv` — 各フレームの `bbox_center, bbox_size, t_x/y/z` と
  それらの前フレーム差分（`d_*`）、`speed_bbox_center`, `speed_t`
  （L2 ノルム差）。jitter を時系列数値として可視化できます。

### Jitter 要因の切り分けフラグ

いずれも **demo 描画にのみ**適用され、`.frame` ファイルには影響しません。

| フラグ | 用途 |
|---|---|
| `--demo-lock-bbox` | 先頭フレームの `bbox_center, bbox_size` を全フレームに固定。jitter が消えれば bbox 起因、残れば SMIRK encoder 起因。 |
| `--demo-smooth-bbox N` | `(2N+1)` フレーム中心平均 bbox。lock-bbox と no-op の中間強度。offline 診断専用（causal 制約なし）。 |
| `--demo-ext-pose` | 旧「`pose` を外部 R に畳み込む（原点中心回転）」convention で再投影。診断・A/B 比較専用。**通常は使わず LBS 既定のまま**で十分です。 |
| `--demo-lbs-pose` | **deprecated no-op**（LBS が既定化済み）。後方互換のため受理されますが何もしません。 |
| `--demo-fps HZ` | mp4 の再生 fps ヒント（既定 25）。SMIRK 処理はフレームインデックス駆動なのでエンコーダへの指示のみ。 |
| `--demo-vertex-stride N` | 投影する FLAME 頂点のサンプリング間隔（既定 8、V=5023 なので ~630 点）。`1` で全頂点を描画（SMIRK `release/cuda128` PR #9 の `--show_vertices` に相当）。 |
| `--demo-vertex-radius PX` | 各頂点ドットの絶対半径 px（既定 1、LINE_AA 付きで ~3x3 ブロブ）。`0` で**単一ピクセル直書き**（LINE_AA なし、SMIRK PR #9 の新既定に一致）。低解像度パネル向け。full-frame 1080p 以上なら既定のままでも視認性は問題ありません。`--demo-vertex-radius-rel` 指定時は無視。 |
| `--demo-vertex-radius-rel FRAC` | 頂点ドット半径を `min(frame_h, frame_w)` に対する割合で指定（SMIRK の `--vertex_radius_rel` と同等）。異なる解像度の overlay を横並びで比較したいときに便利（例: 1080p で 0.001 ≈ 1 px、0.0005 で単一ピクセル）。 |

典型的な使用パターン（3 本並べて目視比較）:

```bash
# baseline（LBS 既定・bbox も生値）
python scripts/preprocess.py smirk --idname <idname> \
    --demo-video demo_baseline.mp4 --demo-fps 30

# bbox 固定で bbox 起因 jitter を切り離したい
python scripts/preprocess.py smirk --idname <idname> \
    --demo-video demo_locked.mp4 --demo-fps 30 --demo-lock-bbox

# 旧 ext-R convention を再現（診断目的のみ）
python scripts/preprocess.py smirk --idname <idname> \
    --demo-video demo_ext.mp4 --demo-fps 30 --demo-ext-pose
```

HUD の `pose=ext|lbs` 表示はそのまま残っています。既定では `lbs` と
描画され、`--demo-ext-pose` を付けたときだけ `ext` になります。

## BBox 安定化（`--bbox-mode`）

SMIRK `release/cuda128` (PR #7) で **crop 用 bbox の安定化**が追加されました。
従来の `crop_face` は MediaPipe ランドマーク **全点** の min/max から
bbox を導出していたため、口の開閉・瞬き・検出ノイズがそのまま bbox の
`size` に漏れ、SMIRK が予測する正射影カメラが振動し、再投影メッシュが
耳・頭頂付近で「呼吸」するように見えていました。

PR #7 は 2 段構成で対策します。

1. **安定ランドマーク部分集合**：目尻・鼻根・こめかみなど、
   **発話・瞬きで動かない** 15 点だけで bbox を導出。口が開いても
   bbox `size` は広がらない。
2. **時間方向の平滑化**：`size` 系列（必要なら `center` も）に LPF を掛ける。
   オンライン／オフラインで**使うフィルタが違う**。

FlashAvatar 側では `--bbox-mode` フラグから 3 つのモードを選べます。

| モード | 挙動 | 用途 | パス数 | `.frame` への影響 |
|---|---|---|---|---|
| `legacy`（**既定**） | 全ランドマーク min/max・平滑化なし（PR#7 前の挙動） | 既存 `.frame` とのビット互換維持 | 1 | なし（既存と一致） |
| `online` | 安定部分集合 + **One-Euro フィルタ**（CHI 2012）を `size` に逐次適用。O(1) 状態でリアルタイム／Webcam 相当 | 消費側が「オンライン平滑化された」入力を期待するとき | 1 | あり |
| `offline` | 安定部分集合 + **ゼロ位相 FIR 低域通過**を全 `size` 系列に事前一括適用。最初に全フレームを走査して `size/center` を集めてからフィルタ→エンコード | 教師データ作成・最高品質。短いクリップ不可 | 2 | あり |

> **重要 — online と offline は意味が違います。**
>
> - **online**：One-Euro は **逐次（causal）** フィルタで、過去サンプルだけから
>   現在の平滑値を計算します。ウォームアップで小さな遅延があり、強い
>   平滑化のもとでも位相遅れは数フレームで収束します。**SMIRK 公式デモの
>   新しい既定**（`--bbox_mode online`）と同じ実装です。
> - **offline**：ゼロ位相 FIR（`scipy.signal.firwin` + 対称畳み込み + エッジ
>   パディング）で、**前向き／後向き両方向**の信号を参照します。位相遅れ
>   ゼロですが、シーケンスが短いと（FIR タップ長の数倍必要）エラー。
>   **先頭・末尾数十フレーム**はエッジ効果を受けます。

### 使い方

online（リアルタイム相当、fps 必須）：

```bash
python scripts/preprocess.py smirk --idname <idname> \
    --bbox-mode online --bbox-fps 30 \
    --demo-video demo_online.mp4 --demo-fps 30
```

offline（最高品質、fps 必須）：

```bash
python scripts/preprocess.py smirk --idname <idname> \
    --bbox-mode offline --bbox-fps 30 \
    --offline-size-cutoff 2.5 \
    --demo-video demo_offline.mp4 --demo-fps 30
```

既存の挙動（`--bbox-mode` 指定なし＝`legacy`）：

```bash
# これまでどおり。bbox は全ランドマーク min/max で導出される。
python scripts/preprocess.py smirk --idname <idname>
```

### フラグ

| フラグ | 既定 | 意味 |
|---|---|---|
| `--bbox-mode` | `legacy` | `legacy` / `online` / `offline`。既存 `.frame` とのビット互換維持が必要なら `legacy`。 |
| `--bbox-all-landmarks` | off | 安定部分集合を使わず全ランドマーク min/max に戻す。安定部分集合が顔以外に落ちる特殊ケースの救済弁。通常は既定のまま。 |
| `--bbox-size-calibration SCALE` | なし（SMIRK 既定 `1.55`） | **安定部分集合**で導出した `size` を legacy 相当のクロップ範囲に合わせるための補正倍率（SMIRK `release/cuda128` PR #8）。安定部分集合は口・顎・眉・額を含まないため、そのまま使うと縦方向の extent が全顔の ~30% しかなく、`online` / `offline` の crop が `legacy` の ~60% の大きさにしかなりません。その結果、再投影された FLAME メッシュや `_draw_mesh` で描かれる**頂点点群が縮小されて**見えます（本バグの直接原因）。未指定のままで SMIRK 側の既定 `STABLE_LANDMARK_SIZE_CALIBRATION = 1.55` が使われ、legacy の crop extent に一致します。診断目的で補正を無効化したいときは `1.0` を渡します。`--bbox-all-landmarks` 指定時や `legacy` モード時は no-op。 |
| `--bbox-fps HZ` | なし | 映像 FPS。`online` / `offline` で**必須**。cutoff の Nyquist 正規化に使う。 |
| `--online-size-min-cutoff HZ` | 1.0 | One-Euro の静止時カットオフ。小さいほど静止時に強く平滑化。 |
| `--online-size-beta` | 0.02 | One-Euro の速度感度。大きいほど速い動きで素早く追従。 |
| `--online-center-cutoff HZ` | なし | `center` も One-Euro 平滑化したいとき指定。通常は未指定のまま（頭部並進は正直に追従させる）。 |
| `--online-center-beta` | 0.02 | `center` 用の One-Euro beta（`--online-center-cutoff` 指定時のみ使用）。 |
| `--offline-size-cutoff HZ` | 2.5 | `size` 用 FIR の cutoff。安定部分集合を使うと `size` はカメラ距離だけ持つ信号になるので 2〜3 Hz で安全。 |
| `--offline-size-taps` | 61 | FIR タップ数（奇数に切り上げ）。シーケンス長より小さい必要あり。30 fps で ~1 s の群遅延（エッジ補正で相殺）。 |
| `--offline-center-cutoff HZ` | なし | `center` 用 FIR の cutoff。未指定で `center` は生値のまま。 |

### 設計指針

- **既定は `legacy`**：`.frame` のビット互換を崩さない設計で、PR#7 を取り込む
  ためだけに既存の学習済みモデルを再学習させる必要はない。
- **新規データで高品質が欲しい**なら `offline`。シーケンスが十分長い
  （目安 FIR タップの 3 倍以上 = 数百フレーム）前提。口の開閉・瞬きの影響が
  メッシュ揺れに現れる動画でまず試す。
- **真のリアルタイム用途**または online 前提の消費側に合わせたい場合は
  `online`。`.frame` は causal 平滑の結果を格納する。
- **`--bbox-fps` は正確に**。フレーム抽出時の `fps=30` 指定や原動画の
  FPS と一致させる。誤差があると cutoff が意図と違う Hz になる。
- **FLAME パラメータへの LPF (`--lpf-cutoff`) との違い**：
  - **`--bbox-mode` は encoder の入力段**を安定化する。`cam` / `t` / `bbox_size`
    の振動源を**上流で**断つ。
  - **`--lpf-cutoff` は encoder の出力**（FLAME パラメータ系列）を平滑化する。
    bbox 安定化で残った residual jitter（特に encoder そのものの出力揺れ）を
    下流で抑える。
  - 併用可。Listening Head Gen の教師データ作成では
    `--bbox-mode offline --lpf-cutoff ...` を重ねて掛けるのが推奨。

### 注意点

- **offline と短いクリップ**：`fir_lowpass_offline` はシーケンス長がタップ数
  より短いと**そのまま素通し（フィルタ無効）**にフォールバックします。
  暗黙に効かなくなるので、`--demo-video` の stats で平滑化が効いているか
  一度は目視確認してください。
- **online の冒頭数フレーム**は One-Euro の初期化中でほぼ未平滑です。学習
  データに使うときは冒頭 ~30 フレームを捨てると安全。
- **`--bbox-mode offline` は 2-pass** なので MediaPipe 検出コストが実質 1 回
  分増えます（encoder は 1 回しか走らせない。ランドマークはキャッシュされる）。
  encoder が律速でないクリップでは所要時間が倍近くになる点に注意。
- **MediaPipe VIDEO モードの時系列状態**：offline の pass-2 では
  MediaPipe を**再実行しません**（pass-1 のランドマークを使い回し）。
  VIDEO トラッキング状態が二重に進行するのを避け、決定性を保つためです。
- **`eye-mode blendshapes`** や FLAME `--lpf-*` との併用は自由。どちらも
  本機能と直交します。
- **安定部分集合のサイズ補正（PR #8）**：安定ランドマーク部分集合は
  口・顎・眉・額を含まないため、縦方向の extent は全顔の ~30% しかあり
  ません。そのまま `(width + height) / 2` で `size` を求めると legacy
  ベースラインの ~60% のクロップになり、`_build_t` を通すと**再投影された
  FLAME メッシュと頂点点群が縦横ともに縮小されて**見えます（本タスクで
  報告された `demo_lbs.mp4` の vertex_point 縮小は直接これが原因）。
  SMIRK `release/cuda128` PR #8 は `extract_bbox_center_size(...,
  size_calibration=1.55)` を導入し、安定部分集合モードでも legacy と
  同じクロップ extent を出すようにしました。FlashAvatar は
  `SmirkConfig.size_calibration` / `--bbox-size-calibration` 経由でこの値を
  そのまま素通しで渡し、未指定時は SMIRK 側の既定 `1.55` を使います。
  旧 SMIRK チェックアウト（PR #8 以前）では `load_bbox_tracker` が
  **明示的に RuntimeError を投げ**、`git pull` の指示を出します
  （silently に 60% クロップで学習が進むより fail-loud 優先）。
- **頂点可視化ユーティリティ（PR #9）**：SMIRK 側は `demos/{demo,demo_video,
  demo_webcam}.py` に重複していた `alpha_blend_mesh_over_input /
  ndc_to_crop_pixels / crop_pixels_to_full_pixels / draw_vertex_points_*`
  を `utils/vertex_viz.py` に集約し、**頂点ドットの既定半径を `1`
  (LINE_AA) → `0`（単一ピクセル直書き）に変更**しました（5023 点を
  224x224 パネルに描画すると LINE_AA 1px が 3x3 ブロブ化してパネルが
  潰れるため）。FlashAvatar の `preprocess/smirk_demo.py` は SMIRK
  renderer を使わず FLAME を自前で投影するため `ndc_to_crop_pixels` 等の
  座標変換ヘルパは不要ですが、**頂点ドットの粒度コントロール**は同じ
  恩恵を受けます。`--demo-vertex-stride` / `--demo-vertex-radius` /
  `--demo-vertex-radius-rel` を追加し、`--demo-vertex-radius 0` で
  SMIRK PR #9 と同じ「単一ピクセル直書き」になります。全 5023 点を
  描画して形状ドリフトを目視確認したい時は
  `--demo-vertex-stride 1 --demo-vertex-radius 0` が推奨値です。

### demo 用フラグとの関係

既存の `--demo-lock-bbox` / `--demo-smooth-bbox` は **demo 描画専用**で、
`.frame` には影響しませんでした。本 `--bbox-mode` は逆で、demo 描画にも
`.frame` にも影響します（encoder 入力そのものを変えるため）。

- jitter 源の診断が目的 → `--demo-lock-bbox` / `--demo-smooth-bbox` を使う。
- 実際に学習・出力を改善したい → `--bbox-mode {online, offline}` を使う。

`--bbox-mode` と `--demo-video` は**併用可能**です。demo オーバーレイは
`FrameResult.bbox_center` / `bbox_size` をそのまま読むため、
`--bbox-mode online` / `offline` を指定した状態では demo 上の bbox 枠・
HUD の `bbox=...`・stats CSV の `bbox_*` 列は **すべて平滑化後の値**
になります。平滑化前後を見比べたい場合は 2 回走らせて出力を並べます。

レンダリング convention は LBS が既定なので、`--demo-lbs-pose` のような
opt-in フラグは不要です（書いても deprecated no-op として受理されます）。

```bash
# 生の bbox（all-landmarks min/max、未平滑）+ LBS 既定
python scripts/preprocess.py smirk --idname <idname> \
    --bbox-mode legacy \
    --demo-video dataset/<idname>/smirk_demo/demo_legacy.mp4 --demo-fps 30

# 安定化後の bbox（stable subset + zero-phase FIR）+ LBS 既定
python scripts/preprocess.py smirk --idname <idname> \
    --bbox-mode offline --bbox-fps 30 \
    --demo-video dataset/<idname>/smirk_demo/demo_offline.mp4 --demo-fps 30 \
    --overwrite
```

なお `--demo-lock-bbox` / `--demo-smooth-bbox` を `--bbox-mode online/offline`
の上に重ねて指定することも可能ですが、**平滑化後の値を更に固定／平均する**
動作になる点に注意してください（素の raw bbox が欲しい場合は上の 2 走行
を比較するのが早いです）。

例（本タスクで共有されたデモコマンドに bbox 安定化を足すケース）：

```bash
python scripts/preprocess.py smirk --idname Mikawa3 \
    --bbox-mode offline --bbox-fps 30 \
    --demo-video dataset/Mikawa3/smirk_demo/demo_lbs.mp4 --demo-fps 30
# 既存 .frame を上書き再生成したい場合は `--overwrite` を追加。
# 旧互換で `--demo-lbs-pose` を残しても no-op として受理されます。
# `--bbox-size-calibration` は未指定で OK（SMIRK 側の 1.55 が自動で掛かり、
# legacy と同じクロップ extent になります）。旧 SMIRK チェックアウトでは
# ロード時に明示エラーとなるので、指示に従って `git pull` してください。
```

## 時間方向 LPF（`--lpf-cutoff`）

Listening Head Generation の学習では 1 次・2 次差分（速度・加速度）
特徴を使うため、**サブピクセル級の per-frame jitter が微分で爆発的に
増幅**します。過去のプロジェクトでは教師データの速度・加速度が jitter
支配で使い物にならなかった実績があり、教師データ作成段階で zero-phase
FIR LPF を掛けるのが標準的な対策です。

```bash
python scripts/preprocess.py smirk --idname <idname> \
    --lpf-cutoff 2.0 --lpf-fps 30 \
    --demo-video demo_lpf.mp4 --demo-fps 30
```

### フラグ

| フラグ | 既定 | 意味 |
|---|---|---|
| `--lpf-cutoff HZ` | なし | カットオフ周波数（Hz）。指定すると LPF が有効。 |
| `--lpf-fps HZ` | なし | 映像 FPS。`cutoff / (fps/2)` で Nyquist 正規化するため、`--lpf-cutoff` と同時指定必須。 |
| `--lpf-window-sec SEC` | 1.0 | フィルタ窓の秒数。タップ数 = `round(window_sec × fps)` を奇数に丸め。 |
| `--lpf-channels LIST` | `cam,pose` | カンマ区切り。使用可能: `cam, pose, exp, jaw, eyelids`。 |

### 設計指針

- **Cutoff (`--lpf-cutoff`)**: 物理的に想定される最大動作周波数の 1〜2 倍。
  - 頭部位置・向き (`cam, pose`): 2〜3 Hz で十分（人が意図的に振る頭の動きは ~2 Hz 以下）
  - 発話中の顎 (`jaw`): 4〜6 Hz（音節速度を超えない範囲）
  - 瞬き (`eyelids`): 6〜8 Hz（閉じる相が ~100 ms と速い）
  - 表情 (`exp`): 3〜5 Hz が無難。強すぎるとマイクロ表情が消える
- **Window (`--lpf-window-sec`)**: 遷移帯域幅の逆数に比例。通常は 1.0〜1.5 秒
  - 窓が長い → 遷移帯が狭い（よく切れる）、ただし端っこが `filtfilt`
    padding の影響を受けるフレーム数が増える
  - 窓が短い → 端の影響は小さいが、通過帯/阻止帯の境目がぼやける
  - 経験則: `window_sec × cutoff_hz ≈ 2〜4` が扱いやすい
- **Channels (`--lpf-channels`)**:
  - **安全な既定** (`cam,pose`): 頭部位置・向きだけ平滑化し、表情系は無加工
  - **Listening Head Gen 用の推奨** (`cam,pose,exp,jaw,eyelids`): 全チャネル平滑化
  - **shape は対象外**: `canonicalize_shape` で sequence 内唯一の値に
    畳み込まれているため LPF 対象にならない

### Zero-phase（filtfilt）

実装は `scipy.signal.filtfilt` による前向き・後向き 2 回適用で、
**位相遅延ゼロ**です。速度・加速度特徴のタイムスタンプが元映像と
ずれないので、後段で音声・テキストとアラインするときに時刻補正が
不要です。

代償として、`filtfilt` はシーケンス冒頭と末尾の数十フレーム
（＝窓長程度）で端効果があります。Listening Head Gen の学習では
先頭・末尾を数秒捨てるのが一般的なので通常は気になりませんが、
短いクリップでは注意してください（`filtfilt` の要求を満たさず
エラーになる場合あり: 最小 `3 × filter_length` 超のフレーム数が必要）。

### 注意点

- **axis-angle を直接平滑化**しています。頭部・顎の典型動作範囲
  （主枝 ±π 内）では安全ですが、大きな回転で branch cut を跨ぐと
  不連続ポイントで乱れる可能性あり。そういうケースに当たったら
  rotation matrix / quaternion 空間での平均化に拡張してください。
- `--lpf-fps` は映像の実 FPS と一致させてください（`preprocess prepare`
  で ffmpeg に設定したもの、または原動画の FPS）。間違えると
  cutoff が意図と違う Hz になります。

## 回転中心問題（解決済み — LBS convention がデフォルト）

> **状態：RESOLVED.** 以前は「次作業の引き継ぎ」として残していた
> 「原点中心回転による系統的 jitter」問題を修正済みです。**LBS
> convention が FlashAvatar 全体のデフォルト**になりました。
> 記録のために症状・原因・どこを変えたかを残します。

### 症状（過去）

- 頭部が正面向きのときはメッシュが顔によく重なる
- 頭部を左右に振ると、メッシュが**奥側にある**ようにズレ（z 軸方向
  の系統的オフセット）
- 瞬き・顎の開閉の瞬間、**耳・頭頂部付近で特に顕著な位置変動**が
  発生（メッシュ全体が小さく揺れる）

### 原因（過去）

FLAME の `pose_params` は本来、**root joint を中心とした回転**
として LBS 内部で適用される。しかし修正前のパイプラインは:

- `preprocess/smirk_convert.py::_build_R` — `pose` を外部 R 行列に
  畳み込み: `R = GL_TO_CV @ axis_angle_to_matrix(pose)`
- `src/deform_model.py::decode` — `forward_geo(rot_params=None)` で
  canonical メッシュを得、外部 R に pose を依存

の形になっていて、**原点中心の回転**として pose を適用していた。
真の root joint と canonical 原点にはオフセット `J_root` があるため、
pose 変化 δθ に対し `δθ × J_root` ぶんの余分な並進が生じ、それが
見かけの z 軸オフセットや瞬き連動の揺れとして可視化されていた。

### 実施した修正

| ファイル | 変更内容 |
|---|---|
| `preprocess/smirk_convert.py::to_flashavatar_frame` | `flame_dict` に `pose`（SMIRK の `pose_params` を rot6d 化）を追加。`opencv.R` は `_GL_TO_CV`（純粋な OpenGL→OpenCV 座標反転）のみ。 |
| `scene/__init__.py::Scene_mica` | `flame_params['pose']` があれば読み出して `head_pose` とし、`Camera` に渡す。**無ければ**単位 rot6d にフォールバック（後方互換）。 |
| `scene/cameras.py::Camera` | `head_pose`（rot6d）を受け取り、GPU 上に保持。未指定時は単位 rot6d。 |
| `src/deform_model.py::decode` | `codedict['head_pose']` を `flame_model.forward_geo(..., rot_params=...)` に渡し、LBS で root joint を中心に回転。未指定時は単位 rot6d。 |
| `train.py` / `test.py` | `codedict['head_pose'] = viewpoint_cam.head_pose`。 |
| `preprocess/smirk_verify.py` | 検証オーバーレイも LBS 経由で投影するように変更（R は座標反転のみ、pose は `forward_geo(rot_params=...)`）。 |
| `preprocess/smirk_demo.py` | `use_lbs_pose` デフォルトを `True` に変更。legacy 挙動を見たいときは `use_ext_pose=True` に opt-in。 |
| `preprocess/cli.py` | `--demo-lbs-pose` は **deprecated no-op**（既に LBS 既定なので何もしない）。legacy 再現用に `--demo-ext-pose` を新設。 |
| `utils/flame_converter.py` | `convert()` の出力 `flame` dict に `pose` を追加。`convert_flame_params_only()` の codedict に `head_pose` を追加（DECA/EMOCA/SMIRK/SPARK 共通）。 |

### 後方互換性

**新規 `.frame`（本 PR 以降に生成）** — `flame.pose` を持ち、`opencv.R`
は純粋な座標反転。`scene` はこの両方を尊重して LBS で pose を適用します。

**旧 `.frame`（本 PR より前に生成。metrical-tracker 由来を含む）** —
`flame.pose` を持たない。`scene` は単位 rot6d をフォールバックとして
`head_pose` に入れ、結果として LBS は無回転メッシュを出力、`opencv.R`
（そこに pose が畳み込まれている）が rasterizer 側で頭を回す形になる。
**従来と同じ出力**が得られるので、**既存学習済みモデルは再学習なしで
そのまま動作**します。

### 移行戦略（新規学習時）

既存データで新しく学習する場合は、以下のどちらかです：

1. **`.frame` を再生成して再学習**（推奨） — `--bbox-mode offline` など
   の他の新機能と同じタイミングで LBS 版の `.frame` に更新し、そこから
   再学習。口の開閉・瞬き連動の揺れ・左右振り時の奥行きずれが消えます。
2. **旧 `.frame` のまま既存モデルを使い続ける** — 後方互換パスが効くので
   何もしなくても壊れません（ただし上記の揺れは残ります）。

`.frame` を混ぜるのは避けてください（`flame.pose` の有無はフレーム単位
で判定されるため実害はありませんが、学習データと test データで convention
が食い違うと混乱の元です）。

### 残りうる純 encoder 起因の揺れ

上記の LBS 化で「原点中心回転」由来の揺れは消えました。`--demo-lock-bbox`
や `--demo-smooth-bbox` と併用しても観測される以下は、SMIRK encoder 自体の
性質による残留揺れです：

- 口の開閉時に bbox 縦幅が変動 → `s, cam` がわずかに揺れる
- メッシュがわずかに膨張/収縮して見える

この残留揺れには `--bbox-mode {online, offline}`（bbox 安定化・本リポの
前段修正）または `--lpf-cutoff`（FLAME パラメータ平滑化）が効きます。
実際、`--bbox-mode offline --lpf-cutoff 2.0` の 3 層適用で教師データ品質が
大きく上がる構成です。

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
