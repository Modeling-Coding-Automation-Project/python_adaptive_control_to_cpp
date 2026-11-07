# データ駆動制御ライブラリ 実装・検証レポート

作成日：2026-11-07

本レポートは、既存数学・QRライブラリの利用、ヘッダ改名、サンプル分割と
C++コード生成・Dash表示への変更を反映しています。

## 実装範囲

『データ駆動制御入門』を参照し、VRFTとFRITについて、反復最適化を使う版と
直接最小二乗法で解く版の計4方式を実装しました。NumPy版は
`external_libraries/MCAP_python_adaptive_control/data_driven/`、C++11版は
本リポジトリの `data_driven/` にあります。

| 手法 | 関数 | 制御器のパラメータ化 | 解法 |
| --- | --- | --- | --- |
| VRFT直接版 | `vrft_least_squares` | `C=sum rho_i beta_i` | C++は既存QR、PythonはNumPy lstsq |
| VRFT反復版 | `vrft_optimize` | 同上 | 既存PANOC、上下限制約付き |
| FRIT直接版 | `frit_least_squares` | **`C^-1=sum rho_i beta_i`** | C++は既存QR、PythonはNumPy lstsq |
| FRIT反復版 | `frit_optimize` | `C=sum rho_i beta_i` | 既存PANOC、解析勾配、上下限制約付き |

FRIT直接版は書籍§4.5.1の逆線形パラメータ化を用いた厳密な解法です。
その戻り値は通常のPI/PIDゲインではありません。書籍§4.5.3のPI近似を
一般的なFRIT最小二乗法とみなす実装はしていません。
各目的関数と書籍の参照ページは [アルゴリズム仕様](vrft_frit_algorithms.md) にまとめました。

## リポジトリ構成

作業開始時に `/opt/ModelingCodingAutomationProject/python_mpc_to_cpp` の
公開ヘッダ・依存関係・単体テスト・SIL・サンプル・CIを調べ、
同じ役割分担を採用しました。Python側は
`/opt/ModelingCodingAutomationProject/MCAP_python_mpc` を参考にしました。

- `data_driven/`：`data_driven_common.hpp`、`data_driven_filter.hpp`、
  `data_driven_least_squares.hpp`、`data_driven_optimization.hpp`、
  `data_driven_vrft.hpp`、`data_driven_frit.hpp`、統合ヘッダ `data_driven.hpp`。
  Pythonからのコード生成は `data_driven_deploy.py` にあります。
- `test_vs/`：`MCAP_tester` とGCCで実行する単体テスト。Visual Studioプロジェクトは作成していません。
- `test_sil/`：C++が出力する数値をPythonで比較するSIL。pybind11を必要としません。
- `sample/`：VRFTとFRITに分けたC++/Python例、`simulation_manager`。
- `vrft_demo.hpp`、`frit_demo.hpp`：リポジトリ直下の生成済みサンプル設定ヘッダ。
- `.github/`：C++テスト、Pythonテスト、サンプル、SIL、サニタイザ、Dependabot。
- `document/`：数式・適用条件・本レポート。

不要と指定された旧 `python_adaptive_control/` と、旧実装を参照するサンプルを削除しました。
PDFは `.gitignore` に明示し、コミット対象に含めていません。
書籍の抽出テキストやページ画像も追加していません。

## 依存ライブラリと最適化の再利用

本体のサブモジュールは次の通りです。

| 配置 | 用途 |
| --- | --- |
| `external_libraries/MCAP_python_adaptive_control` | NumPy参照実装 |
| `external_libraries/python_optimization_to_cpp` | C++ PANOC |
| `external_libraries/python_numpy_to_cpp` | PANOCの固定サイズ行列 |
| `external_libraries/python_math_to_cpp` | 行列・最適化の数値処理 |
| `external_libraries/base_utility_cpp` | 基盤ユーティリティ |
| `external_libraries/python_control_to_cpp` | 既存最適化ヘッダが要求する制御系の型 |
| `test_vs/MCAP_tester` | C++要素テスト |
| `sample/simulation_manager` | シミュレーション支援 |

Pythonリポジトリには `external_libraries/MCAP_python_optimization` と
`sample/simulation_manager` を追加しました。
両言語とも既存PANOCを再利用し、独自の反復最適化アルゴリズムは追加していません。
ただし、既存実装をそのまま呼び出すだけでは解決できなかった点を、次のアダプタで吸収しました。

1. **数学ライブラリとノルム**：`std::hypot` を廃止し、既存の `PythonMath::abs` と
   `PythonMath::sqrt` を使用します。平方根の入力下限は、収束判定だけでなく
   PANOC内部の微小摂動によるLipschitz推定にも影響します。許容値を緩めるだけでは
   検証ケースのFRIT係数に約0.13の差が出たため、専用 `ParameterVector` 型では
   最大絶対値で正規化した後に平方根を呼び、元の尺度へ戻します。
   このとき平方根の入力は1以上になり、零ベクトルは明示的に0を返します。
   標準数学関数は使用せず、他の行列型やサブモジュールの動作も変更しません。
2. **1変数の型変換**：既存C++リングバッファが1要素の場合にスカラーを返すため、
   専用型で変換を補いました。比例1変数、PI2変数、PID3変数で確認しました。
3. **最終評価値**：PANOCが保持する評価値と返す実行可能点の時点が異なる場合があるため、
   返却パラメータの目的関数・勾配・境界値・射影残差を再評価します。

反復版の既定許容値は両言語とも `2e-5` です。これは射影残差の停止条件であり、
推定係数の絶対誤差ではありません。係数の精度が必要な悪条件のケースでは、
許容値を明示して厳しく設定し、目的関数と応答を確認します。

### 既存最小二乗・QRの利用方法

`python_control_least_squares.hpp` の `PythonControl::LeastSquares::fit` は、
切片の列を無条件に追加します。そのまま利用するとVRFT／FRITの制御器基底にない
自由パラメータが増えて別の問題になるため、このAPIの直接利用は適しません。
この理由から、`python_numpy_linalg_qr.hpp` の `PythonNumpy::LinalgSolverQR` を使用します。

独自のGivens回転と後退代入は削除しました。既存QRには1観測ずつ
`(p+1)×(p+1)` の拡大行列を渡し、QR作業領域を `O(p^2)` に保ちます。
全データを一度に渡すと既存実装は `n×n` のQを保持するため、この方式を選びました。
後退代入も同じAPIを呼びます。固定した依存版のRHS列数に依存する分岐には、
p列のRHSの先頭列だけを使用する形で対応し、1～3変数で検証しました。

本リポジトリのコア・サンプル・テストの実装から `<cmath>` と `<algorithm>` の
直接includeおよび標準数学関数の呼び出しを除去しました。有限値判定は
`numeric_limits` と比較演算で行います。依存ライブラリ内部の既存includeや
`std::isfinite`、MCAP_tester内の `std::abs` までは変更していません。

依存するC++最適化の固定リビジョンは `205069211b5ba2364ba65f3087ac58faa113c5ac` です。
上流PANOCのラインサーチ上限時のフォールバック処理はそのまま継承しています。
任意の非線形問題への収束を保証するものではなく、上限到達・非有限計算は成功と区別します。

## 組み込み向けの設計

- C++11、ヘッダオンリー、`float` / `double` 対応。サンプル数・係数数・次数はコンパイル時に固定します。
- 回帰行列・フィルタ状態・QR作業領域は `std::array`。直接解法の実行中にヒープ確保がないことをテストしました。
- 直接解法は正規方程式の逆算を避け、既存QRで逐次処理します。
- VRFT反復版は小さな二次形式を最初に構成し、以後の目的関数・勾配評価をデータ長に依存しない `O(p^2)` にします。
- FRIT反復版は感度フィルタによる解析勾配を使用します。同じ候補でのcost/gradient重複計算をキャッシュします。
- バッチフィルタは係数の正規化を初回にまとめます。調整後の制御器は `state().step(...)` により1周期ずつ実行できます。
- 反復回数を制限し、係数上下限・ランク不足・逆制御器の特異性・非有限値を検出します。

調整計算のバッファは概ね `O(n p + p^2)` で、固定サイズでもデータ長に比例した
メモリが必要です。逐次QRの回転演算は `O(n p^2)`、依存実装の行列コピーを
含めると保守的には `O(n p^3)` です。通常の比例・PI・PIDではpは1～3です。
調整後の制御器の1周期処理は `O(Order)` です。
PANOCのコールバックは上流の `std::function` を使用するため、全標準ライブラリで
初期設定時を含む動的確保ゼロを保証するものではありません。
マイコン実機での実行時間や最大スタック使用量は測定していません。

## サンプル・コード生成・表示

`sample/vrft_demo.py` と `sample/frit_demo.py` は、Python参照リポジトリの
`sample/data_driven/` にある同名ファイルとバイト単位で同一です。
それぞれ自身の場所からリポジトリを見つけ、ライブラリを読み込みます。
本リポジトリでは `external_libraries.MCAP_python_adaptive_control.data_driven` と
`data_driven.data_driven_deploy` のように、import先の階層を明示しています。
旧 `runpy` ラッパーと統合デモは削除し、他方のサンプルへの参照をなくしました。

本リポジトリではPythonサンプルの実行時に `DataDrivenDeploy` を呼び、
リポジトリ直下へ `vrft_demo.hpp` または `frit_demo.hpp` を生成します。
既定の生成先は実行時のカレントディレクトリに依存せず、`--output-dir` で変更できます。
モデル、基底、実験データ、オプション、チューナーと調整済み制御器の生成関数を
C++へ出力し、対応するC++サンプルが実際に使用します。
MPCリポジトリの「Python設計→deployクラス→生成ヘッダ→C++実行」に合わせた構成です。
生成済みヘッダは引き続き追跡し、Pythonを実行しないC++ビルドにも対応します。
C++サンプルは数値結果と検証のPASS／FAILをターミナルに表示して `return 0;` で終了し、
CTestとコード生成統合テストは標準出力のPASS／FAILを検証します。

通常実行では `SimulationPlotterDash` でHTMLを保存し、パスを表示してブラウザを開きます。
画面表示を省く場合だけ `--no-plot` を付けます。Plotlyを依存に追加しました。
表示と生成の操作手順は [サンプル説明](../sample/README.md) に記載しています。

## 検証結果

ローカル環境：GCC 15.2.0、CMake 4.2.3、Python 3.14.4、NumPy 2.5.4、pytest 9.2.0。
C++は `-std=c++11 -Wall -Wextra -Wpedantic -Werror` を使用しました。

| 検証 | 結果 |
| --- | --- |
| GCC Releaseビルド | 成功 |
| CTest：C++単体テストとVRFT／FRITサンプル | 3/3成功 |
| Python単体テスト | 34/34成功 |
| C++/Pythonの4方式サンプル | 成功 |
| SIL：完全一致可能なデータ | 成功 |
| SIL：雑音＋正則化＋先頭サンプル除外 | 成功 |
| SIL：目標モデルの2サンプル遅延 | 成功 |
| コード生成統合テスト | 同一サンプル、別cwd、再生成、古い生成物検出、生成C++ビルド・実行が成功 |
| Dash表示経路 | 通常モードで両HTMLを生成、保存先表示を確認 |
| AddressSanitizer / UndefinedBehaviorSanitizer | CTest 3/3成功 |

単体テストでは、解析的に既知の係数、独立した差分方程式の目的関数、有限差分との
解析勾配比較、悪条件のQR、ランク不足、正則化、上下限が有効なケース、反復上限、
NaN/Inf、不正引数、状態リセット、遅延、フィルタ処理後の区間除外を確認しています。
C++依存ヘッダを先に読み込む場合も検証しました。

SILでは係数に加え、回帰行列、目的変数、評価値、勾配、閉ループ応答を比較しました。
最大係数差は直接解法で `3.17e-9`、反復版で `9.43e-10` でした。
反復版は演算順序・停止点が異なり得るため、ビット単位の一致は要件としていません。

PIサンプルでは正解 `Kp=1.2, Ki=3.0` を再現し、FRIT直接版は
逆制御器の係数 `[1,-1]` を再現します。C++サンプルの反復回数は
VRFTが8回、FRITが11回でした。FRIT反復版の係数は約 `[1.200094,2.999762]`、
ステップ応答RMSEは約 `7.32e-6` です。係数の絶対誤差と停止許容値は区別しています。

初版の再現性検証は、本体 `d1cc065`、Python `22867ae` を新規クローンし、
再帰依存80件を指定リビジョンで取得して実施しました。
今回の変更では、別ディレクトリでサンプルを実行し、そこに新しく生成した
ヘッダを使用してC++11をビルド・実行する統合テストを追加しました。

## GitHub Actionsの設定

ワークフローは指定ブランチのpush、Pull Request、手動実行に対応しています。
`python_mpc_to_cpp` と同じ構成・記述方法に合わせ、統合していた `test.yml` を
次のファイルに分割しました。ステップ名、aptによる依存導入、複数行の `run`、
GCCの直接実行、`/opt/venv_py_MCAP` の明示的な利用を参照先に揃えています。

| ファイル | 検証内容 |
| --- | --- |
| `.github/workflows/run_test_vs.yml` | `test_vs/test_data_driven.cpp` をGCC C++11でビルド・実行。ASan/UBSanによるCTestも実行 |
| `.github/workflows/run_SIL_test.yml` | `test_sil/data_driven_sil.cpp` をビルドし、Python／C++一致性を検証。Python参照実装の単体テストも実行 |
| `.github/workflows/run_samples.yml` | VRFT／FRITサンプル、Pythonサンプルの同一性、生成C++のビルド・実行、コード生成統合テスト |

本体はUbuntuランナーに用意された `python3` から仮想環境を作ってテストし、
Pythonバージョンのマトリクスと `actions/setup-python` は使用しません。
システムのPython環境へ直接pipでインストールせず、同じバージョンの仮想環境を使用します。
SILはJSONによるプロセス間通信を使うため、参照先のSILで使われるpybind11は不要です。
DependabotはActions、サブモジュール、Python依存を更新対象にします。

他のMCAPリポジトリと同じ標準の `actions/checkout` を使用します。
独自の `MCAP_SUBMODULE_TOKEN` 入力とシークレット必須チェックを削除しました。
Actionsが本体の取得に自動で使う `GITHUB_TOKEN` 以外に、トークンの登録は不要です。

この構成では、サブモジュールを公開リポジトリとして取得できる必要があります。
調査時、`MCAP_python_control`、`python_control_to_cpp`、`MCAP_python_optimization`
などは公開されていましたが、`MCAP_python_adaptive_control` だけは
匿名アクセスが404、既存ローカル認証では取得可能でした。
URLの誤記ではなく、公開状態の違いが最初のcheckout失敗の原因です。

2026-11-07にユーザーが `MCAP_python_adaptive_control` をPublicへ変更し、
認証情報を使用しない `git ls-remote` で取得可能になったことを確認しました。
さらに、Gitのシステム・ユーザー認証設定を無効にした新規クローンで、
再帰サブモジュール82件を固定リビジョンどおりに取得できました。
**Run SIL test** と **Run samples** は、シークレットの登録なしで再実行できます。
依存の取得やPython検証をスキップして成功扱いにする処理は追加していません。

`Run test VS` は、本体を通常の `GITHUB_TOKEN` で取得した後、
C++に必要な公開サブモジュール6件だけを非再帰で取得します。
このジョブはPython参照ライブラリを必要としません。

### PID反復解テストの収束条件

`optimized PID FRIT gains` の失敗は、floatの演算結果が変わる構成で再現しました。
PANOCの終了許容値は射影した更新幅のノルムに対する値であり、PIDゲインの
誤差を直接保証しません。従来のこのテストでは終了許容値 `1e-4` が粗く、
ゲインの合格範囲に達する前に正常終了する場合がありました。

テスト内のfloat用終了許容値を `1e-6` へ変更し、係数の合格範囲は
floatで `0.04`、doubleで `0.001` のまま維持しました。加えて、厳密解でゼロになる
目的関数がfloatでは `1e-7`、doubleでは `1e-12` 以下であることも確認します。
失敗時には型・実係数・期待値・状態・目的関数値・反復数を出力します。
ライブラリ本体の既定許容値 `2e-5` と実装、依存ライブラリは変更していません。

GCC 15でFMAを有効にすると、修正前のfloat FRITは `Ki=2.233591` となって
期待値 `2` との差が合格範囲を超えました。修正後は `Ki=2.000167` です。
GCC 13.4／14.3／15.2の通常の `-O2` と `-O2 -mfma` の計6構成で全単体テストが
成功しました。旧終了許容値へ戻した一時的な負例は失敗し、診断出力も確認しています。
特定のGCCバージョンや、GitHubランナーがFMAを使用したことを断定するものではありません。

匿名で取得した新規クローンに今回の修正を反映し、3ワークフローのビルド・実行コマンドも
ローカルで検証しました。単体テスト、サニタイザCTest 3件、SIL 3ケース、Python 34テスト、
両サンプルの描画・コード生成・C++実行、コード生成統合テストがすべて成功しています。
修正のコミット・push、およびGitHub上での再実行は行っていません。

## 適用条件と残る制約

- SISO、離散時間、線形時不変の基底、零初期状態を対象とします。
- 通常のPI/PID調整はVRFTまたはFRIT反復版、FRIT直接版は逆線形の制御器族を使います。
- VRFTの既定プレフィルタ `L=M` は遅延モデルを扱いやすくする選択で、一般の追従誤差最適性を保証しません。
- データの励起不足や極端な条件数は別途確認が必要です。C++のQRとPythonのSVDでは、閾値近傍の数値ランク判定が異なり得ます。
- 制御器逆系の評価が成立する探索領域を指定してください。目的関数の減少だけでは閉ループ安定性を保証しません。
- 現時点の実測確認はホスト上のGCCです。組み込みターゲットでの性能測定と実機応答確認は未実施です。

## コミット・プッシュ

本体・Pythonリポジトリともに `feature/create_data_driven` にのみコミット・プッシュしました。
本体の依存整理 `1b8046f`、実装とテスト `d1cc065`、Pythonの本実装 `4c19f51` と
入力検証改善 `22867ae` を区切りとして記録しています。
他の依存リポジトリにはコミット・プッシュしていません。
