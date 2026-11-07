# データ駆動制御ライブラリ 実装・検証レポート

作成日：2026-11-07

## 実装範囲

『データ駆動制御入門』を参照し、VRFTとFRITについて、反復最適化を使う版と
直接最小二乗法で解く版の計4方式を実装しました。NumPy版は
`external_libraries/MCAP_python_adaptive_control/data_driven/`、C++11版は
本リポジトリの `data_driven/` にあります。

| 手法 | 関数 | 制御器のパラメータ化 | 解法 |
| --- | --- | --- | --- |
| VRFT直接版 | `vrft_least_squares` | `C=sum rho_i beta_i` | C++はGivens QR、PythonはNumPy lstsq |
| VRFT反復版 | `vrft_optimize` | 同上 | 既存PANOC、上下限制約付き |
| FRIT直接版 | `frit_least_squares` | **`C^-1=sum rho_i beta_i`** | C++はGivens QR、PythonはNumPy lstsq |
| FRIT反復版 | `frit_optimize` | `C=sum rho_i beta_i` | 既存PANOC、解析勾配、上下限制約付き |

FRIT直接版は書籍§4.5.1の逆線形パラメータ化を用いた厳密な解法です。
その戻り値は通常のPI/PIDゲインではありません。書籍§4.5.3のPI近似を
一般的なFRIT最小二乗法とみなす実装はしていません。
各目的関数と書籍の参照ページは [アルゴリズム仕様](algorithms.md) にまとめました。

## リポジトリ構成

作業開始時に `/opt/ModelingCodingAutomationProject/python_mpc_to_cpp` の
公開ヘッダ・依存関係・単体テスト・SIL・サンプル・CIを調べ、
同じ役割分担を採用しました。Python側は
`/opt/ModelingCodingAutomationProject/MCAP_python_mpc` を参考にしました。

- `data_driven/`：共通型、フィルタ、最小二乗、最適化アダプタ、VRFT、FRIT、統合ヘッダ。
- `test_vs/`：`MCAP_tester` とGCCで実行する単体テスト。Visual Studioプロジェクトは作成していません。
- `test_sil/`：C++が出力する数値をPythonで比較するSIL。pybind11を必要としません。
- `sample/`：4方式のPI調整と閉ループ応答を確認するC++/Python例、`simulation_manager`。
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

1. **C++のノルムの下限**：固定した依存版では平方根の入力が `1e-10` 以上に
   丸められ、零ベクトルのノルムも `1e-5` になります。そのままでは
   `1e-9` などの収束許容値に到達できません。本ライブラリ専用の
   `ParameterVector` 型だけを `std::hypot` によるノルムへ対応させました。
   他の行列型の動作やサブモジュールのソースは変更していません。
2. **1変数の型変換**：既存C++リングバッファが1要素の場合にスカラーを返すため、
   専用型で変換を補いました。比例1変数、PI2変数、PID3変数で確認しました。
3. **最終評価値**：PANOCが保持する評価値と返す実行可能点の時点が異なる場合があるため、
   返却パラメータの目的関数・勾配・境界値・射影残差を再評価します。

依存するC++最適化の固定リビジョンは `205069211b5ba2364ba65f3087ac58faa113c5ac` です。
上流PANOCのラインサーチ上限時のフォールバック処理はそのまま継承しています。
任意の非線形問題への収束を保証するものではなく、上限到達・非有限計算は成功と区別します。

## 組み込み向けの設計

- C++11、ヘッダオンリー、`float` / `double` 対応。サンプル数・係数数・次数はコンパイル時に固定します。
- 回帰行列・フィルタ状態・QR作業領域は `std::array`。直接解法の実行中にヒープ確保がないことをテストしました。
- 直接解法は正規方程式の逆算を避け、Givens回転で逐次QR分解します。
- VRFT反復版は小さな二次形式を最初に構成し、以後の目的関数・勾配評価をデータ長に依存しない `O(p^2)` にします。
- FRIT反復版は感度フィルタによる解析勾配を使用します。同じ候補でのcost/gradient重複計算をキャッシュします。
- バッチフィルタは係数の正規化を初回にまとめます。調整後の制御器は `state().step(...)` により1周期ずつ実行できます。
- 反復回数を制限し、係数上下限・ランク不足・逆制御器の特異性・非有限値を検出します。

調整計算のバッファは概ね `O(n p + p^2)` で、固定サイズでもデータ長に比例した
メモリが必要です。調整後の制御器の1周期処理は `O(Order)` です。
PANOCのコールバックは上流の `std::function` を使用するため、全標準ライブラリで
初期設定時を含む動的確保ゼロを保証するものではありません。
マイコン実機での実行時間や最大スタック使用量は測定していません。

## 検証結果

ローカル環境：GCC 15.2.0、CMake 4.2.3、Python 3.14.4、NumPy 2.5.4、pytest 9.2.0。
C++は `-std=c++11 -Wall -Wextra -Wpedantic -Werror` を使用しました。

| 検証 | 結果 |
| --- | --- |
| GCC Releaseビルド | 成功 |
| CTest：C++単体テストとサンプル | 2/2成功 |
| Python単体テスト | 31/31成功 |
| C++/Pythonの4方式サンプル | 成功 |
| SIL：完全一致可能なデータ | 成功 |
| SIL：雑音＋正則化＋先頭サンプル除外 | 成功 |
| SIL：目標モデルの2サンプル遅延 | 成功 |
| AddressSanitizer / UndefinedBehaviorSanitizer | CTest 2/2成功 |
| 新規リモートクローンからの再現 | ビルド・CTest・Python・SIL・サンプルが成功 |

単体テストでは、解析的に既知の係数、独立した差分方程式の目的関数、有限差分との
解析勾配比較、悪条件のQR、ランク不足、正則化、上下限が有効なケース、反復上限、
NaN/Inf、不正引数、状態リセット、遅延、フィルタ処理後の区間除外を確認しています。
C++依存ヘッダを先に読み込む場合も検証しました。

SILでは係数に加え、回帰行列、目的変数、評価値、勾配、閉ループ応答を比較しました。
最大係数差は直接解法で `4.88e-15`、反復版で `3.19e-9` でした。
反復版は演算順序・停止点が異なり得るため、ビット単位の一致は要件としていません。

PIサンプルでは正解 `Kp=1.2, Ki=3.0` を再現し、FRIT直接版は
逆制御器の係数 `[1,-1]` を再現します。C++サンプルの反復回数は
VRFTが9回、FRITが13回でした。

再現性検証は、本体 `d1cc065`、Python `22867ae` を新規クローンし、
再帰依存80件を指定リビジョンで取得して実施しました。
その後のCI認証設定・ドキュメント変更は数値実装を変更していません。

## GitHub Actionsの設定

ワークフローは指定ブランチのpush、Pull Request、手動実行に対応しています。
本体はPython 3.10 / 3.12のテストとサニタイザ、Pythonリポジトリ側は
3.10 / 3.12 / 3.14のテストを定義しています。
DependabotはActions、サブモジュール、Python依存を更新対象にします。

実際の [初回Actions実行](https://github.com/Modeling-Coding-Automation-Project/python_adaptive_control_to_cpp/actions/runs/42366044828)
は、非公開の `MCAP_python_adaptive_control` を既定の `GITHUB_TOKEN` で
取得できず、checkoutで停止しました。リモートCIの成功とは報告していません。
確認時点で本体のActionsシークレットは未登録でした。

実行には以下の設定が必要です。

1. 本体と再帰サブモジュール群のリポジトリ内容を読み取れるトークンを用意します。
   Fine-grained PATなら対象リポジトリの `Contents: Read-only` を付けます。
2. GitHubの本体リポジトリで **Settings → Secrets and variables → Actions → New repository secret** を開き、
   名前を `MCAP_SUBMODULE_TOKEN` として登録します。値はソースコードやチャットへ記載しません。
3. Pythonリポジトリの独立CIにも、非公開依存の取得が必要なら同名のシークレットを登録します。
4. Actionsの **Data-driven control** を対象ブランチで手動実行するか、失敗ジョブを再実行します。

ワークフローは `secrets.MCAP_SUBMODULE_TOKEN` を優先し、未設定時は
`github.token` を使います。組織が承認やSSOを要求する場合は、その条件も満たす必要があります。
既存のローカルGit認証情報をシークレットへ転載する操作は行っていません。

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
