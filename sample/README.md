# VRFT／FRITのC++サンプル

C++だけで設定・実験・調整・応答評価を実行します。
Pythonのデモと同じ計算手順を示し、Pythonの事前実行やコード生成は必要ありません。

| 手法 | C++サンプル | 対応するPythonデモ |
| --- | --- | --- |
| VRFT | [data_driven/vrft_demo.cpp](data_driven/vrft_demo.cpp) | [vrft_demo.py](../external_libraries/MCAP_python_adaptive_control/sample/data_driven/vrft_demo.py) |
| FRIT | [data_driven/frit_demo.cpp](data_driven/frit_demo.cpp) | [frit_demo.py](../external_libraries/MCAP_python_adaptive_control/sample/data_driven/frit_demo.py) |

## 設計ワークフロー

`create_design()` で問題を定義し、`run_demo(design)` で次の手順を実行します。

1. 初期PIのゲインを `Kp=0.7, Ki=1.0` に固定し、零初期状態から単位ステップの
   閉ループ実験を128サンプル実行します。`r_ini, u_ini, y_ini` を記録し、実験中は調整しません。
2. 記録済みの同じ `u_ini, y_ini` を使い、最小二乗法と上下限制約付きPANOCで
   オフライン調整します。最適化の反復中にプラントの実験をやり直しません。
3. 初期・直接法・反復法の制御器を構成します。FRIT直接法は逆制御器を同定するため、
   伝達関数を反転してからフィードバック制御に使用します。
4. 各制御器とプラントの状態をゼロに戻し、独立した閉ループで100サンプルの
   単位ステップ応答を計算します。目標モデルの応答と各出力・操作量、追従RMSEを表示します。

サンプリング周期は0.1秒、目標モデルは `M(z)=0.3*z^-1/(1-0.7*z^-1)` です。
シミュレーションのプラントは `y(k)=x(k)`、`x(k+1)=0.8*x(k)+0.2*u(k)` とし、
現在の出力から操作量を計算してから状態を更新します。入出力間の遅延は1サンプルです。
プラントの係数は実験と応答評価に使い、調整器には渡しません。
実機を使う場合は、最初の実験ループを測定済みの入出力データに置き換えます。

VRFTは `L=M` のフィルタを使い、`M u_ini` と `C(rho)(I-M)y_ini` の差を小さくします。
FRITは記録データから `r_tilde=C(rho)^-1 u_ini+y_ini` を作り、
`y_ini` と `M r_tilde` の差を小さくします。
これらの評価と閉ループ応答の評価は別の段階として示しています。

## C++だけでビルド・実行する

リポジトリのルートで実行します。

```sh
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++
cmake --build build --parallel 2
./build/vrft_demo
./build/frit_demo
```

C++は設計の各段階を説明し、調整係数・状態・評価関数値・反復回数・RMSEを表示します。
時系列もCSV形式でターミナルへ出力します。HTMLやプロットは生成しません。

| 出力 | 列 |
| --- | --- |
| 初期PIによる実験 | `experiment_time,experiment_reference,experiment_input,experiment_output` |
| 閉ループ応答 | `time,reference,desired,y_initial,u_initial,y_direct,u_direct,y_iterative,u_iterative` |

`initial` は初期制御器、`direct` は最小二乗法、`iterative` はPANOCで調整した制御器です。
正常終了時は `return 0;` で終了します。

## コード生成が不要な理由

このデモのモデルと制御器基底は、既存の `RationalFilter` と `LinearControllerBasis`
のコンストラクタで直接作れます。データ長・係数数・フィルタ容量も、C++のコンパイル時定数と
テンプレート引数で指定できます。現在のデモは係数数2、フィルタ容量1のPI制御器です。
Pythonで記号式を展開したり、C++では表現できない構造を導出したりする処理はありません。
従来の生成処理はこれらの型と設定を書き出していたため、C++で同じ定義を記述すれば十分です。

周期、初期ゲイン、目標モデル、上下限、ソルバー設定などはC++の `create_design()` で変更します。
実験長やシミュレーション長、係数数・フィルタ容量を変更する場合は、対応するコンパイル時定数・
型と係数配列のサイズを揃えてビルドし直します。Pythonによる再生成は不要です。
両言語の一致性を確認する場合は、Python側も同じ設計条件に揃えてください。

## Pythonの辞書とC++構造体

設定、調整結果、各制御器、時系列を名前付きの構造体で保持します。
`create_design()` と `run_demo()` の入出力を辿ると、Pythonデモの辞書と対応付けられます。

| Pythonの辞書 | C++の型とメンバー |
| --- | --- |
| `design` | `Design`：周期、プラント係数、モデル、基底、初期係数、上下限、オプション |
| `fits` | `Fits`：`direct`、`iterative` |
| `controllers` | `Controllers`（`ControllerValues<Filter>`）：`initial`、`direct`、`iterative` |
| `trajectories`、`controls` | `ControllerValues<Trace>`：各制御器の時系列 |
| `controller_states`、`plant_states` | `ControllerValues<FilterState>`、`ControllerValues<Scalar>`：各ループの独立した状態 |
| `run_demo()` の返す辞書 | `DemoResult`：設定、調整結果、制御器、記録データ、応答時系列をまとめて保持 |

`Design::sample_count` と `Design::simulation_count` は、固定サイズ配列の長さに使う
コンパイル時定数です。辞書の文字列キーは、C++では型の決まったメンバー名になります。

ライブラリは失敗の種類を `Status` で返します。ライブラリ内部に例外送出やエラー表示は追加しません。
デモ側が戻り値を確認し、継続できない場合に実行時エラーとして扱い、`main()` で表示します。

## Python表示とSIL

Python側の描画を確認する場合は、Python参照リポジトリのデモを直接実行します。

```sh
python -m pip install -r requirements.txt
python external_libraries/MCAP_python_adaptive_control/sample/data_driven/vrft_demo.py
python external_libraries/MCAP_python_adaptive_control/sample/data_driven/frit_demo.py
```

Pythonは `SimulationPlotterDash` で「データ取得」「記録データを再利用するオフライン調整」
「初期・調整後制御器の閉ループ評価」を別ページに表示します。描画不要の場合は `--no-plot` を付けます。
本リポジトリのC++サンプルは、Pythonや `simulation_manager` を実行時に参照しません。

数値の一致性は次のSILで確認します。

```sh
python test_sil/test_samples.py
```

SILはC++サンプルをビルドし、対応するPythonデモの計算結果と、実験の参照値・入出力、
調整係数、全時刻の閉ループ出力・操作量、RMSEを比較します。

## 係数の意味

このプラントと目標モデルでは、VRFTの両方式とFRIT反復版で `Kp=1.2, Ki=3.0` が目安になります。
FRIT直接版は `C^-1=(rho0+rho1*z^-1)/(1.5-1.2*z^-1)` の係数を求めるので、
`rho=[1,-1]` をPIゲインとして使わないでください。

反復版の `tolerance=2e-5` は射影残差の停止条件です。係数の誤差は基底の条件数にも依存するため、
調整状態と閉ループ応答を確認します。
