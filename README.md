# データ駆動適応制御ライブラリ（Python / C++11）

一組の入出力データから制御器を調整するVRFTとFRITを実装しています。
Python NumPy版と対応する、固定サイズ・ヘッダオンリーのC++11ライブラリです。

| 手法 | 直接解法（反復最適化なし） | 反復最適化 |
| --- | --- | --- |
| VRFT | 線形制御器のQR最小二乗法 | 同じ評価関数を既存PANOCで最小化、係数上下限に対応 |
| FRIT | **逆制御器が線形**の場合のQR最小二乗法 | 通常の制御器の逆を計算し、解析勾配と既存PANOCで最小化 |

FRITの直接解法が返すのは逆制御器の係数です。通常のPIDゲインとは区別してください。
詳細な式・適用条件は [アルゴリズム仕様](document/vrft_frit_algorithms.md) に記載しています。
既存制御ライブラリへの変更案は [最小二乗クラスの切片選択追加](document/least_squares_intercept_extension.md) を参照してください。

## 構成

- `data_driven/`: C++ライブラリ。Python版とフィルタ・最小二乗・最適化・VRFT・FRITの構造を対応させています。
- `external_libraries/MCAP_python_adaptive_control/`: NumPy参照実装、Pythonテスト、サンプル。
- `test_vs/`: MCAP_testerによるGCC単体テスト。Visual Studioプロジェクトは使用しません。
- `test_sil/frit/`、`test_sil/vrft/`: サンプルの手順に沿ってPython/C++のライブラリAPIを比較するSIL。
- `sample/data_driven/`: C++だけで実行するVRFT／FRIT別のPI調整例。
- `.github/`: GitHub ActionsとDependabot。

## ビルドと確認

C++サンプルと単体テストはGCCとCMake 3.16以降だけでビルドできます。
Pythonの事前実行や生成ヘッダは不要です。

```sh
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
./build/vrft_demo
./build/frit_demo
```

両デモは初期PIを固定した閉ループ実験で128点の入出力を収集し、
同じデータを使ってオフラインで直接法とPANOCを実行します。
その後、初期・直接法・反復法の制御器を独立した零初期状態から100時刻実行します。
C++は実験データ、調整状態・係数、各時刻の参照値・目標応答・出力・操作量、
追従RMSEをターミナルへ表示します。
Pythonの辞書に対応する設定・結果はC++の構造体で保持します。

Python参照実装とのSILには、Python 3.10以降と開発用ヘッダ、CMake 3.18以降、
pybind11などの依存パッケージを使用します。Ubuntuでは `python3-dev` を導入してください。

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest external_libraries/MCAP_python_adaptive_control/tests -q
python test_sil/vrft/vrft_demo_SIL.py
python test_sil/frit/frit_demo_SIL.py
```

各SILは対応するC++ラッパーをビルドして読み込み、データ収集・オフライン調整・
閉ループ評価の各段階でPython/C++の戻り値を比較します。ビルド生成物は一時フォルダへ置きます。

対応するPythonデモは `external_libraries/MCAP_python_adaptive_control/sample/data_driven/`
にあります。Python側は `SimulationPlotterDash` で描画し、C++側はプロットを行いません。
両言語の計算手順とデータの対応、設定変更方法は [サンプル説明](sample/README.md) を参照してください。

組み込み先のCMakeからは `add_subdirectory(...)` と
`target_link_libraries(your_target PRIVATE MCAP::AdaptiveControl)` を使用できます。
ライブラリだけを組み込む場合は、`add_subdirectory(...)` より前に
`ADAPTIVE_CONTROL_BUILD_TESTS` と `ADAPTIVE_CONTROL_BUILD_SAMPLES` を `OFF` に設定します。
`data_driven.hpp` をインクルードします。
使用例は [sample/README.md](sample/README.md) を参照してください。

コアはSISO・離散時間・零初期状態を対象とします。データ長・パラメータ数・
フィルタ次数はC++のコンパイル時に決定します。調整結果は状態コードとともに返り、
実機に適用する前に対象の閉ループ応答を検証する必要があります。

数値演算には `python_math_to_cpp`、直接最小二乗には
`python_numpy_linalg_qr.hpp` の既存QRを使用します。
反復ソルバーの既定許容値は `2e-5` です。これは射影残差の許容値であり、
推定係数そのものの絶対誤差とは異なります。

[実装・検証レポート](document/implementation_report.md) / [MIT License](LICENSE.txt)
