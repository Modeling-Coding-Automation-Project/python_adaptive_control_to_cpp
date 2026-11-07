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
- `test_sil/`: 同一データによるPython/C++一致性検証。
- `sample/`: VRFT／FRIT別のPI調整例、PythonからのC++コード生成、`simulation_manager`。
- `.github/`: GitHub ActionsとDependabot。

## ビルドと確認

GCC、CMake 3.16以降、Python 3.10以降を使用します。

```sh
git submodule update --init --recursive
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python sample/vrft_demo.py
python sample/frit_demo.py
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
python -m pytest external_libraries/MCAP_python_adaptive_control/tests -q
python test_sil/test_data_driven_sil.py --executable build/data_driven_sil
python test_sil/test_codegen.py
./build/vrft_demo
./build/frit_demo
```

Pythonサンプルは通常実行で `SimulationPlotterDash` のグラフをHTMLへ保存し、
ブラウザを開きます。画面を開けない環境では表示された保存先を開いてください。
表示不要の自動実行では `--no-plot` を指定します。
設計した問題のサイズ・目標モデル・制御器基底などを、リポジトリ直下の
`vrft_demo.hpp`／`frit_demo.hpp` へ生成します。生成先はカレントディレクトリに依存しません。
生成ヘッダは追跡しないため、C++サンプルのビルド前に対応するPythonを実行してください。

PythonとC++は、実験データの収集、直接法と反復法による調整、調整した制御器の構成、
閉ループシミュレーションを同じ順序で実行します。C++も入出力データを収集して調整を行い、
各時刻の参照値・目標応答・プラント出力・操作量をターミナルへ表示して `return 0;` で終了します。
時系列と調整結果のPython／C++一致性は `test_sil/test_codegen.py` で検証します。

Pythonを使わずにC++ライブラリと単体テストだけをビルドする場合は、
CMakeに `-DADAPTIVE_CONTROL_BUILD_SAMPLES=OFF` を指定します。

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
