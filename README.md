# データ駆動適応制御ライブラリ（Python / C++11）

一組の入出力データから制御器を調整するVRFTとFRITを実装しています。
Python NumPy版と対応する、固定サイズ・ヘッダオンリーのC++11ライブラリです。

| 手法 | 直接解法（反復最適化なし） | 反復最適化 |
| --- | --- | --- |
| VRFT | 線形制御器のQR最小二乗法 | 同じ評価関数を既存PANOCで最小化、係数上下限に対応 |
| FRIT | **逆制御器が線形**の場合のQR最小二乗法 | 通常の制御器の逆を計算し、解析勾配と既存PANOCで最小化 |

FRITの直接解法が返すのは逆制御器の係数です。通常のPIDゲインとは区別してください。
詳細な式・適用条件は [アルゴリズム仕様](document/algorithms.md) に記載しています。

## 構成

- `data_driven/`: C++ライブラリ。Python版とフィルタ・最小二乗・最適化・VRFT・FRITの構造を対応させています。
- `external_libraries/MCAP_python_adaptive_control/`: NumPy参照実装、Pythonテスト、サンプル。
- `test_vs/`: MCAP_testerによるGCC単体テスト。Visual Studioプロジェクトは使用しません。
- `test_sil/`: 同一データによるPython/C++一致性検証。
- `sample/`: C++/PythonのPI調整例、`simulation_manager`。
- `.github/`: GitHub ActionsとDependabot。

## ビルドと確認

GCC、CMake 3.16以降、Python 3.10以降を使用します。

```sh
git submodule update --init --recursive
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
python -m pytest external_libraries/MCAP_python_adaptive_control/tests -q
python test_sil/test_data_driven_sil.py --executable build/data_driven_sil
python sample/data_driven_demo.py
./build/data_driven_demo
```

組み込み先のCMakeからは `add_subdirectory(...)` と
`target_link_libraries(your_target PRIVATE MCAP::AdaptiveControl)` を使用できます。
`data_driven.hpp` をインクルードします。
使用例は [sample/README.md](sample/README.md) を参照してください。

コアはSISO・離散時間・零初期状態を対象とします。データ長・パラメータ数・
フィルタ次数はC++のコンパイル時に決定します。調整結果は状態コードとともに返り、
実機に適用する前に対象の閉ループ応答を検証する必要があります。

[実装・検証レポート](document/implementation_report.md) / [MIT License](LICENSE.txt)
