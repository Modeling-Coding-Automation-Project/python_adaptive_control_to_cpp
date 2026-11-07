# VRFT／FRITのサンプルとC++生成

手法ごとにPythonとC++のサンプルを分けています。

| 手法 | Python設計・コード生成・表示 | 生成される設定 | C++実行例 |
| --- | --- | --- | --- |
| VRFT | `vrft_demo.py` | `generated/vrft_demo.hpp` | `vrft_demo.cpp` |
| FRIT | `frit_demo.py` | `generated/frit_demo.hpp` | `frit_demo.cpp` |

どちらも直接最小二乗法とPANOCを比較します。サンプリング周期は0.1秒、
プラントは `G=0.2 z^-1/(1-0.8 z^-1)`、目標モデルは
`M=0.3 z^-1/(1-0.7 z^-1)` です。調整関数にはプラントモデルを渡さず、
記録した入出力と目標モデル、制御器基底だけを渡します。

## 実行と表示

リポジトリのルートで実行します。

```sh
python -m pip install -r requirements.txt
python sample/vrft_demo.py
python sample/frit_demo.py
```

通常実行で `SimulationPlotterDash` を使用し、目標値・目標応答・初期制御器・
調整後の応答を同じグラフへ描画します。HTMLは実行時のカレントディレクトリの
`cache/` に保存され、パスをコンソールへ表示してブラウザを開きます。
ヘッドレス環境で画面が開かない場合は、そのHTMLをブラウザで開いてください。

自動テストで表示だけを省く場合も、C++生成とシミュレーションは実行します。

```sh
python sample/vrft_demo.py --no-plot
python sample/frit_demo.py --no-plot
```

## PythonからC++へ

MPCリポジトリのdeployクラスと同様に、Pythonで設計したモデル・制御器を
`data_driven/data_driven_deploy.py` の `DataDrivenDeploy` でC++ヘッダへ変換します。
生成ヘッダには固定サイズの型、実験データ、目標モデル、基底、オプション、
チューナー生成関数、Pythonで調整済みの制御器生成関数が含まれます。

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
./build/vrft_demo
./build/frit_demo
```

C++サンプルは生成ヘッダを読み込み、同じデータで調整を再実行してPythonの係数と
比較します。また、Pythonで調整済みの制御器をそのまま実行する例も含みます。
生成済みヘッダを追跡しているため、Pythonを実行できない環境でもビルドできます。
Python側のモデルや設定を変更したときは、サンプルを実行して再生成してください。

```sh
python sample/vrft_demo.py --no-plot --check-generated
python sample/frit_demo.py --no-plot --check-generated
```

`--check-generated` は現在のPython環境で生成した内容と保存済みヘッダを比較します。
NumPyのバージョンなどで最下位桁が変わった場合は、再生成してC++テストを実行してください。

本フォルダのPythonサンプルと
`external_libraries/MCAP_python_adaptive_control/sample/data_driven/` の同名ファイルは
バイト単位で同一です。それぞれのサンプルが自身のリポジトリを見つけ、ライブラリを
読み込みます。他方のサンプルをimportしたり、実行したりしません。
Python参照リポジトリだけで実行した場合は、設計・シミュレーション・表示を行います。
本リポジトリで実行した場合には、C++のdeployモジュールを利用してコード生成も行います。

## 係数の意味と実機への適用

VRFTの両方式とFRIT反復版の正解は `Kp=1.2, Ki=3.0` です。
FRIT直接版は逆制御器
`C^-1=(rho0+rho1 z^-1)/(1.5-1.2 z^-1)` の係数 `rho=[1,-1]` を求めます。
通常の制御器へ戻すには、求めた伝達関数の逆を取ります。

反復版の `tolerance=2e-5` は射影残差の停止条件です。係数そのものの誤差は
基底の条件数などにも依存するため、必要な精度と応答を個別に確認してください。

調整済み制御器は、状態を保持して1サンプルずつ実行します。

```cpp
// #include "generated/vrft_demo.hpp"
auto controller = vrft_demo::make_optimized_controller();
auto state = controller.state();
double control_input = 0.0;
auto status = state.step(reference - measured_output, control_input);
// stateを各周期で保持し、statusを確認してcontrol_inputを使用します。
```
