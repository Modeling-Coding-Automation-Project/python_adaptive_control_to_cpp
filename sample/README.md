# VRFT／FRITのサンプルとC++生成

手法ごとにPythonとC++のサンプルを分け、両言語で同じ設計ワークフローを示します。

| 手法 | Python設計・コード生成・表示 | 生成される設定 | C++実行例 |
| --- | --- | --- | --- |
| VRFT | `vrft_demo.py` | `../vrft_demo.hpp` | `vrft_demo.cpp` |
| FRIT | `frit_demo.py` | `../frit_demo.hpp` | `frit_demo.cpp` |

どちらも直接最小二乗法とPANOCを比較します。サンプリング周期は0.1秒、
プラントは `G=0.2 z^-1/(1-0.8 z^-1)`、目標モデルは
`M=0.3 z^-1/(1-0.7 z^-1)` です。調整関数にはプラントモデルを渡さず、
記録した入出力と目標モデル、制御器基底だけを渡します。

## 設計ワークフロー

1. Pythonの `create_design()` で、実験長、シミュレーション長、目標モデル、制御器基底、
   初期係数、係数上下限、ソルバー設定を決めます。この問題に対応するC++の型と設定を生成します。
2. 128サンプルの入力をプラントへ与え、入出力 `u, y` を収集します。
   PythonとC++は、同じシードの信号とプラントの状態更新から、それぞれデータを作ります。
3. 収集したデータで直接法と反復法を実行し、返された係数から制御器を構成します。
   FRIT直接法では逆制御器を同定するため、伝達関数の逆を取ってから使用します。
4. 初期制御器、直接法で調整した制御器、反復法で調整した制御器を、独立した零初期状態で
   100サンプル実行します。各周期で参照値と測定出力から操作量を求め、プラントを更新します。
5. Pythonは時系列をDashで表示し、C++は同じ信号をターミナルへ出力します。

シミュレーションのプラントは `y(k)=x(k)`、`x(k+1)=0.8*x(k)+0.2*u(k)` です。
現在の出力から操作量を計算し、その後に状態を更新するので、入出力間に1サンプルの遅延があります。
プラントの式はデモの実験・シミュレーションに使用し、調整アルゴリズムには渡しません。

## 実行と表示

リポジトリのルートで実行します。

```sh
python -m pip install -r requirements.txt
python sample/vrft_demo.py
python sample/frit_demo.py
```

通常実行で `SimulationPlotterDash` を使用し、目標値・目標応答・各制御器の出力、
各操作量、実験時の入出力を描画します。HTMLは実行時のカレントディレクトリの
`cache/` に保存され、パスをコンソールへ表示してブラウザを開きます。
ヘッドレス環境で画面が開かない場合は、そのHTMLをブラウザで開いてください。

自動テストで表示だけを省く場合も、C++生成とシミュレーションは実行します。

```sh
python sample/vrft_demo.py --no-plot
python sample/frit_demo.py --no-plot
```

## PythonからC++へ

MPCリポジトリのdeployクラスと同様に、Pythonで定義した問題の構造と設定を
`data_driven/data_driven_deploy.py` の `DataDrivenDeploy` でC++ヘッダへ変換します。
importには `data_driven.data_driven_deploy` のように、リポジトリ内の階層を明示します。
既定の生成先はリポジトリ直下で、実行時のカレントディレクトリには依存しません。
別の出力先が必要な場合は `--output-dir` で指定できます。
生成ヘッダには固定サイズの型、サンプル数、周期、実験信号のシード、シミュレーション用の
プラント係数、目標モデル、制御器基底、オプション、初期係数・上下限、チューナー生成関数を含めます。
生成は調整計算の前に行い、観測データ、調整済み係数、正解照合用の値は出力しません。
データ収集と制御器の調整はC++サンプル自身が実行します。

生成ヘッダはGitで追跡しないため、初回ビルドの前に上記のPythonサンプルを実行してください。

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
./build/vrft_demo
./build/frit_demo
```

C++は調整の状態・係数・評価関数値に加えて、実験と閉ループシミュレーションの時系列を
CSV形式でターミナルへ出力し、最後は `return 0;` で終了します。C++からHTMLは生成しません。

| 出力 | 列 |
| --- | --- |
| 実験データ | `experiment_time,experiment_input,experiment_output` |
| 閉ループ応答 | `time,reference,desired,y_initial,u_initial,y_direct,u_direct,y_iterative,u_iterative` |

`initial` は初期制御器、`direct` は直接法、`iterative` はPANOCで調整した制御器です。
サンプル内で正解と照合する代わりに、`test_sil/test_codegen.py` がPythonとC++の
実験データ・調整結果・全時刻の出力と操作量を比較します。

`create_design()` の周期、実験長、基底の係数数・次数、目標モデル、上下限などを変更したら、
Pythonを再実行してヘッダを生成し、C++をビルドし直してください。
サンプルを含めずにC++の単体テストをビルドする場合は、
`-DADAPTIVE_CONTROL_BUILD_SAMPLES=OFF` をCMakeへ指定します。

```sh
python sample/vrft_demo.py --no-plot --check-generated
python sample/frit_demo.py --no-plot --check-generated
```

`--check-generated` は現在のPython環境で生成した内容と保存済みヘッダを比較します。
設定を変更した場合は、再生成してC++をビルドし直してください。

本フォルダのPythonサンプルと
`external_libraries/MCAP_python_adaptive_control/sample/data_driven/` の同名ファイルは
バイト単位で同一です。それぞれのサンプルが自身のリポジトリを見つけ、ライブラリを
読み込みます。他方のサンプルをimportしたり、実行したりしません。
Python参照リポジトリだけで実行した場合は、設計・シミュレーション・表示を行います。
本リポジトリで実行した場合には、C++のdeployモジュールを利用してコード生成も行います。

## 係数の意味と実機への適用

このプラントと目標モデルでは、VRFTの両方式とFRIT反復版で `Kp=1.2, Ki=3.0` が目安になります。
FRIT直接版は逆制御器
`C^-1=(rho0+rho1 z^-1)/(1.5-1.2 z^-1)` の係数 `rho=[1,-1]` を求めます。
通常の制御器へ戻すには、求めた伝達関数の逆を取ります。

反復版の `tolerance=2e-5` は射影残差の停止条件です。係数そのものの誤差は
基底の条件数などにも依存するため、必要な精度と応答を個別に確認してください。

調整済み制御器は、状態を保持して1サンプルずつ実行します。

```cpp
// basisとfitは、サンプルで構成した制御器基底と調整結果です。
auto controller = basis.controller(fit.parameters);
auto state = controller.state();
double control_input = 0.0;
auto status = state.step(reference - measured_output, control_input);
// stateを各周期で保持し、statusを確認してcontrol_inputを使用します。
```
