# 使用例

`data_driven_demo.cpp` と `data_driven_demo.py` は一次遅れプラントの記録データを使い、
4手法で制御器を調整します。サンプリング周期は0.1秒です。
目標モデルは遅れを含む `M(z)=0.3 z^-1/(1-0.7 z^-1)` です。

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
./build/data_driven_demo
python sample/data_driven_demo.py
```

VRFTの両版とFRIT反復版の正解は `Kp=1.2, Ki=3.0` です。
FRIT直接版は `C^-1=(rho0+rho1 z^-1)/(1.5-1.2 z^-1)` とパラメータ化し、
`rho=[1,-1]` を求めます。その逆を取ると同じPI制御器になります。

C++サンプルは推定係数・目的関数・反復回数・状態・ステップ追従誤差を表示し、
期待精度を満たさない場合は失敗終了します。Pythonサンプルは参照サブモジュールの
独立実行可能な例を呼び出します。

実測データを使う場合は、同じサンプリング周期で記録した `u`（制御入力）と
`y`（出力）に置き換え、目標モデルと制御器基底を設定してください。
モデル `G` はサンプルのデータ生成と検証にだけ使用し、調整関数へ渡しません。
反復版では `initial`, `lower`, `upper` を指定し、戻り値の収束状態を確認します。

調整後のC++制御器は、状態を保持して1サンプルずつ実行できます。
初期化時に係数を正規化し、`step` では固定サイズの状態だけを更新します。

```cpp
auto controller = forward_basis.controller(result.parameters);
auto state = controller.state();
double control_input = 0.0;
auto status = state.step(reference - measured_output, control_input);
// 各周期でstateを保持し、statusを確認してcontrol_inputを使用します。
```

`simulation_manager/` はシミュレーション可視化・信号生成用のサブモジュールです。
Pythonのオプション表示に利用し、コアライブラリと通常のCIはNumPyだけで計算できます。
