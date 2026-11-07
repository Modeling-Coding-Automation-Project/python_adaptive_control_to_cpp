# 既存最小二乗クラスへ切片選択を追加する設計案

## 結論

`PythonControl::LeastSquares` に、コンパイル時オプション
`FitIntercept` を追加することを推奨します。既定値は `true` として現在の動作を保ち、
VRFT／FRITでは `false` を指定して、渡された回帰行列をそのまま解きます。

```cpp
// バッチLSへ追加した切片選択APIの使用例。
using X_Type = PythonNumpy::DenseMatrix_Type<double, 128, 2>;

PythonControl::LeastSquares<X_Type> conventional;       // 従来どおり切片あり
PythonControl::LeastSquares<X_Type, false> controller; // 切片なし、係数は2個

controller.fit(X, target);
auto rho = controller.get_weights(); // 2×1。余分な切片係数を含まない
```

本書は変更の設計案です。2026-11-07に、バッチLSの切片選択、Python対応、
コード生成、サンプル、単体テスト、SILを実装しました（未コミット）。
実装範囲と検証結果は [実装レポート](least_squares_intercept_implementation.md) を参照してください。
QR・診断・ridge・RLSへの拡張は、引き続き後続の設計候補です。
変更前の調査対象は `python_control_to_cpp` の `b4dde5d` と
`python_numpy_to_cpp` の `5ef4ecc` です。

## 1. 変更前のコードで起きていたこと

対象は [python_control_least_squares.hpp](../external_libraries/python_control_to_cpp/python_control/python_control_least_squares.hpp)
です。変更前のバッチ最小二乗クラスには、切片の有無を選ぶ機能がありませんでした。

| 対象 | 現在の実装 |
| --- | --- |
| `Wights_Type_` | 係数数を `X_Type::COLS + 1` に固定 |
| `LstsqSolver_Type_` | 入力行列の列数も上記の係数数に合わせる |
| `fit()` | `X` の右側へ全要素1の列を追加して解く |
| `predict()` | 同じ列を追加して予測する |
| `get_weights()` | 入力列数＋1個の係数を返す。末尾が切片 |

入力を `X ∈ R^(n×p)`、目的変数を `d ∈ R^n` とすると、解いている問題は

$$
\min_{\rho,b}\|X\rho+\mathbf{1}b-d\|^2
$$

です。一方、今回のVRFT／FRIT直接版が必要とする問題は

$$
\min_\rho\|X\rho-d\|^2
$$

です。制御器基底に含まれない定数項を追加すると、実装する制御器の係数だけを
求める問題から、別の回帰問題へ変わります。

例えば `X=[1,2,3]^T`、`d=[2,3,4]^T` の場合、切片ありでは
`rho=1, b=1`、切片なしでは `rho=20/14=10/7` です。
切片ありで推定した後に末尾の `b` を捨てても、切片なしの解にはなりません。
この例はNumPyの `lstsq` でも確認しました。

ここでいう「回帰の切片」と、制御器の比例ゲイン・伝達関数の直達項は別物です。
`FitIntercept=false` は、P項を削除する指定ではありません。

## 2. 推奨APIと互換性

### クラス・型別名・ファクトリー

次の3箇所へ、同じ既定値付きテンプレート引数を追加します。

```cpp
template <typename X_Type, bool FitIntercept = true>
class LeastSquares;

template <typename X_Type, bool FitIntercept = true>
using LeastSquares_Type = LeastSquares<X_Type, FitIntercept>;

template <typename X_Type, bool FitIntercept = true>
inline LeastSquares<X_Type, FitIntercept> make_LeastSquares() {
  return LeastSquares<X_Type, FitIntercept>();
}
```

`LeastSquares<X_Type>`、`LeastSquares_Type<X_Type>`、
`make_LeastSquares<X_Type>()` は従来どおり切片ありになります。
新しい使用側は `make_LeastSquares<X_Type, false>()` と指定します。

| 項目 | `FitIntercept=true` | `FitIntercept=false` |
| --- | --- | --- |
| 呼び出し側が渡す `X` | 生の特徴量 `n×p` | 生の特徴量 `n×p` |
| 内部の計画行列 | `[X, 1]`、`n×(p+1)` | `X`、`n×p` |
| 係数 | `[rho; b]`、`(p+1)×1` | `rho`、`p×1` |
| 予測 | `X rho + b` | `X rho` |

`PARAMETER_COUNT = X_Type::COLS + (FitIntercept ? 1 : 0)` と
`FIT_INTERCEPT` を公開定数として用意すると、呼び出し側とコード生成側で
次元やモードを確認できます。必要なら `Weights_Type` も公開します。

### 実行時フラグにしない理由

切片の有無によって行列・係数の型とサイズが変わるためです。
コンストラクタの `bool` や `set_fit_intercept(bool)` では、固定サイズ型を
実行時に変更できません。最大サイズの余剰領域を持つ方法より、テンプレートで
必要なサイズだけを確保する方が組み込み実装に適しています。

C++11には `if constexpr` がないため、通常の `if (FitIntercept)` で
異なるサイズの行列を切り替える実装は避けます。部分特殊化などで型を確定します。

通常の利用コードのソース互換性は既定値で維持できます。ただし、既存クラスに対する
独自の前方宣言・特殊化・テンプレートテンプレート引数は、引数数変更の影響確認が必要です。
ヘッダ変更後は利用側を再ビルドします。バイナリ互換性を保証する変更ではありません。
クラスのテンプレート引数数も完全に維持する必要がある場合は、別名の
`LeastSquaresWithoutIntercept<X_Type>` を追加する案を選びます。

## 3. C++11での内部変更

`fit()` と `predict()` が共通の計画行列生成処理を使う構成にします。
以下は、実装方針を示すDense行列向けの抜粋です。

```cpp
template <typename X, bool FitIntercept>
struct LeastSquaresDesign;

template <typename X>
struct LeastSquaresDesign<X, true> {
  using T = typename X::Value_Type;
  using Type = PythonNumpy::DenseMatrix_Type<T, X::ROWS, X::COLS + 1>;

  static Type make(const X &x) {
    auto ones = PythonNumpy::make_DenseMatrixOnes<T, X::ROWS, 1>();
    return PythonNumpy::concatenate_horizontally(x, ones);
  }
};

template <typename X>
struct LeastSquaresDesign<X, false> {
  using Type = X;
  static const X &make(const X &x) { return x; }
};
```

クラス内部の型・処理を次の関係に変更します。

```cpp
// LeastSquares<X_Type_In, FitIntercept> 内の抜粋
using X_Type = X_Type_In;
using T = typename X_Type::Value_Type;
using Design = LeastSquaresDesign<X_Type, FitIntercept>;
using Design_Type = typename Design::Type;
using Weights_Type = PythonNumpy::DenseMatrix_Type<T, Design_Type::COLS, 1>;
using Y_Type = PythonNumpy::DenseMatrix_Type<T, X_Type::ROWS, 1>;
using Solver_Type = PythonNumpy::LinalgLstsqSolver_Type<Design_Type, Y_Type>;

void fit(const X_Type &x, const Y_Type &y) {
  _weights = _lstsq_solver.solve(Design::make(x), y);
}

Y_Type predict(const X_Type &x) const {
  return Design::make(x) * _weights;
}
```

切片なしの `make()` は入力への参照を返すため、不要な全行列コピーを追加しません。
上記はクラス全体の置換コードではありません。既存のコンストラクタ、設定関数、
メンバー宣言も新しい型に揃え、以下を合わせて修正します。

- `_weights`、`_lstsq_solver`、`get_weights()` の型。
- 明示的に `LeastSquares<X_Type>` と書かれたコピー・ムーブの引数型。
  クラス内では `LeastSquares(const LeastSquares &input)` のようなクラス名の省略形を
  使用すれば、`FitIntercept=false` の型も正しく参照できます。
- `fit()` と `predict()` の両方。片方だけの変更は次元不整合や誤った予測を生みます。
- APIコメントの行列サイズ。実際の入力は `X: n×p`、`Y: n×1` です。
- 既存 `Wights_Type_` という綴りの型名を整理する場合、派生クラス利用も考慮し、
  必要なら旧名を新しい型への別名として残します。

この変更に標準数学関数や動的メモリは必要ありません。

## 4. 切片選択と数値解法の変更を分ける

既存 `LeastSquares` は
[LinalgLstsqSolver](../external_libraries/python_numpy_to_cpp/python_numpy/python_numpy_linalg_solver.hpp)
を呼び、そのDense行列向け処理は `gmres_k_rect_matrix` を使用しています。
`FitIntercept=false` を追加しても、自動的にQR解法へ切り替わるわけではありません。

また、現行バックエンドには「行数 > 列数」の制約があります。
切片なしでは `n > p`、切片ありでは `n > p+1` が必要です。
`n=p` や `n<p` を追加対応するには、切片の変更とは別の解法設計が必要です。

まずは切片選択だけを追加して従来バックエンドのまま検証するのが最小変更です。
現在のデータ駆動ライブラリを全面的に置き換える段階では、次の機能も必要になります。

| 機能 | 必要な追加・確認 |
| --- | --- |
| QRによる直接解法 | 既存QRを使う別クラスまたはバックエンド選択。従来解法を無断で置換しない |
| ランク不足・非有限値の通知 | `get_status()`、`get_rank()` 等を追加。現行の `fit()` の戻り値を保つことも可能 |
| ridge | 評価関数の正則化係数として明示的に実装。ソルバーの `decay_rate` と同一視しない |
| 先頭データ除外 | フィルタは全系列へ適用し、その後の有効行だけを回帰へ渡す |
| 固定メモリ | 全系列サイズのQを保持せず、小行列の逐次QR等を選べるようにする |
| 繰り返し `fit()` | 内部ソルバー状態の再利用・初期化の仕様を明確化し、前データの影響を検証する |

今回の目的関数 `||X rho-d||²/n_eff + lambda||rho||²` は、例えば
`X/sqrt(n_eff)` と `sqrt(lambda)I`、`d/sqrt(n_eff)` と零ベクトルを
縦に連結して切片なしで解けば表せます。切片ありにもridgeを導入する場合は、
切片まで正則化するかどうかを別途規定してください。

## 5. RLSと入力ヘルパーへの展開

同じファイルの `RecursiveLeastSquares` も、入力末尾へ1を追加しています。
将来オンライン調整を共通APIで扱うなら、同じ `FitIntercept=true` を追加します。

変更対象は `RLS_SIZE`、係数型、共分散行列 `P_Type_` の次元、`update()` と
`predict()` の入力拡張です。RLSの入力は `p×1` の列ベクトルなので、
バッチLSの横連結とは別に、縦連結用の特殊化を用意します。
この拡張は今回のバッチVRFT／FRITの問題解決には必須ではありません。

RLSの既存ファクトリーには `<X_Type, T>` のテンプレート引数を持つオーバーロードも
あるため、途中へbool引数を挿入して既存の明示型指定を壊さないようにします。
既存オーバーロードを残し、切片なしはクラスを直接構築するか、専用の新しい
ファクトリーを追加する方法が安全です。

`LeastSquaresInputType` と `make_LeastSquaresInput()` も独立に1を追加する
ヘルパーです。これらを使って拡張した入力を、さらに切片追加するクラスへ渡すと
二重追加になる可能性があります。クラスの入力契約は「切片を付けていない生の特徴量」
に統一します。旧ヘルパーの意味は互換性のため維持し、用途を明記します。
必要なら生入力用に別名のヘルパーを追加し、既存の返却サイズを黙って変えません。

## 6. Python側とコード生成の変更

対象は [least_squares_deploy.py](../external_libraries/python_control_to_cpp/python_control/least_squares_deploy.py)
です。現行コードは切片ありの型とファクトリーを固定で出力しています。

Python側のLSモデルへ `fit_intercept=True` 属性を追加し、`fit()`、`predict()`、
重みの次元に同じ規則を適用します。コード生成ではその値をC++のテンプレート引数に
出力します。旧Pythonオブジェクトに属性がない場合は、従来どおり `true` とします。

```cpp
// fit_intercept=False のモデルから生成する例
using type = PythonControl::LeastSquares_Type<X_Type, false>;
inline type make() {
  return PythonControl::make_LeastSquares<X_Type, false>();
}
```

`X_SIZE` は切片を含まない元の特徴量数とします。RLS生成コードには現在
`state_size = rls.RLS_size - 1` があるため、RLSも拡張する場合は
`RLS_size - int(fit_intercept)`、または明示的な元の特徴量数へ変更します。
「全てのモデルから無条件に1を引く」処理は残さないようにします。

SIL側にも次元の固定があります。`test_sil/least_squares/least_squares_SIL.cpp` の
`WEIGHTS_SIZE = X_SIZE + 1` は、新しいクラスの `PARAMETER_COUNT`、または
`Weights_Type::ROWS` から取得する形に変更します。RLSを拡張する場合は
`test_sil/recursive_least_squares/recursive_least_squares_SIL.cpp` の同じ固定式も
変更します。返却配列のサイズとコピー範囲の両方を、実際の係数数に揃える必要があります。

切片選択は、調整済み重みのC++への保存機能とは別の変更です。
現行のLS生成処理は主に型と初期インスタンスの生成を行うため、
`fit_intercept` を出力するだけで学習済み係数も移植されるとは扱いません。

## 7. VRFT／FRITでの使用

追加後は、次の回帰行列と目的変数に対して `LeastSquares<X_Type, false>` を使います。

| 手法 | 回帰列 `X_i` | 目的変数 `d` | 返却係数の意味 |
| --- | --- | --- | --- |
| VRFT直接版（プレフィルタM） | `beta_i (I-M)y` | `Mu` | 順方向制御器の係数 |
| FRIT直接版 | `M beta_i u` | `(I-M)y` | 逆制御器の係数 |

この対応により切片の問題は解消できます。ただし、FRIT直接版で逆制御器が
パラメータに線形でなければならない条件は変わりません。
通常のPI/PIDゲインを調整するFRIT反復版が、この追加だけで直接LSへ変わることもありません。

現行 `data_driven_least_squares.hpp` の置換は、切片なしAPIの実装に加え、
ランク判定・ridge・有効区間・数値精度・メモリ使用量が同等であることを確認した後に行います。

## 8. 変更ファイルと検証項目

| ファイル・領域 | 変更内容 |
| --- | --- |
| `python_control/python_control_least_squares.hpp` | 必須：LSクラス、型、fit/predict、コピー/ムーブ、alias、factory |
| `python_control/least_squares_deploy.py` | 切片の設定を生成コードへ反映 |
| Python制御ライブラリの対応LSモデル | `fit_intercept` 属性、同じ回帰式、重み次元 |
| `test_vs/test_vs_main.cpp` 等 | 従来互換性と切片なしの検証 |
| `test_sil/least_squares/least_squares_SIL.cpp` | 係数数の固定 `X_SIZE + 1` を型から取得する形に変更 |
| サンプル・利用文書 | 入力契約、係数の並び、生成手順を記載 |
| RLS関連 | 任意の後続対応。バッチLSの最小変更とは分離 |

検証は `float` と `double`、GCCのC++11モードで行います。

1. 引数省略と `FitIntercept=true` で、既存の重み・予測・コピー／ムーブのテストが通る。
2. `FitIntercept=false` の重みがp個で、計画行列に1の列を追加しない。
3. `X=[1,2,3]^T, d=[2,3,4]^T` で、切片なしは `10/7`、切片ありは `[1,1]` を返す。
4. 切片なしの零入力に対する予測が0になり、fitとpredictの仕様が一致する。
5. P・PI・PID相当の1～3変数で、既知解とNumPyの切片なし `lstsq` に一致する。
6. 現行バックエンドを残す段階では行数制約を明示する。解法拡張後は正方・劣決定・
   ランク不足・NaN/Inf・ridgeの試験も追加する。
7. 切片あり／なしのPythonモデルから生成したC++がコンパイルされ、重みの次元と予測が一致する。
8. VRFT／FRITの既存SILが通り、切片を捨てる回避策に依存していない。

実装順序は、LSの切片選択と互換性検証、Python／コード生成の対応、
QR・診断・正則化などの追加、データ駆動側の移行の順を推奨します。
