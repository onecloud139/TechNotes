# 随机微分方程（Stochastic Differential Equations, SDE）

## 一、基础知识

### 1. 布朗运动  $W_t$ 的核心性质

$W_t$ 看成“随时间变化的随机函数”，但它不是普通可导函数。标准布朗运动满足：

1. $W_0=0$
2. 独立增量：不重叠时间区间上的增量相互独立
3. 正态增量：

$$
W_t-W_s\sim\mathcal N(0,t-s),\quad t>s
$$

4. 平稳增量：增量分布只与时间差有关
5. 路径连续但几乎处处不可导
6. 一二阶矩：

$$
\mathbb E[W_t]=0,\qquad \mathrm{Var}(W_t)=t
$$

7. 二次变差规则（SDE 计算最重要）：

$$
(dW_t)^2=dt,\qquad dW_t\,dt=0,\qquad (dt)^2=0
$$

8. 数值模拟增量：

$$
\Delta W_n\sim\mathcal N(0,\Delta t),\qquad \Delta W_n=\sqrt{\Delta t}\,\xi_n,
\ \xi_n\sim\mathcal N(0,1)
$$

### 2. SDE 的标准形式

$$
dX_t=b(X_t,t)dt+\sigma(X_t,t)dW_t
$$

- $b$：漂移项（确定性趋势）
- $\sigma$：扩散项（随机波动强度）

### 3. Itô 公式（SDE 版链式法则）

若上式成立，则对光滑函数   $f(x,t)$：

$$
df(X_t,t)=\left(f_t+b f_x+\frac12\sigma^2 f_{xx}\right)dt+\sigma f_x dW_t
$$

和 ODE 的链式法则相比，多出的项

$$
\frac12\sigma^2 f_{xx}dt
$$

本质来自   $(dW_t)^2=dt$。

## 二、路径视角 vs 密度视角（Fokker-Planck）

### 1. 两个视角在看同一个系统

- **路径视角（sample path view）**：关注“单条样本轨迹怎么走”。工具是 SDE。  
  例：给一个具体随机种子，轨迹会抖动前进。

- **密度视角（distribution view）**：关注“很多样本组成的概率云怎么变”。工具是 Fokker-Planck PDE。

所以两者关系是：
- SDE 管“微观个体轨迹”；
- Fokker-Planck 管“宏观分布演化”。

### 2. Fokker-Planck 方程（一维）

$$
dX_t=b(X_t,t)dt+\sigma(X_t,t)dW_t
$$

若   $X_t$ 有密度   $p(x,t)$，则

$$
\partial_t p(x,t)
=-\partial_x\big(b(x,t)p(x,t)\big)
+\frac12\partial_{xx}\big(\sigma^2(x,t)p(x,t)\big)
$$

可以分块读：

- 第一项   $-\partial_x(bp)$：漂移导致“搬运/平移”
- 第二项   $\frac12\partial_{xx}(\sigma^2 p)$：扩散导致“摊开/平滑”

### 3. 用“守恒律”理解这条 PDE

写成连续性方程形式：

$$
\partial_t p + \partial_x J = 0
$$

其中概率流（probability flux）

$$
J(x,t)=b(x,t)p(x,t)-\frac12\partial_x\big(\sigma^2(x,t)p(x,t)\big)
$$

解释：
- 某点密度升高，等价于有净概率流流入；
- 某点密度降低，等价于有净概率流流出。

### 4. 为什么扩散项是二阶导

你可以把扩散想成“局部平均化效应”。二阶导在 PDE 里恰好描述“曲率驱动的平滑”。

- 若某处密度尖峰很陡，二阶项会推动其被抹平；
- 噪声越强（  $\sigma$ 越大），抹平越快。

### 5. 纯扩散

令   $b=0,\sigma=\sigma_0$ 常数：

$$
\partial_t p = \frac{\sigma_0^2}{2}\partial_{xx}p
$$

这就是热方程。初始若是集中在一点，随着时间会变成越来越宽的高斯。

### 6. 线性漂移 + 常扩散（OU 对应）

若   $b(x)=\theta(\mu-x),\ \sigma$ 常数，Fokker-Planck 会同时做两件事：
- 漂移把密度往   $\mu$ 拉；
- 扩散把密度摊开。

最终在“拉回”和“摊开”平衡下形成稳态高斯分布。

### 7. 多维形式

若  $X_t\in\mathbb R^d$，$a(x,t)=\Sigma\Sigma^\top$：

$$
\partial_t p
=-\nabla\cdot\big(b p\big)
+\frac12\sum_{i,j}\partial_{x_i x_j}\big(a_{ij}p\big)
$$

## 三、常见模型与常见数值方法

### 1. 常见模型

#### 1.1 几何布朗运动（GBM）

$$
dS_t=\mu S_tdt+\sigma S_t dW_t
$$

$$
S_t=S_0\exp\left[\left(\mu-\frac12\sigma^2\right)t+\sigma W_t\right]
$$

用途：金融价格基础模型，且 $S_t>0$。

#### 1.2 Ornstein-Uhlenbeck（OU）过程

$$
dX_t=\theta(\mu-X_t)dt+\sigma dW_t
$$

$$
\mathbb E[X_t]=\mu+(X_0-\mu)e^{-\theta t}
$$

$$
\mathrm{Var}(X_t)=\frac{\sigma^2}{2\theta}(1-e^{-2\theta t})
$$

长期稳态：

$$
\mathcal N\left(\mu,\frac{\sigma^2}{2\theta}\right)
$$

#### 1.3 CIR

$$
dX_t=\kappa(\theta-X_t)dt+\sigma\sqrt{X_t}dW_t
$$

用途：利率/方差；优点是更容易保持非负。

### 2. 常见数值方法

#### 2.1 Euler-Maruyama（首选基线）

$$
X_{n+1}=X_n+b(X_n,t_n)\Delta t+\sigma(X_n,t_n)\Delta W_n,
\quad \Delta W_n\sim\mathcal N(0,\Delta t)
$$

实操关键：噪声一定要按 $\mathcal N(0,\Delta t)$ 采样。

#### 2.2 Milstein（一维噪声更精确）

$$
X_{n+1}=X_n+b_n\Delta t+\sigma_n\Delta W_n
+\frac12\sigma_n\sigma_n'\big((\Delta W_n)^2-\Delta t\big)
$$

优点：路径精度更高；缺点：实现更复杂。

#### 2.3 强收敛 vs 弱收敛

- 强收敛：单条路径误差小（路径级）
- 弱收敛：统计量误差小（分布级）

如果关心分布与期望，弱收敛指标通常更实用。


## 四、其他扩展

### 1. Itô 与 Stratonovich

Stratonovich 形式：

$$
dX_t=b^{\circ}(X_t,t)dt+\sigma(X_t,t)\circ dW_t
$$

1维转换为 Itô：

$$
b^{Ito}=b^{\circ}+\frac12\sigma\partial_x\sigma
$$

Itô 常用于概率推导和机器学习实现；Stratonovich 在物理建模中常见。

### 2. 与扩散模型（Diffusion）联系

VP-SDE：

$$
dx=-\frac12\beta(t)xdt+\sqrt{\beta(t)}dW_t
$$

反向时间 SDE：

$$
dx=\left[-\frac12\beta(t)x-\beta(t)\nabla_x\log p_t(x)\right]dt+\sqrt{\beta(t)}d\bar W_t
$$

神经网络学习 score $\nabla_x\log p_t(x)$（或等价噪声预测）即可生成采样。


## 速记

$$
dX_t=b\,dt+\sigma\,dW_t
$$

$$
(dW_t)^2=dt
$$

$$
df=(f_t+b f_x+\tfrac12\sigma^2f_{xx})dt+\sigma f_x dW_t
$$

$$
\partial_t p=-\partial_x(bp)+\tfrac12\partial_{xx}(\sigma^2p)
$$

$$
X_{n+1}=X_n+b_n\Delta t+\sigma_n\Delta W_n
$$
