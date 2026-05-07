# 基础知识点

## 机器学习范式

机器学习根据训练数据中标签的可用性可分为三种主要范式：有监督学习、无监督学习和半监督学习。

### 有监督学习（Supervised Learning）

有监督学习从带有标签的训练数据中学习输入到输出的映射关系。[人话：用x拟合y]

-   **基本概念**

    -   训练数据：  $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$   ，包含输入-输出对

    -   目标：学习映射函数  $f: \mathcal{X} \rightarrow \mathcal{Y}$ 

    -   损失函数： $\mathcal{L} = \frac{1}{N} \sum_{i=1}^N \ell(f(\mathbf{x}_i), y_i)$ 

-   **任务类型**

    -   分类任务：输出离散类别（如垃圾邮件检测、图像分类）

    -   回归任务：输出连续数值（如房价预测、温度预报）

-   **常用算法**

    -   逻辑回归、支持向量机、决策树

    -   神经网络、朴素贝叶斯

### 无监督学习（Unsupervised Learning）

无监督学习从无标签数据中发现内在结构和模式。[人话：用x区分样本，没有y]

-   **基本概念**

    -   训练数据：  $\mathcal{D} = \{\mathbf{x}_i\}_{i=1}^N$  ，仅含输入特征

    -   目标：发现数据中的隐藏结构和模式

-   **主要方法**

    -   聚类分析：将相似数据分组（如K-means、DBSCAN）

    -   降维：减少特征维度（如PCA、t-SNE）

    -   关联规则：发现项集间关系（如Apriori算法）

    -   前两项最为常用

### 半监督学习（Semi-supervised Learning）

半监督学习同时利用少量标注数据和大量未标注数据。[人话：有一部分有y,有一部分没y，你要先最大化地把没y的标上y]

-   **基本概念**

    -   训练数据： $\mathcal{D} = \mathcal{D}_L \cup \mathcal{D}_U$ 

    -   其中   $\mathcal{D}_L = \\{(\mathbf{x}_i, y_i)\\}_{i=1}^L$  为标注数据

    -   其中  $\mathcal{D}_U = \\{\mathbf{x}_j\\}_{j=1}^U$  为未标注数据，通常  $U \gg L$  

-   **核心假设**

    -   平滑假设：相近样本具有相似标签

    -   聚类假设：数据形成离散簇结构

    -   流形假设：数据位于低维流形

-   **常用方法**

    -   自训练：用高置信度预测扩展训练集

    -   协同训练：多视角相互学习

    -   图方法：基于图结构的标签传播

    -   深度半监督：结合一致性正则化
## 损失函数 (Loss Function)

损失函数用于衡量模型预测值 $\hat{y}$ 与真实值 $y$ 之间的差异，是模型优化的目标。

- 均方误差 (MSE - Mean Squared Error)常用于回归问题，对离群点敏感。

$$J(\mathbf{w}) = \frac{1}{m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2$$

- 交叉熵损失 (Cross-Entropy Loss)又称对数损失，是分类问题中最常用的损失函数，尤其适用于逻辑回归和神经网络。它衡量两个概率分布之间的差异。其中  $K$  是类别总数。

$$J(\mathbf{w}) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

$$J(\mathbf{w}) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{j=1}^{K} y_j^{(i)} \log(\hat{y}_j^{(i)})$$

- KL散度损失 (Kullback-Leibler Divergence Loss)又称相对熵，用于衡量真实概率分布与模型预测概率分布之间的信息差异，刻画用预测分布近似真实分布的信息损耗，具备非负、非对称特性，常应用于分布对齐、知识蒸馏等场景。  $P$  代表真实概率分布，  $\hat{P}$  代表模型预测概率分布，  $m$  为样本数，  $K$  为分布维度/类别数。

$$D_{KL}(P || \hat{P}) = \sum_{i=1}^{n} P(i) \log\left( \frac{P(i)}{\hat{P}(i)} \right)$$

$$J(\mathbf{w}) = \frac{1}{m} \sum_{i=1}^{m} D_{KL}(P^{(i)} || \hat{P}^{(i)}) = \frac{1}{m} \sum_{i=1}^{m} \sum_{j=1}^{K} P_j^{(i)} \log\left( \frac{P_j^{(i)}}{\hat{P}_j^{(i)}} \right)$$


## 正则化 (Regularization)

正则化是一种通过向损失函数添加惩罚项来防止模型过拟合的技术，旨在降低模型复杂度，鼓励参数值变小。[注意，这里的  $\mathbf{w}$  可以通俗地理解为ols里的  $\beta$  系数]

-   **L0 正则化**\
    惩罚模型中非零权重的数量（即直接限制参数的数量）。\
    公式：  $J(\mathbf{w}) = \text{Original Loss} + \lambda \|\mathbf{w}\|_0$  ，其中  $\|\mathbf{w}\|_0$  是  $\mathbf{w}$  中非零元素的个数。\
    **缺点：** 优化是一个NP难问题，难以求解。

-   **L1 正则化 (Lasso Regression)**\
    惩罚权重的绝对值之和。倾向于产生稀疏权重矩阵，即产生少量特征权重不为零，其他都为零，因此可用于特征选择。

$$J(\mathbf{w}) = \text{Original Loss} + \lambda \|\mathbf{w}\|_1 = \text{Original Loss} + \lambda \sum_{j=1}^{n} |w_j|$$

-   **L2 正则化 (Ridge Regression)**\
    惩罚权重的平方和。使得所有权重值均向零缩小，但通常不会等于零。

$$J(\mathbf{w}) = \text{Original Loss} + \lambda \|\mathbf{w}\|_2^2 = \text{Original Loss} + \lambda \sum_{j=1}^{n} w_j^2$$

-   在逻辑回归中，添加L2正则化后的损失函数为：

$$J(\mathbf{w}) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(h_{\mathbf{w}}(\mathbf{x}^{(i)})) + ...] + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2$$

## One-Hot 编码 (One-Hot Encoding)

一种将分类变量（通常是离散的、无序的）转换为机器学习算法更易理解的形式的方法。[实际上就是离散情况下为了计数的单位向量]

-   **动机：**
    许多算法无法直接处理类别标签（如"猫"、"狗"、"鸟"），需要将其转换为数值形式。简单的整数编码（如猫=1,
    狗=2, 鸟=3）会引入错误的顺序关系，One-Hot编码可以解决这个问题。

-   **方法：** 对于一个有  $K$  个不同类别的变量，用一个长度为  $K$  的二进制向量来表示。向量中只有一位为1（对应样本所属的类别），其余位均为0。

-   **示例：** 假设特征"颜色"有3个类别：红、绿、蓝。 

$$\begin{aligned}
            \text{红} &\rightarrow [1, 0, 0] \\
            \text{绿} &\rightarrow [0, 1, 0] \\
            \text{蓝} &\rightarrow [0, 0, 1] \\
\end{aligned}$$

-   **用途：**
    广泛用于处理分类特征和分类任务的标签（如在多分类问题中表示真实标签  $y$  ）。

## 学习率 (Learning Rate)

学习率是梯度下降算法中最重要的超参数之一，它控制着每次参数更新的步长大小。

-   **定义：** 学习率  $\alpha$  
    决定了在梯度方向上移动的幅度，即参数更新的速度。

-   **影响：**

    -   学习率过大：可能导致参数在最优解附近震荡，甚至发散（无法收敛）

    -   学习率过小：收敛速度缓慢，训练时间过长，可能陷入局部极小值

-   **学习率调整策略：**

    -   固定学习率：整个训练过程使用恒定学习率

    -   学习率衰减：随着训练进行逐渐减小学习率，常见衰减方式：

        -   指数衰减：  $\alpha = \alpha_0 \cdot e^{-kt}$  

        -   时间衰减：  $\alpha = \frac{\alpha_0}{1 + kt}$  

        -   阶梯衰减：每隔固定epoch将学习率减半

-   **选择原则：** 通常需要通过实验确定最佳学习率，常用范围在  $10^{-6}$  到  $1$  之间

## 优化器 (Optimizers)

### SGD (Stochastic Gradient Descent) 随机梯度下降

最基础的优化算法，每次迭代使用**一个样本或一小批样本(mini-batch)**的梯度来更新参数。

-   **核心思想：** 在负梯度方向上更新参数，以最小化损失函数。

-   **算法步骤：** 对于每个参数  $w$  ：

    $$w_{t+1} = w_t - \alpha \cdot \nabla_w J(w_t; x^{(i)}, y^{(i)})$$

    其中  $(x^{(i)}, y^{(i)})$  是第  $i$  个训练样本。

-   **Mini-batch SGD（小批量梯度下降）：**
    
    实际应用中通常使用小批量样本计算梯度，平衡了计算效率和梯度稳定性：
    
    $$w_{t+1} = w_t - \alpha \cdot \frac{1}{m} \sum_{i=1}^m \nabla_w J(w_t; x^{(i)}, y^{(i)})$$
    
    其中  $m$  是 batch size。

-   **超参数：**
    
    -   $\alpha$  ：学习率（通常范围：  $0.01 \sim 0.1$  ）

-   **优点：** 计算简单，内存小；对凸函数收敛到全局最优；随机噪声助逃浅层局部最优
-   **缺点：** 学习率固定，难适应复杂损失曲面；梯度稀疏/剧变方向收敛困难（峡谷、鞍点）；最优解附近易震荡

### Momentum (动量法)

模拟物理学中的动量概念，引入**速度(velocity)**累积之前的梯度，帮助加速收敛并减少震荡。

-   **核心思想：** 不仅考虑当前梯度，还累积历史梯度的指数加权平均，使得更新方向更加稳定。

-   **算法步骤：**

    1.  计算速度（动量累积）：

        $$v_t = \gamma v_{t-1} + \alpha \cdot \nabla_w J(w_t)$$

    2.  更新参数：

        $$w_{t+1} = w_t - v_t$$

    3.  或者等价写作：

        $$w_{t+1} = w_t - \gamma v_{t-1} - \alpha \nabla_w J(w_t)$$

-   **超参数：**

    -   $\alpha$  ：学习率
    
    -   $\gamma$  ：动量系数（通常设为  $0.9$  ，范围  $0 \sim 1$  ）
        
        -   $\gamma = 0$  ：退化为普通 SGD
        
        -   $\gamma$  越大，历史梯度影响越大，更新越平滑

-   **优点：** 梯度一致时加速收敛；梯度剧变时抑制震荡；动量助冲狭窄局部最优和鞍点

-   **直观理解：** 想象小球滚下山坡，动量使得小球具有惯性，能够滚过平坦区域和小坑洼，而不是每一步都根据当前坡度立即停下。


### Nesterov Accelerated Gradient (NAG) Nesterov加速梯度

对 Momentum 的改进，在**预估的未来位置**计算梯度，具有更强的"前瞻性"。

-   **核心思想：** 先按照累积的动量方向"向前看"一步，然后在这个预估位置计算梯度进行修正，实现更精准的更新。

-   **算法步骤：**

    1.  计算预估位置（先根据动量前进）：

        $$ \tilde{w}_t = w_t - \gamma v_{t-1} $$

    2.  在预估位置计算梯度并更新速度：

        $$v_t = \gamma v_{t-1} + \alpha \cdot \nabla_w J(\tilde{w}_t)$$

    3.  更新参数：

        $$w_{t+1} = w_t - v_t$$

    4.  或等价合并为：

        $$w_{t+1} = w_t - \gamma v_{t-1} - \alpha \nabla_w J(w_t - \gamma v_{t-1})$$

-   **与 Momentum 的区别：**
    
    Momentum 在**当前位置**计算梯度然后前进；NAG 先根据动量前进到**预估位置**，在那里计算梯度，如果预估过头了就提前减速修正。

-   **超参数：**

    -   $\alpha$  ：学习率
    
    -   $\gamma$  ：动量系数（通常设为  $0.9$  ）

-   **优点：** 对Momentum的"超前"修正，收敛更稳定；RNN等任务中通常优于标准Momentum；强凸函数上理论收敛速率更好


### AdaGrad (Adaptive Gradient) 自适应梯度

为每个参数**独立地调整学习率**，对于出现频繁的参数给予较小学习率，对于稀疏的参数给予较大学习率。

-   **核心思想：** 累积梯度的平方和作为自适应学习率的分母，使得更新幅度自动适应参数的历史梯度大小。

-   **算法步骤：**

    1.  累积梯度平方：

        $$G_t = G_{t-1} + g_t^2 = \sum_{\tau=1}^t g_\tau^2$$
        
        其中  $g_t = \nabla_w J(w_t)$，  $G_t$  是历史梯度平方的累加（对角矩阵，每个元素对应一个参数）

    2.  自适应更新：

        $$w_{t+1} = w_t - \frac{\alpha}{\sqrt{G_t + \epsilon}} \odot g_t$$
        
        其中  $\odot$  表示逐元素相乘，  $\epsilon$  是避免除零的小常数（  $10^{-8}$  ）

-   **超参数：**

    -   $\alpha$  ：全局学习率（通常设为  $0.01$  ）
    
    -   $\epsilon$  ：数值稳定项（  $10^{-8}$  ）

-   **优点：** 无需手动调整学习率，自动适应每个参数；特别适合稀疏梯度问题（如NLP词嵌入），对罕见词更新大
-   **缺点：** 累积梯度平方单调递增 → 学习率持续衰减，最终可能提前停止学习

### RMSProp (Root Mean Square Propagation)

解决 AdaGrad 学习率单调递减的问题，通过**指数加权平均**代替累积平方和，使得历史梯度逐渐遗忘。

-   **核心思想：** 使用移动平均（EMA）计算二阶矩，只关注近期的梯度信息，避免学习率过早衰减。

-   **算法步骤：**

    1.  计算梯度平方的指数加权平均：

        $$v_t = \beta v_{t-1} + (1-\beta) g_t^2$$
        
        其中  $g_t^2$  表示梯度各元素的平方，  $\beta$  是衰减率（通常  $0.9$  ）

    2.  自适应更新：

        $$w_{t+1} = w_t - \frac{\alpha}{\sqrt{v_t + \epsilon}} \odot g_t$$

-   **与 AdaGrad 的区别：**
    
    -   AdaGrad：  $G_t = \sum_{\tau=1}^t g_\tau^2$  （累积和，只增不减）
    
    -   RMSProp：  $v_t = \beta v_{t-1} + (1-\beta) g_t^2$  （指数平均，可自适应变化）

-   **超参数：**

    -   $\alpha$  ：学习率（通常设为  $0.001$  ）
    
    -   $\beta$  ：二阶矩衰减率（通常设为  $0.9$  ）
    
    -   $\epsilon$  ：数值稳定项（  $10^{-8}$  ）

-   **优点：** 解决了AdaGrad学习率过早消失的问题；非平稳目标（如RNN）上表现很好；适合非凸优化

### Adadelta

RMSProp 的进一步改进，**完全消除了对全局学习率的依赖**，使用梯度平方的移动平均来动态调整学习单位。

-   **核心思想：** 用参数更新量的平方的指数平均来代替全局学习率  $\alpha$  ，使得更新单位与参数单位一致。

-   **算法步骤：**

    1.  计算梯度平方的指数平均（同 RMSProp）：

        $$ E[g^2]_t = \gamma E[g^2]_{t-1} + (1-\gamma) g_t^2 $$

    2.  计算参数更新量（RMS表示均方根）：

        $$\Delta w_t = - \frac{\sqrt{E[\Delta w^2]_{t-1} + \epsilon}}{\sqrt{E[g^2]_t + \epsilon}} g_t$$
        
        其中  $E[\Delta w^2]_{t-1}$  是前一步参数更新量平方的指数平均

    3.  累积更新量的平方：

        $$ E[\Delta w^2]_t = \gamma E[\Delta w^2]_{t-1} + (1-\gamma) \Delta w_t^2 $$

    4.  更新参数：

        $$ w_{t+1} = w_t + \Delta w_t $$

-   **超参数：**

    -   $\gamma$  ：衰减率（通常设为  $0.9$  ，类似 RMSProp 的  $\beta$  ）
    
    -   $\epsilon$  ：数值稳定项（  $10^{-6} \sim 10^{-8}$  ）
    
    -   **无需设置全局学习率  $\alpha$  **

-   **优点：** 无需手动调整学习率，完全自适应；对学习率超参数不敏感，训练稳定

### Adam (Adaptive Moment Estimation) 自适应矩估计

**目前深度学习中最常用的优化器**，结合 Momentum 和 RMSProp 的优点，同时考虑一阶矩（均值）和二阶矩（方差）。

-   **核心思想：**

    -   计算梯度的一阶矩估计（均值） $m_t$  作为动量项

    -   计算梯度的二阶矩估计（未中心化的方差）  $v_t$  作为自适应学习率项

    -   对这两个矩估计进行偏差校正（解决初始阶段冷启动问题）

-   **算法步骤：** 对于每个参数  $w$  和每个时间步  $t$  ：

    1.  计算梯度：  $g_t = \nabla_w J(w_t)$  

    2.  更新一阶矩估计（动量）：

        $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

    3.  更新二阶矩估计（自适应）：

        $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

    4.  计算偏差校正的一阶矩（解决初始偏差）：

        $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

    5.  计算偏差校正的二阶矩：

        $$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

    6.  更新参数：

        $$w_t = w_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

-   **超参数说明：**

    -   $\alpha$  ：学习率（通常设为  $0.001$  ）

    -   $\beta_1$  ：一阶矩衰减率（通常设为  $0.9$  ）

    -   $\beta_2$  ：二阶矩衰减率（通常设为  $0.999$  ）

    -   $\epsilon$  ：数值稳定项（通常设为  $10^{-8}$  ）

-   **优点：** 自适应学习率：每个参数独立调整；收敛快，适合大规模数据/参数；内存高效（每参数仅2个额外状态）；默认超参数在多数场景表现良好
-   **缺点：** 泛化能力可能不如精心调参的SGD+Momentum；某些情况收敛到次优解（催生了AdamW等改进）

### AdamW (Adam with Weight Decay)

Adam 的改进版本，**解耦了权重衰减(Weight Decay)和梯度更新**，解决了 Adam 中 L2 正则化效果不佳的问题。

-   **核心问题：** 标准 Adam 中，L2 正则化（权重衰减）与自适应学习率耦合，导致大梯度参数的正则化效果被削弱。

-   **算法区别：**

    **标准 Adam + L2 正则：**

    $$w_t = w_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \alpha \lambda w_{t-1}$$

    **AdamW（解耦权重衰减）：**

    $$w_t = w_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \lambda w_{t-1}$$

-   **算法步骤（差异仅在第6步）：**

    1-5步同 Adam（计算  $\hat{m}_t$  和  $\hat{v}_t$  ）,参数更新（解耦权重衰减）：

    $$w_t = w_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \lambda w_{t-1}$$

-   **超参数：**

    -   $\alpha, \beta_1, \beta_2, \epsilon$  ：同 Adam
    
    -   $\lambda$  ：权重衰减系数（通常设为  $0.01$  或  $0.001$  ，比 L2 正则系数稍大）

-   **优点：** 修正了Adam的正则化问题，泛化通常优于Adam；Transformer/BERT/GPT等现代架构的事实标准；大模型预训练更稳定

### Nadam (Nesterov-accelerated Adaptive Moment Estimation)

结合 **NAG (Nesterov)** 和 **Adam** 的优点，在 Adam 的基础上引入 Nesterov 动量，实现更精准的梯度估计。

-   **核心思想：** 将 Nesterov 的前瞻性思想融入 Adam 的一阶矩估计中。

-   **算法改进：**

    Adam 的标准更新项：  $\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$  
    
    Nadam 改为：    $\frac{\beta_1 \hat{m}_t + (1-\beta_1) g_t}{\sqrt{\hat{v}_t} + \epsilon}$
    
    等价于在当前梯度中混入 Nesterov 风格的动量修正。

-   **优点：** 某些任务上收敛速度优于Adam；结合了Nesterov的预见性和Adam的自适应性

## 评估体系（Evaluation System）

评估体系的目标是统一衡量模型效果、避免“只看一个分数”的误判，并确保离线结果能够更可靠地迁移到线上。

### 评估设计原则

-   **任务对齐：** 指标要与业务损失对应，例如漏报成本高时优先提升 Recall。
-   **数据划分规范：** 训练/验证/测试分离，测试集仅用于最终一次报告。
-   **多指标联合：** 主指标 + 稳定性指标 + 资源指标（延迟、内存、吞吐）联合判断。
-   **统计稳健性：** 报告均值与标准差，必要时提供置信区间或显著性检验。

### 分类任务评估（Binary / Multi-class）

-   **混淆矩阵四要素：**
    -   TP：预测正且真实正
    -   FP：预测正但真实负
    -   TN：预测负且真实负
    -   FN：预测负但真实正

-   **常见指标与计算：**

$$
\text{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}
$$

$$
\text{Precision}=\frac{TP}{TP+FP},\qquad
\text{Recall}=\frac{TP}{TP+FN}
$$

$$
\text{Specificity}=\frac{TN}{TN+FP},\qquad
FPR=\frac{FP}{FP+TN}
$$

$$
F1=\frac{2\cdot \text{Precision}\cdot \text{Recall}}{\text{Precision}+\text{Recall}}
$$

-   **何时用什么：**
    -   类别均衡且代价对称时可看 Accuracy
    -   正例稀少或误报代价高时看 Precision / PR-AUC
    -   漏报代价高时看 Recall
    -   Precision 与 Recall 需折中时看 F1 或 $F_\beta$

$$
F_\beta=(1+\beta^2)\frac{\text{Precision}\cdot \text{Recall}}{\beta^2\cdot \text{Precision}+\text{Recall}}
$$

-   **阈值无关评估：**
    -   ROC 曲线由  $(FPR,TPR)$ 构成，其中  $TPR=\frac{TP}{TP+FN}$，AUC 是曲线下面积
    -   PR 曲线由  $(Recall,Precision)$ 构成，不平衡数据通常更有判别力

-   **多分类扩展：**
    -   Micro-F1：先汇总全类别 TP/FP/FN 再计算，偏向大类
    -   Macro-F1：先算每类 F1 再平均，能反映小类表现
    -   Weighted-F1：按类别样本占比加权

### 概率质量与校准

分类模型不仅要“分对类”，还要“概率可信”。

-   **LogLoss（交叉熵）：**

$$
\text{LogLoss}=-\frac{1}{n}\sum_{i=1}^n\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]
$$

-   **Brier Score（二分类）：**

$$
\text{Brier}=\frac{1}{n}\sum_{i=1}^n (p_i-y_i)^2
$$

-   **ECE（Expected Calibration Error）：**
  将预测概率分桶，记第 $m$ 桶样本集合为 $B_m$，则

$$
ECE=\sum_{m=1}^M \frac{|B_m|}{n}\left|\text{acc}(B_m)-\text{conf}(B_m)\right|
$$

其中  $\text{acc}(B_m)$ 为该桶真实准确率，  $\text{conf}(B_m)$ 为该桶平均预测概率。

### 回归任务评估

-   **绝对误差与平方误差：**

$$
MAE=\frac{1}{n}\sum_{i=1}^n|y_i-\hat y_i|
$$

$$
MSE=\frac{1}{n}\sum_{i=1}^n(y_i-\hat y_i)^2,\qquad
RMSE=\sqrt{MSE}
$$

-   **拟合优度：**

$$
R^2=1-\frac{\sum_i (y_i-\hat y_i)^2}{\sum_i (y_i-\bar y)^2}
$$

-   **相对误差（需注意分母接近 0）：**

$$
MAPE=\frac{100\%}{n}\sum_{i=1}^n\left|\frac{y_i-\hat y_i}{y_i}\right|
$$

-   **分位数损失（Pinball Loss）：** 用于分位数回归，预测   $\tau$ 分位点   $\hat y_i^{(\tau)}$

$$
\ell_\tau(y,\hat y)=
\begin{cases}
\tau(y-\hat y), & y\ge \hat y\\
(\tau-1)(y-\hat y), & y<\hat y
\end{cases}
$$

### 聚类任务评估

-   **Silhouette Coefficient（单样本）：**
  设   $a(i)$ 为样本   $i$ 到同簇样本平均距离，  $b(i)$ 为到最近其他簇样本平均距离

$$
s(i)=\frac{b(i)-a(i)}{\max(a(i),b(i))}\in[-1,1]
$$

整体分数为 $\frac{1}{n}\sum_i s(i)$，越大越好。

-   **Davies-Bouldin Index（DBI）：**
  设簇内散度为   $S_i$，簇中心距离为   $M_{ij}$

$$
DBI=\frac{1}{K}\sum_{i=1}^K \max_{j\ne i}\frac{S_i+S_j}{M_{ij}}
$$

越小越好。

-   **ARI（Adjusted Rand Index）：**
  基于样本对一致性，并对随机分群做校正，取值一般在   $[-1,1]$，越大越好。

-   **NMI（Normalized Mutual Information）：**

$$
NMI(U,V)=\frac{2I(U;V)}{H(U)+H(V)}
$$

其中   $U,V$ 为两种聚类划分，值越接近 1 一致性越高。

### 生成模型评估（图像为例）

-   **IS（Inception Score）：**

$$
IS=\exp\left(\mathbb E_x D_{KL}(p(y|x)\|p(y))\right)
$$

希望单样本分类分布尖锐（可辨识）且总体类别分布多样（覆盖广）。

-   **FID（Fr\'echet Inception Distance）：**
  设真实与生成特征分布分别为高斯 $\mathcal N(m_r,C_r)$ 与 $\mathcal N(m_g,C_g)$

$$
FID=\|m_r-m_g\|_2^2 + Tr\left(C_r+C_g-2(C_rC_g)^{1/2}\right)
$$

越小越好，通常对视觉质量更敏感。

## 特征工程与可解释性

特征工程决定模型可学到的信息上限，可解释性决定模型是否可被信任、诊断和迭代。

### 特征工程流程

1.  数据理解：字段类型、业务含义、采样方式、标签定义。
2.  数据清洗：缺失、异常、重复、时间对齐、单位统一。
3.  特征变换：缩放、分箱、编码、分布变换。
4.  特征构造：交叉、聚合、窗口统计、领域规则特征。
5.  特征筛选：相关性/冗余控制、稳定性筛选。
6.  线上一致性：训练与推理特征口径一致（防 train-serving skew）。

### 数值特征处理（怎么算）

-   **标准化（Z-score）：**

$$
x'=\frac{x-\mu}{\sigma}
$$

适合线性模型、SVM、神经网络。

-   **Min-Max 归一化：**

$$
x'=\frac{x-x_{min}}{x_{max}-x_{min}}
$$

适合输入范围有约束的模型。

-   **Robust Scaling（抗异常值）：**

$$
x'=\frac{x-\text{median}(x)}{IQR},\qquad IQR=Q_3-Q_1
$$

-   **对数变换（右偏分布常用）：**

$$
x'=\log(1+x)
$$

### 类别特征编码（怎么算）

-   **One-Hot Encoding：** 类别 $c_k$ 映射到单位向量 $\mathbf e_k$。
-   **频次编码：**

$$
x'=\frac{\#\{x=c\}}{n}
$$

-   **Target Encoding（二分类）：** 对类别 $c$ 计算标签均值

$$
TE(c)=\frac{1}{N_c}\sum_{i:x_i=c} y_i
$$

为减少小样本类别噪声，可做平滑：

$$
TE_{smooth}(c)=\frac{N_c\cdot \mu_c + m\cdot \mu}{N_c+m}
$$

其中   $\mu_c$ 为类别均值，  $\mu$ 为全局均值，$m$ 为平滑强度。  
注意：TE 必须用 out-of-fold 方式构造，避免标签泄漏。

### 缺失值与异常值处理

-   **缺失指示特征：** 增加   $I_{miss}(x)$ 标记是否缺失。
-   **简单填充：** 均值/中位数/众数。
-   **模型填充：** KNN / 回归填充（开销更大）。
-   **异常值检测：**
    -   Z-score：   $|z|>\tau$
    -   IQR 规则：   $x<Q_1-1.5IQR$ 或   $x>Q_3+1.5IQR$


### 特征选择（Filter / Wrapper / Embedded）

-   **方差过滤：** 去掉   $\text{Var}(x_j)<\epsilon$ 的特征。
-   **相关系数过滤：** 去掉与标签弱相关或与其他特征高度共线特征。
-   **互信息：** 衡量非线性依赖关系

$$
I(X;Y)=\sum_{x,y}p(x,y)\log\frac{p(x,y)}{p(x)p(y)}
$$

-   **RFE（递归特征消除）：** 训练模型后按重要性迭代删特征。
-   **L1 稀疏选择：** 通过   $\lambda\|\mathbf w\|_1$ 促使部分权重为 0。
-   **树模型重要性：**
    -   Gain：该特征带来的损失下降总量
    -   Split：该特征被用于分裂的次数

### 可解释性方法

-   **线性模型系数解释：** 在特征同量纲或已标准化条件下，  $|w_j|$ 反映影响强弱。
-   **PDP（Partial Dependence）：**
  对目标特征   $x_s$ 的全局平均效应：

$$
\hat f_s(z)=\frac{1}{n}\sum_{i=1}^n \hat f(z,x_{i,C})
$$

其中   $C$ 为除   $s$ 外的特征集合。

-   **ICE（Individual Conditional Expectation）：**
  不做样本平均，直接看每个样本曲线   $\hat f(z,x_{i,C})$，可发现群体异质性。

-   **LIME（局部线性解释）：**
  在样本   $x$ 周围采样邻域点，拟合可解释模型   $g$：

$$
\xi(x)=\arg\min_{g\in G} L(f,g,\pi_x)+\Omega(g)
$$

其中   $\pi_x$ 为邻域权重，  $\Omega(g)$ 控制解释复杂度。

-   **SHAP（Shapley Additive Explanations）：**
  对特征   $i$ 的贡献值：

$$
\phi_i=\sum_{S\subseteq N\setminus\{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!}\left[v(S\cup\{i\})-v(S)\right]
$$

满足局部加和：

$$
f(x)=\phi_0+\sum_{i=1}^M \phi_i
$$

即模型输出可分解为基线值   $\phi_0$ 与各特征贡献之和。


# 监督学习算法：LR，SVM

## logistic regression

- **模型函数**

逻辑回归使用Sigmoid函数将线性回归的输出映射到(0,1)区间：

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

其中  $z = \mathbf{w}^T \mathbf{x} + b$，$\mathbf{w}$  是权重向量，  $b$  是偏置项。

- **假设函数**

分类概率（'潜变量'或者称为概率变量）表示为：

$$h_{\mathbf{w}}(\mathbf{x}) = P(y=1|\mathbf{x};\mathbf{w}) = \sigma(\mathbf{w}^T \mathbf{x} + b)$$

- **决策边界**

当  $h_{\mathbf{w}}(\mathbf{x}) \geq 0.5$  时预测  $y=1$  ，否则预测  $y=0$  。这等价于  $\mathbf{w}^T \mathbf{x} + b \geq 0$  。  
- **损失函数**

使用交叉熵损失函数（对数似然损失）：

$$J(\mathbf{w}) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_{\mathbf{w}}(\mathbf{x}^{(i)})) + (1-y^{(i)}) \log(1-h_{\mathbf{w}}(\mathbf{x}^{(i)})) \right]$$

其中  $m$  是样本数量。

- **梯度下降优化**

权重更新公式：

$$\mathbf{w} := \mathbf{w} - \alpha \frac{\partial J(\mathbf{w})}{\partial \mathbf{w}}$$

梯度计算：

$$\frac{\partial J(\mathbf{w})}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (h_{\mathbf{w}}(\mathbf{x}^{(i)}) - y^{(i)}) x_j^{(i)}$$

- **正则化**

为避免过拟合，可以添加L1或L2正则化项：

$$J(\mathbf{w}) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_{\mathbf{w}}(\mathbf{x}^{(i)})) + (1-y^{(i)}) \log(1-h_{\mathbf{w}}(\mathbf{x}^{(i)})) \right] + \frac{\lambda}{2m} \|\mathbf{w}\|^2$$

- **多分类扩展**

标准的逻辑回归是一个二分类模型。为了将其扩展到处理多分类问题（即类别数
$K > 2$），主要有两种常用的策略：

-   **One-vs-Rest (OvR)，或称One-vs-All**\
    此策略将多分类问题转化为多个二分类问题。具体而言，为每一个类别  $i$
    训练一个独立的二分类器。该分类器将类别  $i$
    的样本视为正例，将所有其他类别  $j \neq i$  的样本视为反例。\
    **训练：** 共训练  $K$
    个模型，每个模型输出样本属于其对应类别的概率。\
    **预测：** 将新的输入样本  $x$  输入所有  $K$
    个模型，选择输出概率最高的那个类别作为最终的预测结果：

    $$\hat{y} = \arg\max_{i \in \{1, \ldots, K\}} h_{\mathbf{w}_i}(\mathbf{x})$$

    **优点：** 简单直观，易于实现\
    **缺点：** 类别多时需训练大量模型；样本不均衡问题

-   **Softmax回归（多项逻辑回归）**\
    Softmax回归是逻辑回归在多分类问题上的直接推广，它是一个单一的模型，可以直接输出样本属于每个类别的概率。\
    **核心：**
    Softmax函数将模型对于每个类别的线性预测（logits）  $\mathbf{z} = \mathbf{W}^T \mathbf{x} + \mathbf{b}$
    映射为一个概率分布。对于类别  $i$，其概率为：

    $$P(y=i|\mathbf{x}) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}} = \frac{e^{\mathbf{w}_i^T \mathbf{x} + b_i}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^T \mathbf{x} + b_j}}$$

    所有类别的概率之和为 1，即  $\sum_{i=1}^{K} P(y=i|\mathbf{x}) = 1$  。\
    **损失函数：**
    使用交叉熵损失函数来衡量预测概率分布与真实标签分布（one-hot编码）之间的差异。对于单个样本，其损失为：

    $$J(\mathbf{W}, \mathbf{b}) = -\sum_{i=1}^{K} y_i \log(P(y=i|\mathbf{x}))$$

    其中  $\mathbf{y}$  是样本的真实标签的 one-hot 编码向量。\
    **预测：** 输出概率最大的类别作为预测结果：  

    $\hat{y} = \arg\max_{i} P(y=i|\mathbf{x})$   

    **优点：** 多分类自然延伸，概率解释性强，通常优于OvR\
    **缺点：** 计算成本更高（需计算所有类别概率）


## 支持向量机与支持向量回归（SVM/SVR）

支持向量机（SVM）是一种强大的监督学习算法，主要用于分类任务，其扩展形式支持向量回归（SVR）则用于回归问题。SVM/SVR的核心思想是通过寻找最优超平面实现最大间隔分类或回归。

- **基本概念与数学原理**

-   **最优超平面**

    -   分类目标：寻找分离两类样本且具有最大间隔的超平面

    -   回归目标：寻找包含最多样本在  $\epsilon$  管道内的超平面

    -   数学表达：超平面定义为  $\mathbf{w}^T \mathbf{x} + b = 0$

-   **硬间隔SVM（线性可分情况）**

    -   优化目标：

        $$\min_{\mathbf{w},b} \frac{1}{2} \|\mathbf{w}\|^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1, \forall i$$

    -   支持向量：距离超平面最近的样本点

-   **软间隔SVM（非线性可分情况）**

    -   引入松弛变量  $\xi_i$  处理噪声和异常点

    -   优化目标：

        $$\min_{\mathbf{w},b,\xi} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^n \xi_i$$

        $$\text{s.t.} \quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

    -   $C$  为惩罚参数，控制间隔宽度与分类错误的权衡

- **核技巧与非线性SVM**

-   **核函数原理**

    -   将原始特征映射到高维空间： $\phi: \mathcal{X} \to \mathcal{H}$

    -   避免显式计算高维特征，使用核函数  $K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^T \phi(\mathbf{x}_j)$

-   **常用核函数**

    -   线性核：  $K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T \mathbf{x}_j$

    -   多项式核：  $K(\mathbf{x}_i, \mathbf{x}_j) = (\gamma \mathbf{x}_i^T \mathbf{x}_j + r)^d$

    -   高斯径向基（RBF）核：  $K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$

    -   Sigmoid核：  $K(\mathbf{x}_i, \mathbf{x}_j) = \tanh(\gamma \mathbf{x}_i^T \mathbf{x}_j + r)$

- **支持向量回归（SVR）**

SVR是SVM在回归问题中的扩展，目标是找到一个回归函数  $f(\mathbf{x}) = \mathbf{w}^T \phi(\mathbf{x}) + b$  ，使尽可能多的样本位于  $\epsilon$  -不敏感管道内。

-   **优化问题**

$$\min_{\mathbf{w},b,xi,xi^{*}} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^n (\xi_i + \xi_i^*)$$
    
$$\text{s.t.} \begin{cases} 
            y_i - (\mathbf{w}^T \phi(\mathbf{x}_i) + b) \leq \epsilon + \xi_i \\
            (\mathbf{w}^T \phi(\mathbf{x}_i) + b) - y_i \leq \epsilon + \xi_i^* \\
            \xi_i, \xi_i^* \geq 0
        \end{cases}$$

-   **损失函数**

    -   $\epsilon$-不敏感损失：    

$$L_\epsilon(y, f(\mathbf{x})) = \begin{cases} 
                    0 & \text{if } |y - f(\mathbf{x})| \leq \epsilon \\
                    |y - f(\mathbf{x})| - \epsilon & \text{otherwise}
                \end{cases}$$

- **参数选择与模型训练**

-   **关键参数**

    -   $C$（惩罚参数）：控制间隔宽度与训练误差的权衡

    -   $\epsilon$（SVR）：控制回归管道宽度

    -   $\gamma$（RBF核）：控制单个样本的影响范围

    -   核函数选择：根据数据特性选择合适核函数

- **优势与局限**

-   **优势：** 高维空间表现优异；核技巧处理非线性；基于凸优化有全局最优解；泛化能力强，抗过拟合
-   **局限：** 大规模数据训练慢；参数选择对性能影响大；概率估计不直接（需额外处理）；黑盒模型，解释性较差


# 监督学习算法：NN model

## 神经网络

- **基本概念**

神经网络是受人脑神经系统启发而设计的计算模型，由大量相互连接的简单处理单元（神经元）组成。三层的神经网络即可在理论上拟合任意函数。

- **神经元**

$$y = f\left(\sum_{i=1}^{n} w_i x_i + b\right)$$

-   $x_i$ 是输入变量

-   $w_i$ 是连接权重（系数）

-   $b$ 是偏置项（常系数）

-   $f$ 是激活函数

得到y之后，作为下一层神经元的输入变量（因此会有"层"的概念）

- **常见激活函数**

-   **常用激活函数比较：**

    -   **ReLU（Rectified Linear Unit）：** $$f(x) = \max(0, x)$$
        优点：计算简单，缓解梯度消失；缺点：神经元\"死亡\"问题

    -   **Leaky ReLU：**

    $$f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{otherwise} \end{cases}$$

    -   **ELU（Exponential Linear Unit）：**

    $$f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{otherwise} \end{cases}$$

    -   **Sigmoid：**   $\sigma(x) = \frac{1}{1 + e^{-x}}  $
        适用于输出层（二分类），但容易出现梯度饱和

    -   **Tanh：**   $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} $
        输出零中心化，但同样存在梯度饱和问题

- **前向传播**

输入数据通过网络层层传递的过程，就像我前面已经说过的，后一层（l-1）的输出结果将作为本层(l)的输入

$$\mathbf{a}^{\textcolor{red}{(l)}} = f(\mathbf{W}^{(l)} \mathbf{a}^{\textcolor{red}{(l-1)}} + \mathbf{b}^{(l)})$$

- **损失函数**

-   均方误差（回归问题）:
    $J(\mathbf{W}, \mathbf{b}) = \frac{1}{2m} \sum_{i=1}^{m} \| \mathbf{y}_i - \mathbf{\hat{y}}_i \|^2$

-   交叉熵损失（分类问题）:
    $J(\mathbf{W}, \mathbf{b}) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{j=1}^{k} y_{ij} \log(\hat{y}_{ij})$

- **反向传播**

通过链式法则计算损失函数对网络参数的梯度：

$$\frac{\partial J}{\partial w_{ij}^{(l)}} = \frac{\partial J}{\partial z_j^{(l)}} \cdot \frac{\partial z_j^{(l)}}{\partial w_{ij}^{(l)}} = \delta_j^{(l)} \cdot a_i^{(l-1)}$$

其中 $\delta_j^{(l)}$  是第  $l$  层第  $j$  个神经元的误差项。对于最经典的sigmoid函数：

$$a^{(l)} = \frac{1}{1+e^{wa^{(l-1)}+b}}=\frac{1}{1+e^{z}}, \frac{\partial a^{(l)}}{\partial z} = a^{(l)}(1-a^{(l)})$$

- **正则化技术**

正则化技术是防止机器学习模型过拟合（Overfitting）的重要手段，通过在训练过程中引入各种约束或噪声，提高模型的泛化能力。除了之前介绍过的L1/L2权重正则化外，还有以下几种重要的正则化技术：

-   **Dropout**:一种在深度学习中最常用且有效的正则化技术，通过在训练过程中随机\"丢弃\"一部分神经元来防止过拟合。

    -   **工作原理：**
        简单来说就是训练时让一部分神经元输出为0，训练神经网络，从而降低神经元之间的依赖性。测试时不让他输出为0，但最终输出进行一定缩放得到正确的值

    -   **训练阶段：** 前向传播时，每个神经元以概率$p$被丢弃：

    $$h_i^{(l)} = \begin{cases}
        0 & \text{以概率 } p \\
        f(z_i^{(l)}) & \text{以概率 } 1-p
    \end{cases}$$

    -   **测试阶段：**
        使用所有神经元，但将输出乘以保留概率  $(1-p)$  以保持期望值一致：   $h_i^{(l)} = (1-p) \cdot f(z_i^{(l)})$

    -   **正则化机制：**

        -   防止神经元间复杂的共适应关系，迫使每个神经元都能独立提供有用特征

        -   相当于训练了大量共享参数的子网络，测试时相当于这些子网络的集成

        -   增加网络稀疏性，提高泛化能力

    -   **变体：**

        -   DropConnect：随机丢弃权重连接而非神经元

        -   Spatial Dropout：在卷积网络中按通道随机丢弃

-   **早停（Early Stopping）**

    早停是一种简单而有效的正则化方法，通过监控验证集性能来决定何时停止训练。

    -   **实现方式：**

        -   将数据集分为训练集、验证集和测试集

        -   在每个梯度下降后计算模型在验证集上的性能

        -   当验证集误差连续多个轮次不再下降时停止训练

    -   **原理：**
        随着训练进行，模型会逐渐从欠拟合到过拟合。早停在即将过拟合时停止训练，获得泛化能力最好的模型。

    -   **优点：** 无需修改损失函数或网络结构；自动确定训练轮数；计算效率高，节省训练时间

    -   **实现细节：**

        -   设置耐心参数(patience)：允许验证集性能不提升的连续轮数

        -   保存验证集性能最佳时的模型参数

        -   可以配合学习率调度一起使用

-   **批量归一化（Batch
    Normalization）**：意味着每层输入数据，我先对每个维度的变量进行标准化，然后进行放缩，再作为新的输入数据，输入进网络。

    -   **算法：** 对每个小批量数据逐层进行标准化：
        $$\mu_B = \frac{1}{m} \sum_{i=1}^m x_i, \quad \sigma_B^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_B)^2$$
        $$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$
        其中  $\gamma$  和  $\beta$  是可学习的缩放和平移参数。

    -   **正则化机制：**

        -   为每层输入添加的噪声具有正则化效果(因为人工放缩了)

        -   减少对大规模权重的依赖，提高泛化能力

        -   允许使用更高的学习率，间接起到正则化作用

    -   **优点：** 加速训练收敛；减少对参数初始化敏感性；避免梯度消失/爆炸（人工放缩使梯度合理）；一定正则化效果，可减少/替代Dropout

    -   **变体：**

        -   Layer Normalization：沿特征维度归一化

        -   Instance Normalization：对每个样本单独归一化

        -   Group Normalization：将特征分组后归一化

-   **其他正则化技术**

    -   **数据增强（Data Augmentation）：**
        通过对训练数据进行各种变换（旋转、缩放、裁剪等）人工增加训练数据量

    -   **标签平滑（Label Smoothing）：**
        将硬标签（0或1）转换为软标签（如0.1和0.9），防止模型过度自信

    -   **随机深度（Stochastic Depth）：**
        在训练过程中随机丢弃整个网络层

    -   **噪声注入：** 在输入数据或网络激活中添加随机噪声

- **超参数调优**

超参数调优是神经网络设计与训练过程中的关键环节，直接影响模型的性能、训练效率和泛化能力。与模型参数（通过训练学习得到）不同，超参数需要在训练开始前手动设置。

-   **学习率（Learning Rate）**
    学习率是最重要的超参数之一，控制参数更新的步长大小。

    -   **重要性：** 直接影响收敛速度和最终性能

        -   过大：可能导致震荡、发散或不收敛

        -   过小：收敛速度慢，容易陷入局部最优

    -   **常用范围：** $10^{-6}$ 到 $1$ 之间，常用值：

        -   Adam优化器：  $10^{-4}$   到   $10^{-2}$（默认0.001）

        -   SGD：  $10^{-2}$ 到   $10^{-1}$

    -   **调优策略：**

        -   学习率网格搜索：在对数尺度上进行搜索（如0.001, 0.01, 0.1）

        -   学习率衰减：随训练进程逐渐减小学习率 

        $$\begin{aligned}
                        \text{指数衰减：} & \alpha = \alpha_0 \cdot e^{-kt} \\
                        \text{阶梯衰减：} & \text{每隔固定epoch将学习率减半}        
        \end{aligned}$$

        -   循环学习率（Cyclical LR）：在合理范围内周期性变化学习率

        -   warmup：训练初期使用较小学习率，逐步增加到设定值。

    -   **自适应学习率算法：**
        Adam算法为每个参数自适应调整学习率  

-   **批量大小（Batch Size）**

    批量大小指每次参数更新时使用的样本数量，影响训练稳定性、速度和内存需求。

    -   **影响分析：**

        -   **小批量（32-256）：**
            梯度估计有噪声，有助于跳出局部最小值，泛化性能通常更好

        -   **大批量（1024+）：**
            梯度估计更准确，训练速度更快（可并行计算），但可能泛化能力较差

    -   **选择建议：**

        -   一般从32、64、128等常用值开始尝试

        -   受GPU内存限制，需要在内存允许范围内选择最大批量

-   **超参数优化方法**

    -   **网格搜索（Grid Search）：** 在预定范围内系统尝试所有参数组合

    -   **随机搜索（Random Search）：**
        在参数空间中随机采样，通常比网格搜索更高效

    -   **贝叶斯优化(optuna等)**
        建立代理模型指导参数选择，适合计算昂贵的模型

## 卷积神经网络（CNN）

卷积神经网络（Convolutional Neural Networks,
CNN）是一类专门设计用于处理网格状数据（如图像、视频、语音）的深度学习架构。其核心思想是通过局部连接、权重共享和空间下采样来有效捕获数据的空间层次特征。

-   **核心概念与优势**

    -   **局部感受**
        每个神经元仅与输入数据的局部区域相连，这符合图像中相邻像素相关性更强的先验知识。

    -   **权重共享**
        同一卷积核在整个输入数据上滑动并计算，极大减少了需要学习的参数数量。

    -   **空间下采样**
        通过池化层逐步降低特征图分辨率，增加感受野并提高模型对平移的鲁棒性。

-   **卷积操作详解**

    卷积是CNN的核心操作，通过卷积核（滤波器）在输入数据上滑动并进行局部加权求和来提取特征。

    -   **单通道卷积示例：**

        假设输入为一个 $5 \times 5$ 的矩阵 $\mathbf{I}$，卷积核为
        $3 \times 3$ 的矩阵 $\mathbf{K}$，步长为1（每次滑动1）：

        $$\mathbf{I} = \begin{bmatrix}
            \textcolor{red}{1} & \textcolor{red}{2} & \textcolor{red}{1} & 0 & 2 \\
            \textcolor{red}{0} & \textcolor{red}{1} & \textcolor{red}{2} & 1 & 0 \\
            \textcolor{red}{2} & \textcolor{red}{0} & \textcolor{red}{1} & 2 & 1 \\
            1 & 2 & 0 & 1 & 2 \\
            0 & 1 & 2 & 1 & 0 \\
        \end{bmatrix}, \quad
        \mathbf{K} = \begin{bmatrix}
            1 & 0 & -1 \\
            1 & 0 & -1 \\
            1 & 0 & -1 \\
        \end{bmatrix}$$

    -   计算输出特征图中位置 $(1,1)$ 的值（左上角）： 
        
        $$\begin{aligned}
            O_{1,1} &= \sum_{i=1}^{3} \sum_{j=1}^{3} I_{i,j} \cdot K_{i,j} \\
            &= (1\times1) + (2\times0) + (1\times-1) + \\
            &\quad (0\times1) + (1\times0) + (2\times-1) + \\
            &\quad (2\times1) + (0\times0) + (1\times-1) \\
            &= 1 + 0 - 1 + 0 + 0 - 2 + 2 + 0 - 1 = -1
        \end{aligned}$$

    -   **多通道卷积：**

        对于RGB三通道输入（[大白话：有三个矩阵I先后叠在一起类似一个长方体，那么卷积核是从左上角开始慢慢移动]），每个卷积核也是三维的：
        $$O_{x,y} = \sum_{c=1}^{C} \sum_{i=1}^{K_h} \sum_{j=1}^{K_w} I_{c, x+i, y+j} \cdot K_{c, i, j} + b$$
        其中  $C$  为输入通道数，  $K_h \times K_w$   为卷积核空间尺寸，  $b$  为偏置。

    -   **超参数影响：**

        -   **步长（Stride）：** 控制卷积核滑动间隔，影响输出尺寸

        -   **填充（Padding）：**
            在输入周围添加零值，控制输出尺寸（[通俗理解：如果是4×8的矩阵，卷积核是2×2，那么正常乘完就只有3×7.因此，当你在原来4×8矩阵周围填充一圈0，最终乘完的结果就是4×8]{style="color: red"}）

        -   **输出尺寸计算公式：**
            $$O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$
            其中   $W$  为输入尺寸，  $K$  为卷积核尺寸，  $P$   为填充数，  $S$  为步长。

-   **池化操作（Pooling Operation）**

    池化层是CNN中的重要组成部分，其主要功能是进行特征降维和空间不变性增强。通过减少[特征图(你刚才卷积核算出来的图)]{style="color: red"}的空间尺寸，池化操作能够有效降低计算复杂度、控制过拟合，并使模型对输入的小幅平移和形变更加鲁棒。

    -   **最大池化（Max Pooling）**
        最大池化选取局部区域中的最大值作为输出，能够有效保留最显著的特征响应。

        -   **操作示例：** 假设 $2\times2$ 池化窗口，步长为2：

            输入特征块： 
            
            $$\begin{bmatrix}
                        1 & 3 & 2 & 1 \\
                        4 & 2 & 0 & 3 \\
                        2 & 1 & 4 & 2 \\
                        3 & 2 & 1 & 5 \\
                    \end{bmatrix}$$

            池化过程： 
            
            $$\begin{bmatrix}
                        \max(1,3,4,2) & \max(2,1,0,3) \\
                        \max(2,1,3,2) & \max(4,2,1,5) \\
                    \end{bmatrix}
                    = 
                    \begin{bmatrix}
                        4 & 3 \\
                        3 & 5 \\
                    \end{bmatrix}$$

        -   **特点：**

            -   更好地保留纹理特征和边缘信息

            -   对噪声具有一定的鲁棒性

            -   在实际应用中最为常用

    -   **平均池化（Average Pooling）**

        平均池化计算局部区域的平均值作为输出，能够保留整体特征分布信息。

        -   **操作示例：** 同样使用 $2\times2$ 池化窗口：

            输入特征块： 
            
            $$\begin{bmatrix}
                        1 & 3 & 2 & 1 \\
                        4 & 2 & 0 & 3 \\
                        2 & 1 & 4 & 2 \\
                        3 & 2 & 1 & 5 \\
                    \end{bmatrix}$$

            池化过程： 
            
            $$\begin{bmatrix}
                        \frac{1+3+4+2}{4} & \frac{2+1+0+3}{4} \\
                        \frac{2+1+3+2}{4} & \frac{4+2+1+5}{4} \\
                    \end{bmatrix}
                    = 
                    \begin{bmatrix}
                        2.5 & 1.5 \\
                        2.0 & 3.0 \\
                    \end{bmatrix}$$

        -   **特点：**

            -   平滑特征响应，减少背景噪声影响

            -   适用于需要保留整体信息的任务

            -   在网络的深层有时比最大池化效果更好

    -   **池化操作的超参数**

        -   **池化窗口大小：** 常见尺寸为 $2\times2$ 或 $3\times3$

        -   **步长（Stride）：** 通常与窗口大小相同（如 $2\times2$
            窗口使用步长2）

        -   **填充（Padding）：** 较少使用，通常为零填充

        -   **输出尺寸计算：**

            $$ O = \left\lfloor \frac{W - K}{S} \right\rfloor + 1 $$  
            
            其中
            $W$ 为输入尺寸，  $K$   为池化窗口尺寸，  $S$ 为步长。

    -   **池化层的作用与意义**

        -   **降维与计算效率：** 显著减少特征图尺寸，降低后续层的计算量

        -   **平移不变性：** 使模型对输入特征的位置变化不敏感

        -   **特征抽象：** 逐步构建更加抽象和高层的特征表示

        -   **过拟合控制：** 通过减少参数数量间接起到正则化作用

-   **CNN架构组成**

    -   **卷积层：** 核心特征提取层，使用多个卷积核提取不同特征

    -   **激活函数：** 引入非线性，常用ReLU及其变体

    -   **池化层：** 下采样操作，减少参数数量并增加感受野

        -   最大池化：  $\text{MaxPool}(x) = \max(x_{i,j})$

        -   平均池化：  $\text{AvgPool}(x) = \frac{1}{n} \sum x_{i,j}$

    -   **全连接层：**
        将提取的特征映射到最终输出类别（[简单来说，就是一层正常的神经网络]）

    -   **批归一化层：(batch normalization)**
        加速训练并提高稳定性[不过这里的normalization通常是指对r,g,b三个变量分别进行标准化，输入样本是每个像素点]

-   **参数数量计算示例**

    假设输入为   $224\times224\times3$ 的RGB图像，使用32个   $3\times3$   卷积核：
    $$\text{参数量} = (3 \times 3 \times 3 + 1) \times 32 = 896$$
    相比之下，同等条件下的全连接层参数量为：
    $$224 \times 224 \times 3 \times 32 = 4,816,896$$
    这充分展示了CNN的参数效率优势。

## 循环神经网络（RNN）

循环神经网络（Recurrent Neural Network,
RNN）是一类专门设计用于处理序列数据的神经网络架构。与传统的前馈神经网络不同，RNN具有循环连接，使其能够保持对先前信息的记忆，从而有效地处理时间序列、自然语言等具有时序依赖性的数据。

-   **核心思想与架构**
    RNNs的核心特点是其隐藏层神经元之间存在循环连接，使网络能够保持内部状态来捕获序列中的时序信息。

    -   **循环结构：** 在每个时间步  $t$，RNN接收当前输入  $\mathbf{x}_t$
        和上一时间步的隐藏状态  $\mathbf{h}_{t-1}$：

        $$\mathbf{h}_t = f(\mathbf{W}_{xh} \mathbf{x}_t + \mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{b}_h)$$

        其中  $\mathbf{W}_{xh}$   是输入到隐藏层的权重矩阵，  $\mathbf{W}_{hh}$  是隐藏层到隐藏层的权重矩阵。

    -   **输出计算：** 当前时间步的输出为：

        $$\mathbf{o}_t = g(\mathbf{W}_{ho} \mathbf{h}_t + \mathbf{b}_o)$$

    -   **参数共享：** 所有时间步共享相同的参数集  $(\mathbf{W}_{xh}, \mathbf{W}_{hh}, \mathbf{W}_{ho})$  ，大大减少了参数量。

-   **RNN的展开形式**

    为了更清晰地理解RNN的计算过程，通常将其按时间步展开：

            时间步: t-1         t          t+1
            ──────────────────────────
            输入:   x_{t-1}  →  x_t    →  x_{t+1}
            ↓         ↓         ↓
            隐藏层: h_{t-1} → h_t     → h_{t+1}
            ↓         ↓         ↓
            输出:   o_{t-1}    o_t       o_{t+1}
            
            参数共享: W 在所有时间步相同
            信息流向: 从左向右（→）表示前向传播
            循环连接表示隐藏状态在时间步间传递

    这种展开形式显示了RNN如何通过时间传播信息，每个时间步都可以看作是一个共享参数的前馈网络。

-   **反向传播通过时间（BPTT）**

    RNN的训练使用反向传播通过时间算法，将误差从最终时间步反向传播到初始时间步：

    -   **梯度计算：** 需要计算损失函数对每个时间步参数的梯度

    -   **梯度流动：** 误差通过时间反向传播，更新所有时间步的共享参数

    -   **数学表达：** 对于时间步  $t$  的梯度：
        $$\frac{\partial L}{\partial \mathbf{W}} = \sum_{k=1}^{t} \frac{\partial L}{\partial \mathbf{o}_t} \frac{\partial \mathbf{o}_t}{\partial \mathbf{h}_t} \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_k} \frac{\partial \mathbf{h}_k}{\partial \mathbf{W}}$$

-   **梯度消失与梯度爆炸问题**

    RNN在训练长序列时面临梯度消失和梯度爆炸的挑战：

    -   **梯度消失：**
        当误差通过多个时间步反向传播时，梯度可能指数级减小，导致早期时间步的参数无法有效更新

    -   **梯度爆炸：** 梯度可能指数级增大，导致训练不稳定

    -   **解决方案：**

        -   梯度裁剪（Gradient Clipping）：限制梯度的大小

        -   使用改进的RNN架构：LSTM、GRU等

-   **双向RNN（Bi-RNN）**

    双向RNN同时考虑过去和未来的上下文信息：

    -   **前向与后向：** 包含两个独立的RNN，分别从两个方向处理序列

    -   **输出组合：** 将两个方向的隐藏状态组合形成最终输出：

        $$\mathbf{h}_t = [\overrightarrow{\mathbf{h}_t}, \overleftarrow{\mathbf{h}_t}]$$

    -   **应用：**
        特别适合需要全局上下文信息的任务，如机器翻译、语音识别

-   **RNN的应用领域**

    -   **自然语言处理：** 语言建模、机器翻译、文本生成

    -   **时间序列预测：** 股票价格预测、天气预测

    -   **语音处理：** 语音识别、语音合成

    -   **视频分析：** 动作识别、视频描述生成

-   **局限性与发展趋势**

    -   **局限性：** 尽管LSTM/GRU缓解了梯度问题，但处理极长序列仍有困难

    -   **发展趋势：**
        注意力机制和Transformer架构在许多序列任务上逐渐取代传统RNN

    -   **混合架构：** CNN与RNN结合，用于视频分析等多模态任务

## 长短期记忆网络（LSTM）

长短期记忆网络（Long Short-Term Memory, LSTM）是循环神经网络（RNN）的一种特殊变体，由Hochreiter和Schmidhuber于1997年提出。LSTM专门设计用于解决传统RNN在处理长序列时遇到的[梯度消失和梯度爆炸]问题

-   **核心创新：门控机制与细胞状态**
    LSTM的核心思想是通过三个门控单元（输入门、遗忘门、输出门）和一个细胞状态来调控信息的流动。

    -   **细胞状态（Cell State）：**   $\mathbf{C}_t$
        是LSTM中的核心组件，充当\"信息传送带\"，能够在序列处理过程中保持和传递信息。其更新公式为：

        $$\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t$$

        其中   $\odot$   表示逐元素乘法（Hadamard积）。

    -   **遗忘门（Forget Gate）：** 决定从细胞状态中丢弃哪些信息：

        $$\mathbf{f}_t = \sigma(\mathbf{W}_f \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$$

        输出值在0到1之间，1表示\"完全保留\"，0表示\"完全丢弃\"。

    -   **输入门（Input Gate）：** 决定哪些新信息将被存储到细胞状态中：

        $$\mathbf{i}_t = \sigma(\mathbf{W}_i \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$$

        同时计算候选细胞状态：

        $$\tilde{\mathbf{C}}_t = \tanh(\mathbf{W}_C \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_C)$$

    -   **输出门（Output Gate）：** 基于细胞状态决定输出什么信息：

        $$\mathbf{o}_t = \sigma(\mathbf{W}_o \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)$$
        最终隐藏状态为：

        $$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t)$$

-   **解决梯度消失问题的机制**

    LSTM通过以下设计有效缓解了传统RNN的梯度消失问题：

    -   **常数误差传送带：** 细胞状态  $\mathbf{C}_t$
        的更新主要是线性操作，梯度可以长时间流动而不衰减

    -   **门控调节：**
        遗忘门允许网络[自主决定(实际上就是加了一层参数，控制遗忘多少)]{style="color: red"}保留多少历史信息，避免梯度指数级衰减

    -   **梯度路径：** 误差通过细胞状态反向传播时，路径相对直接且稳定

-   **变体与改进**

    基于标准LSTM，研究者提出了多种改进版本：

    -   **窥孔连接（Peephole Connections）：**允许门控单元查看细胞状态

        $$\mathbf{f}_t = \sigma(\mathbf{W}_f \cdot [\mathbf{C}_{t-1}, \mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$$

    -   **耦合遗忘门和输入门：** 将遗忘门和输入门合并，减少参数数量：

        $$\mathbf{C}_t = (1 - \mathbf{i}_t) \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t$$

    -   **GRU（Gated Recurrent Unit）：**
        LSTM的简化版本，将遗忘门和输入门合并为更新门，细胞状态和隐藏状态合并

-   **LSTM的参数数量计算**

    对于一个LSTM单元，参数数量计算如下：

    $$\text{参数数量} = 4 \times (\text{输入维度} + \text{隐藏层维度} + 1) \times \text{隐藏层维度}$$

    其中4对应四个门控相关矩阵（输入门、遗忘门、输出门、候选细胞状态），+1考虑偏置项。


# 监督学习算法：Tree model

## 决策树模型

决策树是一种基于树结构进行决策的机器学习模型，它通过递归地将数据划分成更纯的子集来进行预测。决策树可用于分类和回归任务，具有直观易懂、可解释性强的特点。[人话：分类枚举，总有一个树叶能分到你]

- **基本概念与结构**

-   **树结构组成**

    -   根节点：起始节点

    -   内部节点：表示特征测试，每个分支代表一个测试结果[（人话：进行分类，比如age小于20的一类，age大于20的一类）]

    -   叶节点：表示最终的预测结果（类别或数值）[（人话：比如age大于20的，收入为income=3age+3,age小于20的，income=2age+4）]

    -   分支：特征测试的可能结果路径

-   **关键术语**

    -   特征选择：选择最优划分特征的标准（age）

    -   划分点：连续特征的切分阈值（20）

    -   节点纯度：节点中样本类别的同质性程度（每个节点多少样本）

    -   树深度：从根节点到最远叶节点的路径长度（你分了多少次类）

- **决策树构建算法**

决策树的构建通常采用贪心的递归分裂策略：

-   **ID3算法**（Iterative Dichotomiser 3）

    -   使用信息增益作为特征选择标准

    -   仅支持离散特征，不支持连续特征和缺失值

    -   倾向于选择取值较多的特征

-   **C4.5算法**

    -   ID3的改进版本，使用信息增益比

    -   支持连续特征离散化和处理缺失值

    -   通过后剪枝防止过拟合

-   **CART算法**（Classification and Regression Trees）

    -   支持分类和回归任务

    -   使用基尼系数（分类）或平方误差（回归）

    -   生成二叉树，计算效率高

- **划分标准与数学表达**

-   **信息增益（Information Gain）**
    $IG(D, f) = H(D) - \sum_{v=1}^{V} \frac{|D_v|}{|D|} H(D_v)$ 其中
    $H(D)$ 为数据集 $D$ 的经验熵，$D_v$ 为特征 $f$ 取第 $v$
    个值时的样本子集。

-   **信息增益比（Gain Ratio）** $$GR(D, f) = \frac{IG(D, f)}{IV(f)}$$
    其中
    $IV(f) = -\sum_{v=1}^{V} \frac{|D_v|}{|D|} \log_2 \frac{|D_v|}{|D|}$
    为特征 $f$ 的固有值。

-   **基尼系数（Gini Index）** $$Gini(D) = 1 - \sum_{k=1}^{K} p_k^2$$
    其中 $p_k$ 为第 $k$ 类样本在数据集 $D$ 中的比例。

-   **回归树的划分标准**
    $$\text{选择划分} = \arg\min_f \left[ \sum_{x_i \in D_l} (y_i - \bar{y}_l)^2 + \sum_{x_i \in D_r} (y_i - \bar{y}_r)^2 \right]$$
    其中 $D_l$ 和 $D_r$ 为划分后的左右子集，$\bar{y}_l$ 和 $\bar{y}_r$
    为对应的输出均值。

- **剪枝技术**

为防止过拟合，决策树需要进行剪枝：

-   **预剪枝（Pre-pruning）**

    -   在构建过程中提前停止分裂

    -   标准：节点样本数少于阈值、深度限制、纯度提升不足

    -   优点：计算高效；缺点：可能欠拟合

-   **后剪枝（Post-pruning）**

    -   先构建完整树，然后自底向上剪枝

    -   方法：代价复杂度剪枝、错误率降低剪枝

    -   优点：保留更多分支，泛化能力更好

-   **代价复杂度剪枝** $$R_\alpha(T) = R(T) + \alpha |\tilde{T}|$$ 其中
    $R(T)$ 为预测误差，$|\tilde{T}|$ 为叶节点数，$\alpha$ 为复杂度参数。

- **决策树的优势与局限**

-   **优势：** 可解释性强，决策过程可视化；数据准备简单，不需特征缩放；非参数方法，不假设数据分布；有缺失值处理机制
-   **局限性：** 易过拟合，需剪枝/停止条件；对数据微小变化敏感，不稳定；偏向取值多的特征；忽略特征间相关性

## 集成树模型：随机森林与GBDT

随机森林（Random Forest）和梯度提升决策树（GBDT）是基于决策树的两种强大集成学习方法，它们通过组合多个弱学习器（决策树）来构建更强大的模型。

### 随机森林（Random Forest）

随机森林是一种Bagging（Bootstrap
Aggregating）集成方法，通过构建多棵决策树并综合其预测结果来提高模型性能。[人话：训练多个树，投票决定最终结果]

-   **核心思想**

    -   数据随机性：对训练数据进行有放回抽样（Bootstrap抽样）

    -   特征随机性：在每棵树分裂时随机选择特征子集

    -   投票机制：分类任务采用多数投票，回归任务采用平均

-   **算法流程**

    1.  从原始训练集中进行Bootstrap抽样（有放回抽样），创建k个不同的训练子集

    2.  对每个训练子集构建一棵决策树，分裂时从所有特征中随机选择m个特征（$m \ll M$）

    3.  每棵树独立生长，不进行剪枝

    4.  预测时，所有树的预测结果进行聚合： 
    $$\hat{y} = \begin{cases} 
                    \text{mode}\{h_1(x), \dots, h_k(x)\} & \text{(分类)} \\
                    \frac{1}{k}\sum_{i=1}^k h_i(x) & \text{(回归)}
                \end{cases}$$

-   **关键特性**

    -   并行训练：各树独立构建，可并行化

    -   袋外误差（OOB Error）：未参与某棵树训练的样本可用作验证集

    -   特征重要性：基于特征在分裂中的平均纯度提升计算

    -   抗过拟合：通过随机性和聚合降低方差

-   **参数调优**

    -   `n_estimators`：树的数量（通常100-500）

    -   `max_features`：分裂时考虑的特征数（  $\sqrt{M}$  或  $\log_2 M$  ）

    -   `max_depth`：树的最大深度

    -   `min_samples_split`：节点分裂的最小样本数

-   **优势与局限**

    -   优势：高准确性，抗过拟合，处理高维数据，内置特征选择
    -   局限：计算开销大，模型可解释性差（相比单棵树）

### 梯度提升决策树（GBDT）

GBDT是一种Boosting集成方法，通过顺序构建决策树，每棵树学习修正前一棵树的残差。[人话：A树拟合了一个值，和真实值有差距a，那么我B树对A树的a进行拟合，这样一步一步减少误差]
-   **核心思想**

    -   顺序训练：树按顺序构建，每棵树学习前一棵树的残差

    -   梯度下降：使用梯度下降优化任意可微损失函数

    -   加法模型：最终预测是所有树的加权和
        $F_m(x) = F_{m-1}(x) + \nu h_m(x)$ 其中 $\nu$
        为学习率（收缩因子）

-   **算法流程（回归任务）**

    1.  初始化模型：$F_0(x) = \arg\min_\gamma \sum_{i=1}^n L(y_i, \gamma)$

    2.  对于 $m=1$ 到 $M$：

        1.  计算伪残差：$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F(x)=F_{m-1}(x)}$

        2.  用决策树拟合伪残差 $\{(x_i, r_{im})\}_{i=1}^n$，得到区域划分
            $R_{jm}$

        3.  计算每个区域的输出值：$\gamma_{jm} = \arg\min_\gamma \sum_{x_i \in R_{jm}} L(y_i, F_{m-1}(x_i) + \gamma)$

        4.  更新模型：$F_m(x) = F_{m-1}(x) + \nu \sum_{j=1}^{J_m} \gamma_{jm} I(x \in R_{jm})$

    3.  输出最终模型 $F_M(x)$

-   **损失函数**

    -   回归：均方误差（MSE）、绝对误差（MAE）、Huber损失

    -   分类：对数损失（Log Loss）

-   **关键特性**

    -   顺序训练：树按顺序构建，无法并行化训练

    -   学习率：控制每棵树的贡献，防止过拟合

    -   子采样：随机选择部分样本训练每棵树（类似随机森林）

-   **参数调优**

    -   `n_estimators`：树的数量（通常100-1000）

    -   `learning_rate`：学习率（通常0.01-0.1）

    -   `max_depth`：树的最大深度（通常3-8）

    -   `subsample`：样本采样比例（0.5-1.0）

    -   `min_samples_split`：节点分裂的最小样本数

-   **优势与局限**
    -   优势：高预测精度，处理混合数据，特征重要性
    -   局限：训练时间长，对超参数敏感，可能过拟合

## XGBoost（Extreme Gradient Boosting）

XGBoost是一种高效的梯度提升框架，由陈天奇于2016年提出。它结合了梯度提升决策树的优势与多项创新优化，在各类机器学习竞赛和工业应用中表现出色。

- **核心思想与创新**

XGBoost在传统GBDT基础上引入了多项关键技术改进：

-   **正则化目标函数**
    $$\mathcal{L}(\phi) = \sum_{i} l(\hat{y}_i, y_i) + \sum_{k} \Omega(f_k)$$
    其中
    $\Omega(f_k) = \gamma T + \frac{1}{2}\lambda \|w\|^2$，$T$为叶节点数，$w$为叶节点权重（你看，自带正则化）

-   **损失函数二阶导数**
    使用损失函数的二阶导数信息（把损失函数泰勒展开到二阶，把高阶小量扔掉即可）：
    $$\mathcal{L}^{(t)} \approx \sum_{i=1}^n \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)$$
    其中 $g_i = \partial_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$,
    $h_i = \partial^2_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$

-   **加权分位法** 近似算法加速最优分裂点查找，处理大规模数据

-   **稀疏感知算法** 自动处理缺失值和稀疏数据（[实践中巨方便]）

- **算法原理**

XGBoost的树构建过程基于贪心算法：

1.  计算每个样本的一阶导数$g_i$和二阶导数$h_i$

2.  遍历所有特征，寻找最优分裂点：
    $$Gain = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma$$
    其中 $G_L = \sum_{i \in I_L} g_i$, $H_L = \sum_{i \in I_L} h_i$

3.  选择增益最大的分裂点构建树

4.  计算叶节点权重：
    $$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

5.  更新模型：$F_t(x) = F_{t-1}(x) + \eta w_j^*$

- **关键特性**

-   **并行计算**

    -   特征并行：在不同特征上并行寻找最优分裂

    -   数据并行：对数据进行分块处理

-   **稀疏数据处理**

    -   默认方向：自动学习缺失值的最佳处理方向

    -   稀疏矩阵存储：高效处理高维稀疏数据

-   **正则化技术**

    -   L1/L2正则化控制模型复杂度

    -   最大深度限制

    -   最小损失减少阈值

-   **内置交叉验证**

    -   支持k-fold交叉验证

    -   早停机制防止过拟合

- **参数体系**

XGBoost参数分为三类：

-   **通用参数**

    -   `booster`: 基础学习器类型（gbtree, gblinear, dart）

    -   `nthread`: 并行线程数

-   **Booster参数**

    -   `eta` ($\eta$): 学习率（默认0.3）

    -   `gamma` ($\gamma$): 分裂所需最小损失减少（默认0）

    -   `max_depth`: 树的最大深度（默认6）

    -   `lambda` ($\lambda$): L2正则化系数（默认1）

    -   `alpha`: L1正则化系数（默认0）

    -   `subsample`: 样本采样比例（默认1）

    -   `colsample_bytree`: 特征采样比例（默认1）

-   **学习任务参数**

    -   `objective`: 损失函数（reg:squarederror, binary:logistic等）

    -   `eval_metric`: 评估指标（rmse, mae, logloss, auc等）

    -   `seed`: 随机种子

### 树模型对比分析

| 特性 | 决策树 | 随机森林 | GBDT | XGBoost |
|------|--------|----------|------|---------|
| 类型 | 单模型 | Bagging集成 | Boosting集成 | Boosting集成（优化） |
| 训练方式 | 递归贪心 | 并行独立构建 | 串行依赖构建 | 串行依赖+特征/数据并行 |
| 过拟合风险 | 高（需剪枝） | 低（随机性+聚合） | 中-高（需学习率/早停） | 中（内置L1/L2正则化） |
| 可解释性 | 强（可视化规则） | 弱（多树投票） | 弱（多树加权） | 弱（多树加权） |
| 训练速度 | 快 | 快（可并行） | 慢（串行） | 较快（工程优化+并行） |
| 预测精度 | 低-中 | 高 | 高 | 非常高 |
| 缺失值处理 | 有机制 | 无内置 | 无内置 | 稀疏感知，自动学习方向 |
| 核心创新 | — | Bootstrap + 特征随机采样 | 梯度下降拟合残差 | 二阶泰勒展开 + 正则化 + 加权分位法 |
| 关键参数 | depth, min_split | n_estimators, max_features | n_estimators, lr, max_depth | eta, gamma, lambda, subsample |

# 无监督学习算法

## K-means聚类

K-means是一种经典的无监督学习算法，用于将数据划分为K个互斥的簇。该算法通过迭代优化簇内样本的相似度，广泛应用于数据挖掘、模式识别和图像分割等领域。[做聚类算法的入门级知识点]

- **核心思想与数学原理**

K-means算法的核心思想是通过最小化簇内平方误差来实现数据聚类。

-   **目标函数** $$J = \sum_{i=1}^{k} \sum_{x \in C_i} \|x - \mu_i\|^2$$
    其中 $C_i$ 表示第 $i$ 个簇，$\mu_i$ 表示第 $i$ 个簇的质心

-   **优化目标**
    $$\min_{C_1,\ldots,C_k} \sum_{i=1}^{k} \sum_{x \in C_i} \|x - \mu_i\|^2$$
    该优化问题是NP难问题，K-means采用启发式迭代方法求解

- **算法流程**

K-means算法通过以下步骤实现聚类：

1.  **初始化阶段**

    -   随机选择K个样本作为初始质心
        $\{\mu_1^{(0)}, \mu_2^{(0)}, \ldots, \mu_k^{(0)}\}$

    -   或者使用改进的初始化方法（如K-means++）

2.  **分配阶段**

    -   将每个样本分配到最近的质心所在的簇：
        $$C_i^{(t)} = \{x_j : \|x_j - \mu_i^{(t)}\|^2 \leq \|x_j - \mu_l^{(t)}\|^2 \ \forall l \neq i\}$$

3.  **更新阶段**

    -   重新计算每个簇的质心：
        $$\mu_i^{(t+1)} = \frac{1}{|C_i^{(t)}|} \sum_{x_j \in C_i^{(t)}} x_j$$

4.  **收敛判断**

    -   当质心不再显著变化或达到最大迭代次数时停止：
        $$\|\mu_i^{(t+1)} - \mu_i^{(t)}\| < \epsilon \quad \forall i$$

- **关键参数与选择**

-   **K值选择**

    -   肘部法则（Elbow Method）：绘制不同K值对应的损失函数值

    -   轮廓系数（Silhouette Score）：衡量簇内紧密度和簇间分离度

    -   间隙统计量（Gap Statistic）：比较实际数据与参考数据的聚类质量

-   **距离度量**

    -   欧几里得距离：$\|x - y\|_2 = \sqrt{\sum_{i=1}^d (x_i - y_i)^2}$

    -   曼哈顿距离：$\|x - y\|_1 = \sum_{i=1}^d |x_i - y_i|$

    -   余弦相似度：$\cos(\theta) = \frac{x \cdot y}{\|x\|\|y\|}$

-   **初始化方法**

    -   随机初始化：多次运行取最优结果

    -   K-means++：基于概率的智能初始化

    -   基于先验知识的初始化

- **算法变体与改进**

-   **K-means++**

    -   改进初始质心选择，提高收敛速度和结果质量

    -   初始化步骤：

        1.  随机选择第一个质心

        2.  根据与已有质心的距离概率选择后续质心

        3.  重复直到选择K个质心

-   **Mini-Batch K-means**

    -   使用数据子集进行迭代，提高大规模数据处理效率

    -   每次迭代随机选择小批量样本更新质心

- **优势与局限性**

-   **优势：** 算法简单，易实现理解；计算效率高，适合大规模数据；结果可解释性强；对数据分布假设较少
-   **局限性：** 需预先指定K值；对初始质心选择敏感；对异常值敏感；假设簇为凸形且各向同性；可能收敛到局部最优

## UMAP

UMAP（Uniform Manifold Approximation and Projection）是一种非线性降维算法，用于高维数据可视化和特征压缩。

- **原理与公式**

UMAP 基于**流形假设**：高维数据位于低维流形上。

1. **高维邻域图构建**

对于每个样本点 \(i\)，计算其到每个邻居 \(j\) 的距离 \(d_{ij}\)。使用指数核将距离转化为邻近概率：

\[
p_{j|i} = \exp\Bigg(- \frac{\max(0, d_{ij} - \rho_i)}{\sigma_i} \Bigg)
\]

其中：

- \(\rho_i\) 是点 \(i\) 最近邻的距离，保证每个点至少有一个邻居
- \(\sigma_i\) 控制局部邻域大小，使每个点的有效邻居数为 `n_neighbors`

最终高维空间的对称邻接概率：

\[
p_{ij} = p_{j|i} + p_{i|j} - p_{j|i} \cdot p_{i|j}
\]

2. **低维嵌入优化**

在低维空间中，每对点 \(i,j\) 用距离 \(||y_i - y_j||\) 计算相似度：

\[
q_{ij} = \frac{1}{1 + a ||y_i - y_j||^{2b}}
\]

其中 \(a,b\) 为预先拟合超参数。UMAP 通过最小化交叉熵损失优化低维嵌入：

\[
\mathcal{L} = \sum_{i\neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}} + (1-p_{ij}) \log \frac{1 - p_{ij}}{1 - q_{ij}}
\]

## HDBSCAN

HDBSCAN（Hierarchical Density-Based Spatial Clustering of Applications with Noise）是 DBSCAN 的层次化扩展。它不是用单一阈值直接划分簇，而是先在不同密度尺度下构建层次结构，再从中选出最稳定的簇。

* **核心思想**

  * 原始距离经过密度修正后，转化为“互相可达距离”

  * 基于互相可达距离构造最小生成树（MST）

  * 在不同密度水平下切分生成树，得到层次聚类结构

  * 根据簇的稳定性选择最终聚类结果，并将不稳定点标记为噪声

* **核心距离**

  * 定义：
    $\operatorname{core}_k(x_i)=d\bigl(x_i,x_i^{(k)}\bigr)$

  * 其中  $x_i^{(k)}$  表示点  $x_i$  的第  $k$  个近邻，且  $k=\text{min\_samples}$

  * 含义：核心距离用于衡量点  $x_i$  周围的局部密度
    核心距离越小，说明该点附近更容易找到  $k$  个邻居，局部区域越稠密；核心距离越大，则说明该点位于更稀疏的区域

* **互相可达距离**

  * 定义：  $d_{\mathrm{mreach}}(x_i,x_j)=\max\{\operatorname{core}_k(x_i),\operatorname{core}_k(x_j),d(x_i,x_j)\}$

  * 含义：两个点之间是否容易连接，不仅取决于它们的几何距离，还取决于它们各自所在区域的密度

  * 作用：互相可达距离在原始距离基础上加入了“密度惩罚”，从而把“几何接近”转化为“密度意义下接近”

* **最小生成树构造**

  * 将每个样本点  $x_i$  看作图中的一个顶点

  * 任意两点之间连边，边权定义为：
    $w_{ij}=d_{\mathrm{mreach}}(x_i,x_j)$

  * 在该加权图上构造最小生成树：
    $T=\arg\min_{T'}\sum_{(i,j)\in T'}w_{ij}$

  * 其中  $T'$  表示所有能够连接全部顶点且不含回路的生成树之一

  * 作用：最小生成树保留了样本之间的全局连接骨架，同时避免保留全部边带来的冗余结构

* **层次聚类结构**

  * 定义密度参数：
    $\lambda=\frac{1}{d_{\mathrm{mreach}}}$

  * 当  $d_{\mathrm{mreach}}$  越小时，表示连接越紧密，因此  $\lambda$  越大，对应更高密度

  * HDBSCAN 会从高密度到低密度逐步观察聚类结构变化：

    * 在高密度水平下，只有非常紧密的小群体可以保持连通

    * 随着密度阈值降低，一些小群体会逐渐扩张

    * 继续降低阈值后，不同群体之间可能发生合并

  * 这一过程形成了 HDBSCAN 的层次聚类树，也是其 “Hierarchical” 的来源

* **簇稳定性选择**

  * 稳定性定义：
    $\operatorname{Stability}(C)=\sum_{x_i\in C}\bigl(\lambda_i-\lambda_{\mathrm{birth}}(C)\bigr)$

  * 其中：

    * $\lambda_{\mathrm{birth}}(C)$ ：簇  $C$  出现时的密度水平

    * $\lambda_i$ ：样本点  $x_i$  在该簇中能够持续存在的密度水平

  * 含义：如果一个簇包含的点较多，并且在较大的密度范围内都能持续存在，那么它的稳定性就更高

  * 最终 HDBSCAN 会优先选择稳定性高的簇，不够稳定的点群会被舍弃，其中部分样本会被标记为噪声

* **聚类流程**

  * 原始距离  $\rightarrow$  核心距离  $\rightarrow$  互相可达距离    $\rightarrow$  最小生成树  $\rightarrow$  层次聚类树  $\rightarrow$  按稳定性选簇

  * 各步骤作用如下：

    * 核心距离：估计局部密度

    * 互相可达距离：将密度信息融入点间距离

    * 最小生成树：提取全局连接骨架

    * 层次聚类树：描述不同密度尺度下的簇结构

    * 稳定性选择：筛选真正可靠的簇

* **相比 DBSCAN 的优势**

  * DBSCAN 使用固定邻域阈值  $\varepsilon$ ：
    $N_{\varepsilon}(x_i)={x_j\mid d(x_i,x_j)\le\varepsilon}$

  * 这意味着 DBSCAN 只在单一密度尺度下进行聚类

  * 当数据中同时存在高密度簇和低密度簇时，单一  $\varepsilon$  往往难以兼顾所有结构

  * HDBSCAN 不预先固定  $\varepsilon$ ，而是考察所有密度尺度，再从中选择最稳定的簇，因此更适合处理不同密度分布的数据

* **主要参数**

  * `min_samples`：决定核心距离中的  $k$ ，控制局部密度的定义方式

  * `min_cluster_size`：限制一个簇至少包含多少个样本点

  * `metric`：指定样本之间的距离度量方式

  * `cluster_selection_method`：指定从层次树中选取最终簇的方式

* **优缺点**
  * 优点：不需预设簇数；能处理不同密度簇；自动识别噪声点；适合复杂形状簇
  * 缺点：原理更复杂；参数影响结果；大规模数据计算开销大

## 主成分分析（PCA）

PCA是一种线性降维方法，通过正交变换将相关变量转换为线性不相关的主成分变量。

-   **数学原理**

    -   数据中心化：  $\mathbf{X}_{centered} = \mathbf{X} - \mathbf{1}\bar{\mathbf{x}}^T$

    -   协方差矩阵计算：  $\mathbf{C} = \frac{1}{n-1}\mathbf{X}_{centered}^T\mathbf{X}_{centered}$

    -   特征分解：  $\mathbf{C}\mathbf{v}_i = \lambda_i\mathbf{v}_i$

    -   主成分方向：按特征值  $\lambda_i$  从大到小排序的特征向量  $\mathbf{v}_i$

-   **优化目标**

    $$\max_{\mathbf{w}} \mathbf{w}^T\mathbf{C}\mathbf{w} \quad \text{s.t.} \quad \|\mathbf{w}\| = 1$$
    该优化问题等价于寻找最大方差投影方向

-   **主成分计算** 第  $i$  个主成分得分：

    $$\mathbf{z}_i = \mathbf{X}_{centered}\mathbf{v}_i$$

    保留前  $k$  个主成分的近似重构：

    $$\hat{\mathbf{X}} = \sum_{i=1}^k \mathbf{z}_i\mathbf{v}_i^T + \mathbf{1}\bar{\mathbf{x}}^T$$

-   **方差解释率** 第  $i$  个主成分的方差贡献率：  $r_i = \frac{\lambda_i}{\sum_{j=1}^p \lambda_j}$  累计方差解释率：

    $$R_k = \sum_{i=1}^k r_i$$

## 核主成分分析（KPCA）

KPCA是PCA的非线性扩展，通过核技巧将数据映射到高维特征空间后进行线性PCA。

-   **核方法原理**

    -   非线性映射：$\phi: \mathbb{R}^d \to \mathcal{F}$

    -   核函数：$K(\mathbf{x}_i, \mathbf{x}_j) = \langle\phi(\mathbf{x}_i), \phi(\mathbf{x}_j)\rangle$

    -   避免显式计算高维映射$\phi(\mathbf{x})$

-   **核矩阵中心化** 中心化核矩阵：
    $$\tilde{K} = K - \mathbf{1}_N K - K \mathbf{1}_N + \mathbf{1}_N K \mathbf{1}_N$$
    其中$\mathbf{1}_N$为全1矩阵

-   **特征分解** 求解核矩阵的特征问题：
    $$\tilde{K}\boldsymbol{\alpha}_i = \lambda_i\boldsymbol{\alpha}_i$$
    归一化特征向量：$\boldsymbol{\alpha}_i \leftarrow \boldsymbol{\alpha}_i / \sqrt{\lambda_i}$

-   **核主成分计算** 新样本$\mathbf{x}$在第$i$个核主成分上的投影：
    $$z_i(\mathbf{x}) = \sum_{j=1}^n \alpha_{ij} K(\mathbf{x}_j, \mathbf{x})$$

- **常用核函数**

-   **线性核**（退化为标准PCA）：
    $$K(\mathbf{x}, \mathbf{y}) = \mathbf{x}^T\mathbf{y}$$

-   **多项式核**：
    $$K(\mathbf{x}, \mathbf{y}) = (\gamma\mathbf{x}^T\mathbf{y} + r)^d$$

-   **高斯径向基（RBF）核**：
    $$K(\mathbf{x}, \mathbf{y}) = \exp\left(-\gamma\|\mathbf{x} - \mathbf{y}\|^2\right)$$

-   **Sigmoid核**：
    $$K(\mathbf{x}, \mathbf{y}) = \tanh(\gamma\mathbf{x}^T\mathbf{y} + r)$$

- **算法实现步骤**

1.  **数据预处理**

    -   中心化：$\mathbf{X} \leftarrow \mathbf{X} - \bar{\mathbf{x}}$

    -   标准化：$\mathbf{X} \leftarrow \mathbf{X} / \boldsymbol{\sigma}$（可选）

2.  **PCA实现**

    1.  计算协方差矩阵$\mathbf{C}$

    2.  特征分解：$\mathbf{C}\mathbf{V} = \mathbf{V}\boldsymbol{\Lambda}$

    3.  选择前$k$个特征向量$\mathbf{V}_k$

    4.  投影：$\mathbf{Z} = \mathbf{X}\mathbf{V}_k$

3.  **KPCA实现**

    1.  计算核矩阵$K$

    2.  中心化核矩阵$\tilde{K}$

    3.  特征分解：$\tilde{K}\boldsymbol{\alpha} = \boldsymbol{\alpha}\boldsymbol{\Lambda}$

    4.  选择前$k$个特征向量$\boldsymbol{\alpha}_k$

    5.  新样本投影：$z_i(\mathbf{x}) = \sum_j \alpha_{ij} K(\mathbf{x}_j, \mathbf{x})$

- **优势与局限性**

-   **PCA优势：** 计算效率高，有解析解；保持最大方差，信息损失最小；去除特征间相关性；可解释性强
-   **PCA局限性：** 线性假设，无法处理非线性结构；对缩放敏感，需标准化；方差大≠信息量大（可能受噪声影响）
-   **KPCA优势：** 处理非线性数据结构；保持PCA理论框架和优点；通过核技巧避免显式高维计算
-   **KPCA局限性：** 计算复杂度高（$O(n^3)$）；核函数和参数选择困难；可解释性降低；需存储所有训练样本

# 自监督学习算法：生成建模  

## 自编码器（Autoencoder）

自编码器（Autoencoder）是一种特殊的人工神经网络架构，主要用于无监督学习任务[(尽管他属于监督学习，但由于他是用x拟合x,没有正常意义上的'用x拟合y'，因此看作无监督学习也是可以的)]，其核心目标是学习数据的有效表示（编码）。它通过尝试将输入复制到输出来学习数据的压缩表示，在这个过程中发现数据的重要特征和结构。

-   **基本架构与工作原理**
    自编码器由三个主要组件构成：编码器（Encoder）、潜在表示（Latent
    Representation）和解码器（Decoder）。[（注意，这里用encoder、decoder只是为了区分，和llms的encoder、decoder不是一个概念）]

    -   **编码器：** 将输入数据映射到潜在空间（编码）
        $$\mathbf{h} = f(\mathbf{x}) = \sigma(\mathbf{W}_e \mathbf{x} + \mathbf{b}_e)$$
        其中 $\mathbf{h}$ 是编码后的潜在表示，$\mathbf{W}_e$ 和
        $\mathbf{b}_e$ 是编码器的权重和偏置。

    -   **潜在表示：** 也称为瓶颈层（bottleneck），是数据的压缩表示
        $$\text{dim}(\mathbf{h}) \ll \text{dim}(\mathbf{x})$$
        这种维度压缩迫使网络学习数据中最重要的特征。

    -   **解码器：** 从潜在表示重构原始输入
        $$\hat{\mathbf{x}} = g(\mathbf{h}) = \sigma(\mathbf{W}_d \mathbf{h} + \mathbf{b}_d)$$
        其中 $\hat{\mathbf{x}}$ 是重构的输出，$\mathbf{W}_d$ 和
        $\mathbf{b}_d$ 是解码器的权重和偏置。

-   **损失函数与训练目标**

    自编码器通过最小化重构误差来学习参数：

    $$\mathcal{L}(\mathbf{x}, \hat{\mathbf{x}}) = \|\mathbf{x} - \hat{\mathbf{x}}\|^2$$

    对于不同的数据和应用场景，可以选择不同的损失函数：

    -   **均方误差（MSE）：** 适用于连续数据

    -   **二元交叉熵：** 适用于二值化数据

-   **主要变体与应用**

    -   **欠完备自编码器（Undercomplete Autoencoder）**

        -   潜在空间的维度小于输入空间

        -   通过瓶颈约束迫使学习数据的重要特征

        -   主要用于特征学习和降维

    -   **正则化自编码器（Regularized Autoencoder）**

        -   潜在空间维度可以等于或大于输入空间

        -   通过正则化项防止过拟合，确保泛化能力

        -   包括稀疏自编码器、去噪自编码器等变体

    -   **去噪自编码器（Denoising Autoencoder, DAE）**

        -   输入被添加噪声：$\tilde{\mathbf{x}} = \text{corrupt}(\mathbf{x})$

        -   训练目标：从带噪声输入重构原始干净数据

        -   学习到的特征对噪声和损坏具有鲁棒性
    
    -   **卷积自编码器（Convolutional Autoencoder）**

        -   编码器使用卷积层，解码器使用转置卷积层

        -   特别适合图像数据，能够捕获空间局部模式

        -   在图像去噪、超分辨率等任务中表现优异

## 变分自编码器（VAE）

-  **基本思想**

    - VAE 是一种结合了神经网络与概率图模型思想的生成模型，本质上不是直接学习“把输入压缩成一个确定的向量”，而是学习输入数据在潜在空间中的**概率分布**。

    - 与普通自编码器不同，普通自编码器编码器输出的是一个确定性的隐向量  $\mathbf{h}$  ，而 VAE 编码器输出的是潜变量分布的参数，通常是均值向量  $\boldsymbol{\mu}$ 和方差向量  $\boldsymbol{\sigma}^2$：

        $$\boldsymbol{\mu} = f_{\mu}(\mathbf{x}), \qquad \log \boldsymbol{\sigma}^2 = f_{\sigma}(\mathbf{x})$$

    - 也就是说，编码器学习的是近似后验分布：

        $$q_{\phi}(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}, {diag}(\boldsymbol{\sigma}^2))$$

        其中  $\mathbf{z}$  是潜变量，  $\phi$  表示编码器参数。

- **潜变量采样与重参数化技巧**

    - 如果直接从分布  $q_{\phi}(\mathbf{z}|\mathbf{x})$ 中采样，会导致梯度无法直接反向传播，因此 VAE 使用**重参数化技巧（Reparameterization Trick）**：

        $$\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}), \qquad\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$$

        其中  $\odot$ 表示逐元素乘法。

    - 这样就把随机性从  $\mathbf{z}$  转移到了独立噪声   $\boldsymbol{\epsilon}$  上，使得  $\boldsymbol{\mu}$  和  $\boldsymbol{\sigma}$  仍然可以通过梯度下降进行优化。

- **解码器与生成过程**

    - 解码器接收潜变量  $\mathbf{z}$  ，并输出对原始数据的重构分布：    $p_{\theta}(\mathbf{x}\mid\mathbf{z})$

    - 对于给定输入  $\mathbf{x}$，VAE 的整体生成过程可以理解为：

        1. 由编码器得到潜变量分布参数  $\boldsymbol{\mu}, \boldsymbol{\sigma}^2$
        2. 从该分布中采样潜变量  $\mathbf{z}$
        3. 由解码器根据  $\mathbf{z}$ 重构输入  $\hat{\mathbf{x}}$

    - 在生成新样本时，不需要输入真实数据，只需从先验分布中采样：

        $$ \mathbf{z} \sim p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I}) $$

    - 然后送入解码器生成新样本：

        $$ \hat{\mathbf{x}} \sim p_{\theta}(\mathbf{x}|\mathbf{z}) $$

- **训练目标与损失函数**

    - VAE 的目标是最大化数据对数似然  $\log p(\mathbf{x})$，但该项通常难以直接计算，因此实际优化的是其变分下界（ELBO, Evidence Lower Bound）：

        $$\mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})}[\log p_{\theta}(\mathbf{x}|\mathbf{z})]$$
        $$D_{\mathrm{KL}}\left(q_{\phi}(\mathbf{z}|\mathbf{x}) | p(\mathbf{z})\right)$$

    - 其中两部分含义分别为：

    * **重构项（Reconstruction Term）**
      衡量解码器根据潜变量  $\mathbf{z}$ 重构原始输入的能力，鼓励生成结果尽可能接近输入数据。

    * **KL 散度项（Regularization Term）**
      衡量编码器输出的近似后验分布  $q_{\phi}(\mathbf{z}|\mathbf{x})$  与先验分布  $p(\mathbf{z})$ 的差异，约束潜在空间分布尽量接近标准正态分布，从而使潜在空间更平滑、可采样。

    - 在实际训练中，常将损失写成最小化形式：

        $$ -\mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})}[\log p_{\theta}(\mathbf{x}|\mathbf{z})] + D_{\mathrm{KL}}\left(q_{\phi}(\mathbf{z}|\mathbf{x}) | p(\mathbf{z})\right) $$

    - 若采用  $\beta$-VAE，还可以对 KL 项加权，以增强潜在表示的可解释性和解耦性：

        $$ -\mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})}[\log p_{\theta}(\mathbf{x}|\mathbf{z})] + \beta , D_{\mathrm{KL}}\left(q_{\phi}(\mathbf{z}|\mathbf{x}) | p(\mathbf{z})\right) $$

- **VAE 的特点**

    - 相比普通自编码器，VAE 学到的不是单个确定点，而是潜在空间中的连续概率分布，因此具有更好的生成能力。

    - 由于潜在空间被约束为接近标准正态分布，不同样本之间的潜在表示通常更加平滑，便于进行插值、采样和生成。

-   **应用领域**

    -   **特征学习与降维：** 学习数据的低维表示，替代PCA

    -   **异常检测：** 正常数据重构误差小，异常数据重构误差大

    -   **数据去噪：** 去除图像、音频等数据中的噪声

    -   **数据生成：** 特别是VAE可以生成新的数据样本

    -   **迁移学习：** 使用预训练的自编码器作为特征提取器


## 扩散模型（Diffusion Models）

扩散模型的核心是：把复杂数据分布的生成问题，拆成很多个“逐步去噪”的子问题。训练时学习反向去噪过程，采样时从高斯噪声逐步还原样本。

- **1.建模框架（前向加噪 + 反向去噪）**

- 记号：$x_0\sim q_{data}(x_0)$，时间步 $t=1,\dots,T$，噪声日程 $\beta_t\in(0,1)$

$$
\alpha_t=1-\beta_t,\qquad \bar\alpha_t=\prod_{s=1}^t\alpha_s
$$

- 前向加噪（固定，不训练）：

$$
q(x_t|x_{t-1})=\mathcal N\bigl(x_t;\sqrt{\alpha_t}x_{t-1},(1-\alpha_t)\mathbf I\bigr)
$$

$$
x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\,\epsilon_t,\quad \epsilon_t\sim\mathcal N(0,\mathbf I)
$$

- 递推展开后得到闭式（训练关键）：

$$
q(x_t|x_0)=\mathcal N\bigl(x_t;\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)\mathbf I\bigr)
$$

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\epsilon,\\ \epsilon\sim\mathcal N(0,\mathbf I)
$$

- 反向生成（需要学习）：

$$
p_\theta(x_{0:T})=p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}|x_t),\qquad p(x_T)=\mathcal N(0,\mathbf I)
$$

$$
p_\theta(x_{t-1}|x_t)=\mathcal N\bigl(x_{t-1};\mu_\theta(x_t,t),\Sigma_\theta(x_t,t)\bigr)
$$

- 真实后验（DDPM 推导枢纽）：

$$
q(x_{t-1}|x_t,x_0)=\mathcal N\bigl(x_{t-1};\tilde\mu_t(x_t,x_0),\tilde\beta_t\mathbf I\bigr)
$$

$$
\tilde\mu_t=
\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0
+
\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t,
\qquad
\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t

- **2.损失函数与参数化**

- 训练从似然下界（ELBO）出发：

$$
\mathcal L_{VLB}=\mathbb E_q\Bigl[
D_{KL}(q(x_T|x_0)\|p(x_T))
+\sum_{t=2}^T D_{KL}(q(x_{t-1}|x_t,x_0)\|p_\theta(x_{t-1}|x_t))
-\log p_\theta(x_0|x_1)
\Bigr]
$$

- DDPM 常见做法：固定（或半固定）方差，主要学习均值；用噪声预测参数化：

$$
\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right)
$$

$$
\mathcal L_{simple}=\mathbb E_{x_0,\epsilon,t}\left[\|\epsilon-\epsilon_\theta(x_t,t)\|_2^2\right]
$$

- 三种主流等价参数化：

$$
\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\hat\epsilon}{\sqrt{\bar\alpha_t}},
\qquad
\hat\epsilon=\frac{x_t-\sqrt{\bar\alpha_t}\,\hat x_0}{\sqrt{1-\bar\alpha_t}}
$$

$$
v=\sqrt{\bar\alpha_t}\,\epsilon-\sqrt{1-\bar\alpha_t}\,x_0
$$

其中 $\epsilon$-pred 最常见，$v$-pred 在大模型里常用于平衡不同噪声区间的训练。

- **3.采样方法与主流变体（DDPM / DDIM / SDE）**

- DDPM 祖先采样（随机）：

$$
x_{t-1}=\frac{1}{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right)
+\sigma_t z,
\quad z\sim\mathcal N(0,\mathbf I)
$$

- DDIM（可确定性、可跳步）：

$$
x_{t-1}=\sqrt{\bar\alpha_{t-1}}\,\hat x_0
+\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\,\hat\epsilon
+\sigma_t z
$$

$$
\sigma_t=\eta\sqrt{\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}}\sqrt{1-\frac{\bar\alpha_t}{\bar\alpha_{t-1}}}
$$

$\eta=0$ 时为确定性采样，推理速度通常更快。

- Score/SDE 统一视角：

$$
s_\theta(x_t,t)=\nabla_{x_t}\log q_t(x_t)
= -\frac{1}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)
$$

VP-SDE：

$$
dx = -\frac{1}{2}\beta(t)x\,dt + \sqrt{\beta(t)}\,dw
$$

其反向 SDE 与概率流 ODE：

$$
dx = \left[-\frac{1}{2}\beta(t)x - \beta(t)\nabla_x\log p_t(x)\right]dt + \sqrt{\beta(t)}\,d\bar w
$$

$$
\frac{dx}{dt}= -\frac{1}{2}\beta(t)x - \frac{1}{2}\beta(t)\nabla_x\log p_t(x)

- **4.训练策略与工程实践（条件控制 + LDM）**

- 噪声日程常见有线性与 cosine，可用SNR分析损失加权与采样步长分配。

    $$\text{SNR}(t)=\frac{\bar\alpha_t}{1-\bar\alpha_t}$$

- 条件引导：
  - Classifier Guidance：

    $$\nabla_{x_t}\log p(x_t|y)=\nabla_{x_t}\log p(x_t)+\nabla_{x_t}\log p_\phi(y|x_t)$$

  - Classifier-Free Guidance（CFG）：

    $$\hat\epsilon_{cfg}=\hat\epsilon_{uncond}+s(\hat\epsilon_{cond}-\hat\epsilon_{uncond})$$

- Latent Diffusion（LDM）：先编码到潜空间再做扩散

    $$z_0=E(x_0),\quad \hat x_0=D(\hat z_0)$$


## 生成对抗网络（GAN）

GAN 的核心是博弈优化：判别器学习区分真伪，生成器学习“骗过”判别器。理论上可解释为分布距离最小化，工程上重点是稳定训练。

- **1.基本原理与理论推导**

- 记号：$z\sim p_z(z)$，$x_g=G(z)$，$x_g\sim p_g$，判别器 $D(x)\in(0,1)$。

- 原始 min-max 目标：

$$
\min_G\max_D V(D,G)=\mathbb E_{x\sim p_{data}}[\log D(x)]+\mathbb E_{z\sim p_z}[\log(1-D(G(z)))]
$$

- 固定生成器时，最优判别器：

$$
D^*(x)=\frac{p_{data}(x)}{p_{data}(x)+p_g(x)}
$$

- 代回后：说明理想情况下是在最小化 JS 散度，最优点为  $p_g=p_{data}$。

$$
V(D^*,G)=-\log4+2\,JS(p_{data}\|p_g)

- **2.损失函数设计**

- Vanilla BCE： `non-saturating` 形式在早期梯度更强，实践更常用。

$$
\mathcal L_D=-\mathbb E_{x\sim p_{data}}[\log D(x)]-\mathbb E_{z\sim p_z}[\log(1-D(G(z)))]
$$

$$
\mathcal L_G^{mm}=\mathbb E_z[\log(1-D(G(z)))],\qquad
\mathcal L_G^{ns}=-\mathbb E_z[\log D(G(z))]
$$

- LSGAN：

$$
\mathcal L_D^{LS}=\frac12\mathbb E_x[(D(x)-1)^2]+\frac12\mathbb E_z[D(G(z))^2],
\qquad
\mathcal L_G^{LS}=\frac12\mathbb E_z[(D(G(z))-1)^2]
$$

- Hinge GAN：

$$
\mathcal L_D^{hinge}=\mathbb E_x[\max(0,1-D(x))]+\mathbb E_z[\max(0,1+D(G(z)))]
$$

$$
\mathcal L_G^{hinge}=-\mathbb E_z[D(G(z))]
$$

- WGAN / WGAN-GP：

$$
W_1(p_{data},p_g)=\sup_{\|f\|_L\le1}\mathbb E_{x\sim p_{data}}[f(x)]-\mathbb E_{x\sim p_g}[f(x)]
$$

$$
\mathcal L_D^{GP}=
\mathbb E_{\tilde x\sim p_g}[f(\tilde x)]-\mathbb E_{x\sim p_{data}}[f(x)]
+\lambda\,\mathbb E_{\hat x}(\|\nabla_{\hat x}f(\hat x)\|_2-1)^2
$$

$$
\mathcal L_G^{W}=-\mathbb E_z[f(G(z))]

- **3.训练策略与稳定性**

- 常见不稳定来源：判别器过强、模式崩塌、博弈振荡。
- 主流技巧：
  1. 损失优先试 Hinge 或 WGAN-GP。
  2. 加正则（SN、R1）。R1 公式：

$$
\mathcal L_{R1}=\frac{\gamma}{2}\mathbb E_{x\sim p_{data}}\|\nabla_x D(x)\|_2^2
$$

  3. 控制更新比 $n_D:n_G$（如 1:1 或 5:1）。
  4. 使用更稳的网络结构（残差、注意力、归一化策略）。

- **4.条件生成与图像翻译（实战常用）**

- cGAN：

$$
\min_G\max_D
\mathbb E_{x,y\sim p_{data}}[\log D(x,y)]
+
\mathbb E_{z,y}[\log(1-D(G(z,y),y))]
$$

- Pix2Pix（配对）：

$$
\mathcal L=\mathcal L_{cGAN}+\lambda\,\mathcal L_{L1},\qquad
\mathcal L_{L1}=\mathbb E_{x,y}\|y-G(x)\|_1
$$

- CycleGAN（非配对）循环一致性：

$$
\mathcal L_{cyc}=
\mathbb E_{x\sim X}\|F(G(x))-x\|_1
+\mathbb E_{y\sim Y}\|G(F(y))-y\|_1

- **5.评估指标**

- IS：

$$
IS=\exp\bigl(\mathbb E_x D_{KL}(p(y|x)\|p(y))\bigr)
$$

- FID：

$$
FID=\|m_r-m_g\|_2^2 + {Tr}\bigl(C_r+C_g-2(C_rC_g)^{1/2}\bigr)
$$


# 强化学习基础

强化学习（Reinforcement Learning, RL）研究智能体如何在与环境交互中，通过试错最大化长期累积奖励。

## 马尔可夫决策过程（MDP）

一个标准 MDP 可表示为五元组：

$$
\mathcal M=(\mathcal S,\mathcal A,P,R,\gamma)
$$

-   $\mathcal S$ ：状态空间
-   $\mathcal A$ ：动作空间
-   $P(s'|s,a)$：状态转移概率
-   $R(s,a)$：即时奖励
-   $\gamma\in[0,1)$：折扣因子

时刻 $t$ 的回报定义为：

$$
G_t=\sum_{k=0}^{\infty}\gamma^k r_{t+k+1}
$$

若是有限步回合（episodic）问题，终止时刻为 $T$，则

$$
G_t=\sum_{k=0}^{T-t-1}\gamma^k r_{t+k+1}
$$

折扣因子 $\gamma$ 越大，模型越关注长期收益；$\gamma$ 越小，越偏向短期收益。

## 价值函数与贝尔曼方程

-   **状态价值函数：**

$$
V^\pi(s)=\mathbb E_\pi[G_t|s_t=s]
$$

-   **动作价值函数：**

$$
Q^\pi(s,a)=\mathbb E_\pi[G_t|s_t=s,a_t=a]
$$

-   **贝尔曼期望方程：**

$$
V^\pi(s)=\sum_a\pi(a|s)\sum_{s'}P(s'|s,a)\left[R(s,a)+\gamma V^\pi(s')\right]
$$

-   **贝尔曼最优方程：**

$$
V^*(s)=\max_a \sum_{s'}P(s'|s,a)\left[R(s,a)+\gamma V^*(s')\right]
$$

$$
Q^*(s,a)=\sum_{s'}P(s'|s,a)\left[R(s,a)+\gamma \max_{a'}Q^*(s',a')\right]
$$

定义时序差分误差（TD Error）：

$$
\delta_t=r_{t+1}+\gamma V(s_{t+1})-V(s_t)
$$

它衡量“当前估计”与“自举目标”的差距，是大多数 RL 更新的核心信号。

## 方法谱系：DP / MC / TD

-   **动态规划（DP）：**
    -   要求已知环境转移概率 $P(s'|s,a)$
    -   典型方法：策略迭代、价值迭代
    -   价值迭代更新：

$$
V_{k+1}(s)=\max_a\sum_{s'}P(s'|s,a)\left[R(s,a)+\gamma V_k(s')\right]
$$

-   **蒙特卡洛（MC）：**
    -   不需要环境模型
    -   用完整回合回报更新价值，方差较大但无 bootstrap 偏差
    -   增量式更新：

$$
V(s)\leftarrow V(s)+\alpha\left(G_t-V(s)\right)
$$

-   **时序差分（TD）：**
    -   不需要完整回合，可在线更新
    -   用一步目标自举，方差更低、学习更快
    -   TD(0) 更新：

$$
V(s_t)\leftarrow V(s_t)+\alpha\left[r_{t+1}+\gamma V(s_{t+1})-V(s_t)\right]
$$

## 经典控制方法：Q-Learning 与 SARSA

### Q-Learning（离策略）

$$
Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha\left[r_{t+1}+\gamma\max_{a'}Q(s_{t+1},a')-Q(s_t,a_t)\right]
$$

-   使用下一状态的贪心价值作为目标（off-policy）。
-   目标策略是贪心策略，行为策略通常是 $\epsilon$-greedy。

### SARSA（在策略）

$$
Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha\left[r_{t+1}+\gamma Q(s_{t+1},a_{t+1})-Q(s_t,a_t)\right]
$$

-   使用实际采样到的下一动作（on-policy），通常更保守稳定。

- **Q-Learning 与 SARSA 的差异**

-   Q-Learning 的目标是 $\max_{a'}Q(s_{t+1},a')$，更激进，学习最优策略速度常更快。
-   SARSA 的目标是 $Q(s_{t+1},a_{t+1})$，考虑了行为策略探索噪声，安全约束任务更常用。

## 策略梯度与 Actor-Critic

### 策略梯度（Policy Gradient）

直接优化参数化策略 $\pi_\theta(a|s)$：

$$
J(\theta)=\mathbb E_{\tau\sim\pi_\theta}\left[\sum_t r_t\right]
$$

$$
\nabla_\theta J(\theta)=\mathbb E_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a_t|s_t)\, G_t\right]
$$

常见改进是用优势函数 $A(s,a)=Q(s,a)-V(s)$ 来降低方差，即

$$
\nabla_\theta J(\theta)\approx
\mathbb E\left[\nabla_\theta\log\pi_\theta(a_t|s_t)\,\hat A_t\right]
$$

并使用 baseline（如 $V(s_t)$）降低梯度估计方差而不引入偏差。

### Actor-Critic

-   Actor：更新策略 $\pi_\theta$
-   Critic：估计价值函数 $V_\phi$ 或 $Q_\phi$
-   通过 Critic 提供低方差学习信号，提高训练效率

常见一阶更新形式：

$$
\theta\leftarrow \theta+\alpha_\pi \nabla_\theta\log\pi_\theta(a_t|s_t)\,\hat A_t
$$

$$
\phi\leftarrow \phi-\alpha_v \nabla_\phi\left(V_\phi(s_t)-\hat V_t^{target}\right)^2
$$

## 深度强化学习（DQN / PPO / SAC）

### DQN（值函数方法，离散动作）

用神经网络 $Q_\theta(s,a)$ 近似动作价值函数，最小化 TD 损失：

$$
y_t=r_{t+1}+\gamma \max_{a'}Q_{\theta^-}(s_{t+1},a')
$$

$$
\mathcal L(\theta)=\mathbb E\left[(y_t-Q_\theta(s_t,a_t))^2\right]
$$

其中 $\theta^-$ 是目标网络参数，周期性从 $\theta$ 同步，用于稳定训练。  
经验回放（Replay Buffer）通过随机采样打破样本相关性、提升样本利用率。

### PPO（策略优化，on-policy）

定义概率比：

$$
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

裁剪目标：

$$
\mathcal L^{clip}(\theta)=
\mathbb E\left[
\min\left(r_t(\theta)\hat A_t,\;
\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t\right)
\right]
$$

通过限制策略步长，避免一次更新过大导致性能崩塌；工程上鲁棒且调参相对容易。

### SAC（最大熵 Actor-Critic，off-policy）

在奖励之外加入策略熵，鼓励探索：

$$
J(\pi)=\sum_t \mathbb E_{(s_t,a_t)\sim \rho_\pi}\left[r(s_t,a_t)+\alpha \mathcal H(\pi(\cdot|s_t))\right]
$$

其中 $\alpha$ 为温度系数，控制“奖励最大化”和“熵最大化”的权衡。  
SAC 常用于连续控制任务，样本效率高、稳定性好。

## 探索-利用权衡

-   **$\epsilon$-greedy：** 以概率 $\epsilon$ 随机探索，否则选 $\arg\max_a Q(s,a)$。
-   **退火策略：** $\epsilon$ 随训练逐步下降（如线性或指数衰减）。
-   **熵正则：** 在目标中加入 $\alpha \mathcal H(\pi(\cdot|s))$ 保持策略随机性。
-   **UCB（Bandit）：**

$$
a_t=\arg\max_a\left(\hat\mu_a + c\sqrt{\frac{\ln t}{N_t(a)}}\right)
$$

其中 $N_t(a)$ 为动作被选择次数，不确定性高的动作会获得探索奖励。

## 离线强化学习（Offline RL）基础

离线 RL 只使用固定历史数据集 $\mathcal D=\{(s,a,r,s')\}$ 训练，不再与环境交互。

-   **核心难点：** 分布外动作（OOD action）导致 Q 值过估计。
-   **常见约束思路：**
    -   行为克隆约束策略不要偏离数据分布太远
    -   保守 Q 学习，惩罚对未见动作的过高估计
    -   支持集约束（只在数据分布附近优化）

## RL 评估与实验规范

-   **在线评估：** 在独立随机种子环境中运行多回合，报告平均回报和标准差。
-   **离线评估：** 可用 OPE（Off-Policy Evaluation）估计新策略价值。
-   **常见 OPE 方法：**
    -   Importance Sampling（IS）
    -   Weighted IS
    -   Doubly Robust（DR）

-   **训练曲线解读：**
    -   看收敛速度（sample efficiency）
    -   看稳定性（是否高频震荡/崩塌）
    -   看最终性能方差（多 seed）

## 实践注意事项

-   奖励设计要避免“奖励黑客”（模型钻规则漏洞）。
-   训练与评估环境要分离，防止对特定随机种子过拟合。
-   关注样本效率、稳定性与安全约束（动作约束、风险惩罚）。
-   离线 RL 需重点控制分布外动作带来的估计偏差。
-   强化学习实验可复现性要记录：环境版本、seed、超参数、评估协议。
