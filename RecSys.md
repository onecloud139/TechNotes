# 一、推荐系统基本知识

这一章只讲“跨召回/排序/重排都要用到的通用知识”。后面三章会讲各模块特有框架，第五章再讲具体模型。

## 1.1 推荐系统的本质：受约束的在线决策

推荐系统不是离线分类器，而是在线决策系统。每次请求都要在有限时延内输出一个列表，并且同时满足用户目标、业务目标和系统约束。

### 1.1.1 问题形式化

给定用户 $u$、上下文 $c$、候选集 $\mathcal{I}$，输出列表 $\pi=(i_1,\dots,i_K)$：

$$
\pi^* = \arg\max_{\pi \in \Pi(\mathcal{I})}\;\mathbb{E}[R(\pi\mid u,c)]
$$

其中 $R$ 是列表收益函数，通常是多目标组合：

$$
R = w_{ctr}\cdot CTR + w_{cvr}\cdot CVR + w_{wt}\cdot WT + w_{rev}\cdot REV
$$

如果你只优化单一指标（比如 CTR），经常会牺牲长期留存或内容生态健康。

### 1.1.2 为什么是“列表优化”不是“单项预测”

即使单个 item 的点击概率估计很准，也不代表整页体验好。因为列表还涉及：

- 同质化（10 个类似内容）
- 曝光公平（头部垄断）
- 新颖性（探索不足）
- 业务约束（广告位、库存、合规）

所以推荐的最终优化对象是“页面级结果”，不是单 item 分数。

### 1.1.3 三类核心约束

- 时延约束：通常毫秒级预算，不能超时。
- 稳定性约束：流量峰值不崩、结果分布不剧烈抖动。
- 合规约束：内容安全、隐私、行业监管要求。

## 1.2 分阶段架构与模块边界

推荐系统通用链路：

```text
召回（Recall） -> 排序（Ranking） -> 重排（Re-ranking）
```

### 1.2.1 每层职责（必须边界清晰）

- 召回：从千万/亿级物品中快速取几百~几千候选，目标是“高覆盖 + 低时延”。
- 排序：在候选集合上做更准确的效用估计，目标是“相关性和收益”。
- 重排：做列表级约束优化，目标是“体验、生态、商业平衡”。

### 1.2.2 输入输出契约（工程里必须定义）

每层都要有稳定契约：

- 输入字段版本（user/context/feature schema）
- 输出字段定义（score、source、explain、trace_id）
- 异常行为（超时、空结果、降级）

如果契约不清楚，线上问题会很难排查。

### 1.2.3 为什么不能“一层做完”

- 全库精排太贵：复杂模型无法全量逐项打分。
- 召回和排序的优化目标不同：召回更偏覆盖，排序更偏精度。
- 重排处理的是列表结构问题，排序分数本身无法完全表达。

## 1.3 数据体系、特征体系与样本构造

这部分是全链路共享地基，做错了后面都不稳。

### 1.3.1 事件日志最小单元

推荐日志建议至少包含：

```text
(request_id, user_id, item_id, action, timestamp, position, scene, device, source)
```

常见动作链：

$$
\text{曝光} \rightarrow \text{点击} \rightarrow \text{深度行为} \rightarrow \text{转化}
$$

### 1.3.2 时间口径（最容易踩坑）

至少区分三类时间：

- event_time：行为真实发生时间
- process_time：日志入仓时间
- feature_time：特征快照时间

训练时必须满足：

$$
feature\_time \le label\_time
$$

否则就是信息泄漏（未来信息穿越）。

### 1.3.3 标签构造

标签不是“直接拿动作”，而是要定义窗口和归因规则。

- 观测窗口：例如用最近 14 天训练。
- 标签窗口：例如曝光后 24 小时点击记正样本。
- 延迟标签：转化和留存需要回填。

### 1.3.4 负样本来源（全链路通用）

常见负样本：

- 曝光未点（最常用）
- 随机负采样（缓解类不平衡）
- 难负样本（提升区分能力）

实际训练常组合使用，并记录来源字段，便于调参与解释。

### 1.3.5 特征一致性

全链路要保证“离线训练特征”和“线上推理特征”一致。

常见问题：

- 离线用全量统计，线上只有实时窗口
- 字段填充规则不同
- 类别编码字典版本不一致

这类问题会导致“离线很好，线上掉线”。

### 1.3.6 样本加权与再平衡（实操）

推荐任务普遍存在正负极度不平衡，可通过样本权重缓解：

$$
\mathcal{L}=\frac{1}{N}\sum_{i=1}^{N} w_i\cdot \ell(\hat y_i,y_i)
$$

常见权重来源：

- 类别不平衡权重（正样本更高权重）
- 行为强度权重（购买 > 点击 > 曝光）
- 新鲜度权重（近期样本更高）

时间衰减权重常见形式：

$$
w_i=\exp(-\lambda \Delta t_i)
$$

其中 $\\Delta t_i$ 是样本距当前时间的间隔。

### 1.3.7 数据质量清单（必须落地）

每次训练前建议固定检查：

- 主键完整性：`request_id/user_id/item_id` 缺失率
- 事件时序：`event_time` 乱序比例
- 标签异常：CTR/CVR 是否突变
- 特征异常：空值率、极值率、枚举爆炸
- 分布漂移：训练窗口 vs 线上实时窗口 PSI/KL

## 1.4 分数语义、归一化与校准

“有分数”不等于“分数可直接比较”。

### 1.4.1 三类分数语义

- 检索分（召回）：相似度/距离，跨路通常不可比。
- 预测分（排序）：概率或效用估计。
- 策略分（重排）：相关性 + 约束项组合。

### 1.4.2 跨路分数归一化

多路融合前常用：

- Min-Max：

$$
x' = \frac{x - x_{min}}{x_{max}-x_{min}}
$$

- Z-score：

$$
x' = \frac{x-\mu}{\sigma}
$$

- Sigmoid：

$$
x' = \frac{1}{1+e^{-x}}
$$

也可用 rank-based 归一，减少尺度漂移影响。

### 1.4.3 概率校准

如果分数要用于阈值决策、收益估计、跨场景比较，需要校准。

校准目标：

$$
P(\hat y=p)\approx P(y=1\mid \hat y=p)
$$

常见方法：

- Platt Scaling
- Isotonic Regression

常见校准评估：

- Brier Score
- ECE（Expected Calibration Error）：

$$
ECE=\sum_{m=1}^{M}\frac{|B_m|}{n}\left|acc(B_m)-conf(B_m)\right|
$$

## 1.5 评估体系：离线、在线、反事实

推荐系统评估必须三位一体：离线评估、在线实验、反事实评估。

### 1.5.1 离线指标（定义要统一）

- AUC：区分正负样本能力。
- LogLoss：概率质量。
- Recall@K：

$$
Recall@K=\frac{|Rel_u \cap TopK_u|}{|Rel_u|}
$$

- Precision@K：

$$
Precision@K=\frac{|Rel_u \cap TopK_u|}{K}
$$

- MRR：

$$
MRR=\frac{1}{|U|}\sum_{u\in U}\frac{1}{rank_u}
$$

- DCG@K / NDCG@K：

$$
DCG@K=\sum_{k=1}^{K}\frac{2^{rel_k}-1}{\log_2(k+1)}
$$

$$
NDCG@K=\frac{DCG@K}{IDCG@K}
$$

### 1.5.2 在线实验（A/B）

在线实验要同时看：

- 主指标：CTR/CVR/时长/收入
- 护栏：延迟、负反馈、投诉、崩溃率
- 长期指标：留存、复访、生命周期价值

实验要点：

- 分流一致性（用户级稳定分流）
- 样本量与检验功效
- 观察周期覆盖周内周期性
- 回滚阈值预先定义

常见显著性检验（以两组 CTR 为例）：

$$
z=\frac{\hat p_A-\hat p_B}{\sqrt{\hat p(1-\hat p)\left(\frac{1}{n_A}+\frac{1}{n_B}\right)}}
$$

其中

$$
\hat p=\frac{click_A+click_B}{n_A+n_B}
$$

再根据 $z$ 值计算 p-value 判断显著性。

效果提升建议同时报告相对提升：

$$
Lift=\frac{metric_B-metric_A}{metric_A}
$$

并给出置信区间，而不是只报单点值。

### 1.5.3 反事实评估（Off-policy）

当线上试验成本高时，需要离线估计新策略价值。

- IPS：

$$
\hat V_{IPS}=\frac{1}{n}\sum_{i=1}^{n}\frac{\pi_e(a_i\mid x_i)}{\pi_b(a_i\mid x_i)}r_i
$$

- SNIPS：

$$
\hat V_{SNIPS}=\frac{\sum_i w_i r_i}{\sum_i w_i},\;w_i=\frac{\pi_e}{\pi_b}
$$

- DR：结合模型估计与重要性加权，方差和偏差更平衡。

为避免 IPS 权重爆炸，常用截断：

$$
w_i^{clip}=\min\left(\frac{\pi_e(a_i|x_i)}{\pi_b(a_i|x_i)},c\right)
$$

截断阈值 $c$ 过小会增大偏差，过大会增大方差，需要通过离线回放调优。

## 1.6 偏差、探索与反馈闭环

推荐系统的核心风险是“被自己的历史策略困住”。

### 1.6.1 主要偏差

- 曝光偏差：仅曝光样本可观测。
- 位置偏差：前排更容易被看到。
- 选择偏差：购买往往以点击为前提。
- 流行度偏差：热门获得更多曝光，进一步变热。

位置偏差常见拆分：

$$
P(click)=P(exam)\cdot P(relevance)
$$

### 1.6.2 探索机制

全用“贪心最优”会快速陷入局部最优。

常用探索：

- epsilon-greedy
- UCB
- Thompson Sampling

三者常见形式：

- epsilon-greedy：

$$
a_t=
\begin{cases}
\text{random action}, & \text{with prob } \epsilon\\
\arg\max_a \hat Q(a), & \text{with prob } 1-\epsilon
\end{cases}
$$

- UCB：

$$
a_t=\arg\max_a\left(\hat\mu_a + c\sqrt{\frac{\ln t}{N_t(a)}}\right)
$$

- Thompson Sampling：从后验分布采样参数后选择最优动作。

推荐实践：

- 小比例探索流量（例如 1%~5%）
- 分场景探索（新用户、新内容优先）
- 对高风险内容设安全边界

### 1.6.3 反馈闭环管理

模型上线后会改变曝光分布，进而改变下一轮训练数据。

要做：

- 数据回放和版本对齐
- 分群监控（新老用户、头尾内容）
- 周期性校验“是否越来越单一”

建议加两个固定监控量：

- 曝光熵（越低越同质）：

$$
H=-\sum_{k}p_k\log p_k
$$

- 头部集中度（如 Top1% 曝光占比）

## 1.7 工程治理与可靠性

### 1.7.1 全链路版本治理

至少要有三类版本：

- 特征版本
- 模型版本
- 索引/策略版本

线上日志要可回放到“某时刻的完整配置”。

### 1.7.2 降级与容灾

典型降级链：

```text
深度召回失败 -> 退化到 CF/热门
精排超时 -> 退化到轻量模型
重排异常 -> 回退排序结果
```

### 1.7.3 监控面板最低配置

- 请求量、错误率、时延 P95/P99
- 召回覆盖率、空召回率
- 排序分数分布、校准漂移
- 重排约束命中率
- 关键业务指标与护栏

## 1.8 统一符号与口径（建议全书统一）

- $u$：用户（user）
- $i$：物品（item）
- $c$：上下文（context）
- $\\mathcal{I}$：全量候选库
- $\\mathcal{S}(u,c)$：召回候选集
- $\\pi$：最终展示列表
- $K$：展示长度（如 topK）

统一口径建议：

- CTR 口径：`click / exposure`
- CVR 口径：明确是 `purchase / click` 还是 `purchase / exposure`
- 新物品定义：按“上线天数 <= T”统一
- 长尾定义：按曝光分位（如后 80%）统一

---

# 二、召回层（Recall）

召回层的核心是“用最少计算找出尽可能多的高潜候选”。召回做得好，排序才有发挥空间。

## 2.1 召回层目标、输出与时延预算

### 2.1.1 目标三角

召回通常要平衡三件事：

- 覆盖率（Recall）
- 时延（Latency）
- 成本（Compute/Storage）

任何一项极端优化都会伤另外两项。

### 2.1.2 输出定义

召回输出不只是 item 列表，建议至少包含：

```text
(item_id, route, raw_score, route_rank, reason, request_id)
```

这样后面才可做融合解释、故障定位、策略审计。

### 2.1.3 分路预算

建议对每路明确预算：

- 候选条数上限
- 单路超时阈值
- 单路失败降级策略

没有预算管理，多路并行很容易失控。

常见时延预算拆解：

$$
T_{total}\approx \max_r(T_r)+T_{merge}+T_{dedup}+T_{serialize}
$$

其中 $T_r$ 是各路召回耗时。并行体系里瓶颈往往来自最慢通路，而不是平均通路。

吞吐容量粗估（单实例）：

$$
QPS \approx \frac{concurrency}{avg\\_latency}
$$

该估算适合做容量预估，不替代压测结果。

## 2.2 召回通路框架（不是模型细节）

这里只讲方法框架，具体模型放第五章。

### 2.2.1 协同过滤类通路

核心思想：利用历史共现。

- UserCF：相似用户迁移偏好
- ItemCF：相似物品扩展

典型相似度形式（ItemCF）：

$$
sim(i,j)=\frac{\sum_{u\in U_{ij}}w_u}{\sqrt{|U_i||U_j|}}
$$

其中 $U_{ij}$ 为同时交互过 $i,j$ 的用户集合。

### 2.2.2 向量近邻类通路

核心思想：把 user/item 映射到同一向量空间做近邻检索。

在线通常流程：

```text
实时算 user 向量 -> ANN 检索 -> 返回 topK
```

常见 ANN 方案：HNSW、IVF-PQ 等。关键是“召回率-时延”调参而不是单看检索速度。

### 2.2.3 内容语义类通路

核心思想：基于文本、标签、图像、音频等内容特征。

适用场景：

- 新物品冷启动
- 行为数据稀疏
- 强语义匹配需求

### 2.2.4 热门/趋势/规则类通路

用途：

- 新用户兜底
- 系统异常兜底
- 活动/运营规则注入

风险：热门通路权重过大时，会压制个性化和长尾探索。

## 2.3 多路融合、去重、截断

### 2.3.1 融合前必须做的标准化

因为跨路 raw score 通常不可比，需先归一化或改用 rank 融合。

常见融合：

- 线性加权融合：

$$
score(i)=\sum_r w_r\cdot norm(score_r(i))
$$

- Rank 融合（RRF）：

$$
score(i)=\sum_r\frac{w_r}{k+rank_r(i)}
$$

### 2.3.2 去重策略

去重时不要丢信息，建议保留：

- source routes
- 每路分数/名次
- multi-hit 次数
- 最终保留原因

这样后续可以做“多路命中加权”和归因分析。

### 2.3.3 场景化融合

不同场景应动态配权：

- 新用户：热门 + 内容权重高
- 老用户：行为/向量权重高
- 强时效场景：趋势/新鲜通路权重高

### 2.3.4 通路重叠与互补性分析

通路之间既可能互补，也可能重复。建议固定监控路由重叠度（Jaccard）：

$$
J(A,B)=\\frac{|A\\cap B|}{|A\\cup B|}
$$

解释方式：

- 重叠过高：通路功能重复，浪费预算
- 重叠过低：可能覆盖不同人群，但也可能质量不稳定

建议同时看“独占命中贡献”，即某一路独有候选对最终点击/转化的贡献占比。

### 2.3.5 截断策略（TopN 不是拍脑袋）

每路截断深度建议按“边际收益”确定：当追加候选对最终指标增益趋近于 0 时停止扩深。

可用离线曲线观察：`N=50/100/200/400` 时 Recall@N、后续精排增益、时延成本三者的拐点。

## 2.4 召回评估与诊断体系

### 2.4.1 离线评估

至少分三层看：

- 总体：Recall@K、HitRate@K
- 结构：新物品覆盖率、长尾覆盖率
- 归因：每路独立增益、重叠度、互补性

常见定义建议写入口径文档：

$$
HitRate@K=\frac{1}{|U|}\sum_{u\in U}\mathbf{1}(|TopK_u\cap Rel_u|>0)
$$

$$
Coverage@K=\frac{\left|\bigcup_{u\in U} TopK_u\right|}{|\mathcal{I}|}
$$

如果只看 Recall@K，容易忽视“全站内容是否被有效触达”。

### 2.4.2 在线评估

- 延迟 P95/P99
- 空召回率
- 每路命中率
- 进入排序后有效率
- 对最终业务指标的边际贡献

建议增加“分路健康看板”：

- 路由请求量占比
- 路由超时率
- 路由独占点击贡献
- 路由边际 GMV 贡献

### 2.4.3 问题排查手册

常见问题 -> 常见原因：

- 召回突然下降：索引更新失败、上游特征空值、路由开关变更
- 结果同质化：热门路过强、融合权重失衡、去重策略粗糙
- 新内容进不来：增量索引延迟、内容路权重太低

## 2.5 召回工程化要点

- 索引生命周期：构建、增量、压缩、热切换、回滚
- 缓存策略：热门候选、用户短期向量缓存
- 并行调度：慢路超时熔断，不拖累主链路
- 观测能力：按 route 维度监控而不是只看总量

---

# 三、排序层（Ranking）

排序层是“候选上的精细效用估计器”，它决定了页面前几位内容的质量上限。

## 3.1 排序任务定义与目标拆解

给定候选集 $\mathcal{S}(u,c)$，学习评分函数：

$$
s_{uic}=f(u,i,c)
$$

输出排序：

$$
\pi(u,c)=\operatorname{sort}_{i\in\mathcal{S}(u,c)}(s_{uic})
$$

排序层目标通常拆为三类：

- 相关性目标：用户是否会点击/消费
- 价值目标：是否转化、收益多大
- 稳定目标：跨场景分布稳定、可校准

## 3.2 特征与样本框架

### 3.2.1 特征分层

推荐排序常用“分层特征”思维：

- 用户层：画像、长期兴趣、短期会话
- 物品层：质量、时效、内容信号
- 上下文层：时间、设备、场景、入口
- 交互层：用户与物品匹配关系

### 3.2.2 训练样本构造

关键是定义好“样本空间”与“负样本语义”：

- 曝光未点负样本：最贴近真实决策边界
- 随机负样本：扩大覆盖
- 难负样本：增强排序前列区分能力

样本还要做：

- 时间切分（防穿越）
- 分桶均衡（防头部类压制）
- 去重去噪（刷量与异常流量）

### 3.2.3 延迟反馈处理

转化/留存存在延迟，常见做法：

- 截断窗口训练（牺牲部分样本）
- 标签回填机制（延后训练）
- 多任务拆分（短期目标 + 长期代理目标）

延迟标签常见处理口径：

- 训练时使用“成熟标签窗口”（例如曝光后 T+7 才纳入 CVR）
- 线上监控使用“实时代理目标”（例如深度点击/加购）
- 周期性用成熟标签回溯修正模型偏差

### 3.2.4 多目标训练目标（框架）

多任务常写成加权和：

$$
\mathcal{L}=\lambda_{ctr}\mathcal{L}_{ctr}+\lambda_{cvr}\mathcal{L}_{cvr}+\lambda_{wt}\mathcal{L}_{wt}
$$

权重可固定，也可按任务不确定性或梯度大小动态调整。上线时必须同步评估“主目标提升是否伤害护栏目标”。

## 3.3 排序范式与损失函数（框架级）

这里讲范式，不展开到模型实现细节。

### 3.3.1 Pointwise

单样本监督，最常见是 BCE：

$$
\mathcal{L}_{BCE}=-y\log\hat y-(1-y)\log(1-\hat y)
$$

优点：简单稳定；缺点：对列表前列质量不够敏感。

### 3.3.2 Pairwise

学习正负样本相对顺序，典型形式：

$$
\mathcal{L}_{pair}= -\log\sigma(s^+ - s^-)
$$

优点：直接优化顺序；缺点：样本构造和采样策略更敏感。

### 3.3.3 Listwise

直接面向列表指标优化（如 NDCG 近似），更贴近线上体验，但训练复杂度更高。

排序范式选择建议：

- 数据规模小、先求稳：Pointwise
- 更关注前列顺序：Pairwise
- 强调列表质量与页面体验：Listwise

实践中常见组合：先 Pointwise 训练稳态模型，再用 Pairwise/Listwise 做增量精调。

## 3.4 排序评估、校准与一致性

### 3.4.1 离线评估

常用组合：

- AUC（区分能力）
- LogLoss/Brier（概率质量）
- NDCG@K/MRR（前列排序质量）

### 3.4.2 校准评估

如果排序分数要用于收益决策、预算控制，必须看校准。

常见检查：

- Reliability Diagram
- ECE
- 分群校准（新老用户/类目/入口）

### 3.4.3 离线在线一致性

避免“离线涨、线上掉”的关键做法：

- 用线上真实曝光分布构造评估集
- 加入反事实评估（IPS/SNIPS/DR）
- 强化分群评估，而不是只看全量均值

建议固定分群：

- 新用户/老用户
- 高活跃/低活跃
- 头部类目/长尾类目
- 首页/详情页等入口

每次迭代至少给出“全量 + 关键分群”的一致性报告。

## 3.5 排序线上服务与风控

- 推理服务要限时与批量化
- 特征缺失要有默认值与告警
- 模型输出异常（全 0/全 1）要自动熔断
- 灰度阶段监控分数分布漂移
- 对高风险场景保留可解释字段

常见线上风控阈值建议：

- 超时阈值：超过阈值直接降级到轻量排序
- 空特征阈值：关键特征缺失率超过阈值触发告警
- 分数漂移阈值：按 KS/PSI 对比历史分布
- 负反馈阈值：评论/投诉/跳出率异常立即回滚

---

# 四、重排层（Re-ranking）

重排层不负责“重新预测用户喜好”，而是用列表约束和策略把排序结果变成“可展示、可运营、可持续”的页面结果。

## 4.1 重排目标与策略边界

重排层通常解决以下问题：

- 列表同质化
- 头部挤压与长尾生存
- 新内容冷启动曝光
- 商业位与内容位协同
- 合规与安全约束

边界原则：

- 重排不能把排序主信号完全打乱
- 重排必须可解释，便于运营与排障
- 重排要可配置化，支持快速策略变更

## 4.2 列表级优化形式

可以把重排看作多目标约束优化：

$$
\max_{\pi}\left(\sum_{k=1}^{K}\alpha_k\,rel(i_k)+\lambda_D D(\pi)+\lambda_N N(\pi)+\lambda_B B(\pi)\right)
$$

其中：

- $rel(i_k)$：排序层相关性收益
- $D(\pi)$：多样性收益
- $N(\pi)$：新颖性/探索收益
- $B(\pi)$：商业收益或约束收益

约束可分两类：

- 硬约束：必须满足（如违规内容不能出现）
- 软约束：尽量满足（如类目占比不超过阈值）

软约束常转为惩罚项：

$$
score' = score - \beta\cdot penalty(\pi)
$$

### 4.2.1 常见可操作约束

- 类目上限：某类最多 $m$ 个
- 作者间隔：同作者最小间隔位数
- 头部抑制：头部内容单页上限
- 新品保底：新内容最少曝光配额

## 4.3 重排策略框架（不展开模型）

### 4.3.1 相关性-多样性平衡

代表性框架（MMR 思路）：

$$
\operatorname*{argmax}_{i\in C\setminus S}[\lambda\,Rel(i)-(1-\lambda)\max_{j\in S}Sim(i,j)]
$$

含义：每一步选“既相关、又不太重复”的 item。

### 4.3.2 配额与公平框架

- 按类目/创作者/商家做配额
- 分群看曝光占比是否健康
- 防止“强者愈强”导致生态退化

供给侧集中度可用 Gini 粗衡量：

$$
Gini = \frac{\sum_{i=1}^{n}\sum_{j=1}^{n}|x_i-x_j|}{2n\sum_{i=1}^{n}x_i}
$$

其中 $x_i$ 可取作者或商家的曝光量。Gini 越高，说明曝光越集中。

### 4.3.3 商业策略框架

典型组合分：

$$
score_{rerank}=a\cdot rel+b\cdot value-c\cdot risk
$$

关键不是公式本身，而是“参数随场景动态化”。

## 4.4 重排评估与上线控制

### 4.4.1 离线评估

- 列表质量：NDCG@K
- 多样性：ILD、类目覆盖率
- 新颖性：新内容占比、长尾占比
- 生态指标：作者/商家曝光分布

ILD 一个常见定义：

$$
ILD=\frac{2}{K(K-1)}\sum_{a<b}(1-sim(i_a,i_b))
$$

### 4.4.2 在线评估

- 用户体验：时长、负反馈率、返回率
- 业务收益：CTR/CVR/GMV/广告收益
- 生态健康：供给侧曝光 Gini、头尾比

### 4.4.3 上线节奏与回滚

- 从小流量灰度开始
- 先开软约束，再逐步加硬约束
- 预设回滚触发条件（护栏触发即回滚）

建议上线分三段：

1. 影子评估：仅记录不生效，观察约束命中与分数漂移。
2. 小流量灰度：验证主指标与护栏是否同时稳定。
3. 全量放开：保留一键回滚和参数热修复能力。

## 4.5 重排工程化要点

- 时延预算紧：重排逻辑必须轻量
- 配置中心化：策略参数可热更新
- 解释日志：记录每次提权/降权原因
- 异常兜底：重排失败时直接回退排序列表

重排解释日志建议字段：

```text
item_id, old_rank, new_rank, trigger_rule, penalty_type, penalty_value, request_id
```

这样线上异常时能快速回答“为什么这条内容被压下去/提上来”。

---
# 五、主流模型介绍

这一章只讲“模型本身”，不再重复前四章的流程框架。写法遵循一个统一模板：
- 模型解决什么问题
- 关键结构与核心公式
- 常用训练目标（loss）
- 工程落地怎么做
- 优缺点与适用场景

## 5.1 召回模型（Candidate Generation）

召回的目标是：在海量候选中先“快而准”地找回一个相对小的候选集（例如几百到几千个），供后续排序使用。召回阶段的分数可以不等于最终点击概率，但要有较高覆盖和较低漏召。

### 5.1.1 协同过滤（UserCF / ItemCF / Swing）

这是工业里最常见的“首层兜底召回”，即使上了深度模型也很少完全替代。

1. `UserCF`：找相似用户，再把相似用户喜欢的物品推荐给当前用户。  
用户相似度常见定义：
$$
\operatorname{sim}(u,v)=\frac{|I_u\cap I_v|}{\sqrt{|I_u||I_v|}}
$$
其中 $I_u$ 表示用户 $u$ 交互过的物品集合。  
对候选物品 $i$ 的打分常见形式：
$$
\operatorname{score}(u,i)=\sum_{v\in N_k(u)}\operatorname{sim}(u,v)\cdot r_{v,i}
$$
其中 $r_{v,i}$ 可取点击/收藏/购买的加权行为值。

2. `ItemCF`：找相似物品，比 UserCF 更稳，更适合大规模在线服务。  
物品相似度常见定义：
$$
\operatorname{sim}(i,j)=\frac{|U_i\cap U_j|}{\sqrt{|U_i||U_j|}}
$$
其中 $U_i$ 是与物品 $i$ 有过交互的用户集合。  
对用户 $u$ 的候选物品 $i$：
$$
\operatorname{score}(u,i)=\sum_{j\in I_u}\operatorname{sim}(i,j)\cdot w_{u,j}
$$
$w_{u,j}$ 可体现行为强度、时间衰减等。

3. `Swing`：在 ItemCF 基础上抑制“高活跃用户导致的伪共现噪声”。  
常见思想是：若两个用户都很“泛兴趣”，其共同行为贡献应更小。一个常见形式：
$$
\operatorname{swing}(i,j)=\sum_{u,v\in U_i\cap U_j,\,u<v}\frac{1}{\alpha+|I_u\cap I_v|}
$$
$\alpha$ 是平滑项。交集越大（两人越像“什么都点”），分母越大，贡献越小。

工程经验：
- CF 需要做时间衰减，否则旧兴趣污染严重。
- 要做去热门惩罚，否则“爆款挤占”长尾召回。
- CF 往往是冷启动和故障场景的关键兜底层。

### 5.1.2 双塔召回（DSSM / YouTube DNN）

双塔是现代大规模召回主力：把用户和物品编码到同一向量空间，做向量近邻检索。

1. 结构：
- 用户塔输出用户向量 $\mathbf{u}=f_\theta(x_u)$
- 物品塔输出物品向量 $\mathbf{v}_i=g_\phi(x_i)$
- 匹配分数常用点积或余弦：
$$
s(u,i)=\mathbf{u}^\top\mathbf{v}_i
$$

2. 训练目标（常用 sampled softmax / in-batch softmax）：
$$
\mathcal{L}=-\log\frac{\exp(s(u,i^+)/\tau)}{\sum_{i\in\{i^+\}\cup\mathcal{N}}\exp(s(u,i)/\tau)}
$$
其中 $i^+$ 是正样本，$\mathcal{N}$ 为负样本集合，$\tau$ 为温度系数。

3. 线上检索：
- 离线/准实时构建 ANN 索引（如 HNSW、IVF-PQ）。
- 在线只算用户向量并做 TopK 近邻查询，满足毫秒级时延。

关键工程点：
- 负采样质量决定上限：随机负样本太容易，建议混入 hard negative。
- 向量与索引更新频率要匹配业务变化（内容场景通常更高频）。
- 双塔得分未必可直接作为 CTR 概率，通常仍需后续精排校准。

### 5.1.3 图召回（LightGCN 为代表）

图召回把“用户-物品交互”建成二分图，利用多跳传播建模高阶协同。

LightGCN 的核心传播：
$$
\mathbf{e}_u^{(k+1)}=\sum_{i\in\mathcal{N}(u)}\frac{1}{\sqrt{|\mathcal{N}(u)||\mathcal{N}(i)|}}\mathbf{e}_i^{(k)}
$$
$$
\mathbf{e}_i^{(k+1)}=\sum_{u\in\mathcal{N}(i)}\frac{1}{\sqrt{|\mathcal{N}(i)||\mathcal{N}(u)|}}\mathbf{e}_u^{(k)}
$$
最后把各层 embedding 加权求和得到最终表示。训练常配合 BPR：
$$
\mathcal{L}_{BPR}=-\sum\log\sigma\big(s(u,i^+)-s(u,i^-)\big)
$$

优点：
- 对稀疏场景通常强于纯 ID 双塔。
- 高阶协同关系表达能力强。

注意点：
- 图更新成本高，在线增量更新复杂。
- 热点节点容易过平滑，需控制层数和正则。

### 5.1.4 序列召回（GRU4Rec / SASRec / BERT4Rec）

当用户兴趣随时间快速变化时，序列召回比静态兴趣更有效。

1. `SASRec`（自注意力序列建模）：
- 输入为行为序列 embedding + 位置编码。
- 自注意力：
$$
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$
- 用最后位置表征预测下一物品。

2. `BERT4Rec`（双向掩码建模）：
- 随机 mask 序列中的物品 token。
- 用双向 Transformer 预测被 mask 的物品，目标是 masked item 的交叉熵。

3. 工程落地注意：
- 需做序列截断（如最近 50/100/200 条）平衡效果与成本。
- 行为类型要区分（点/赞/购）并加权。
- 时间间隔特征（time gap）对短内容场景非常关键。

### 5.1.5 多兴趣召回（MIND / ComiRec）

单一用户向量在“兴趣分裂”用户上会失真，多兴趣模型为同一用户学习多个兴趣向量。

典型做法：
- 从用户行为序列通过动态路由/自注意力得到 $K$ 个兴趣向量 $\{\mathbf{z}_u^{(1)},...,\mathbf{z}_u^{(K)}\}$。
- 对物品 $i$ 的匹配分数常取：
$$
\operatorname{score}(u,i)=\max_{k\in\{1,...,K\}}\big(\mathbf{z}_u^{(k)}\cdot\mathbf{v}_i\big)
$$

优点：
- 在“一个用户多个主题偏好”的信息流场景通常显著提升召回覆盖。

代价：
- 在线侧需要维护多向量检索或多路合并，系统复杂度更高。

### 5.1.6 生成式召回（DSI / Generative Retrieval / HSTU 等）

这是近年前沿方向：把“检索”转成“生成 item 标识序列”的问题。

核心思路：
- 将 item ID 离散化为 token 序列（或语义 token）。
- 给定用户上下文 $x_u$，直接生成目标 item token 序列 $y=(y_1,...,y_T)$。
- 训练目标：
$$
\mathcal{L}=-\sum_{t=1}^{T}\log p(y_t\mid y_{<t},x_u)
$$

优势：
- 可统一“理解 + 检索”过程，具备更强语义泛化潜力。
- 在超大物品库可结合分层 token 降低检索复杂度。

挑战：
- ID 生成错误会直接导致不可用候选。
- 训练与部署成本高，线上延迟控制难。
- 工程上通常先作为增量通道并行于双塔，而非一次性替换。

---

## 5.2 排序模型（Ranking）

排序目标是：对召回候选估计“业务目标概率或价值”，并给出稳定可控的最终顺序。

### 5.2.1 线性与因子分解基线（LR / FM / FFM）

1. `LR`：
$$
\hat{y}=\sigma(\mathbf{w}^\top\mathbf{x}+b)
$$
解释强、鲁棒、上线简单，常作为回归基线与兜底模型。

2. `FM`（二阶特征交叉自动学习）：
$$
\hat{y}=w_0+\sum_i w_i x_i+\sum_{i<j}\langle\mathbf{v}_i,\mathbf{v}_j\rangle x_i x_j
$$
适合高维稀疏 one-hot 特征。

3. `FFM`：不同 field 使用不同隐向量，表达更强但参数更大、训练更慢。

工程建议：
- 不要小看 LR/FM，它们是排查线上问题和做增量评估的关键锚点。
- 深度模型退化时，基线模型可快速托底。

### 5.2.2 深度特征交叉模型（Wide&Deep / DeepFM / DCN-v2 / DLRM）

1. `Wide&Deep`：
- Wide 分支记忆共现规则，Deep 分支学习泛化。
- 最终 logit 常为两支输出加和后过 sigmoid。

2. `DeepFM`：
- 用 FM 分支替代人工 Wide 交叉，和 Deep 分支共享底层 embedding。
- 兼顾“低阶可解释交叉 + 高阶非线性交叉”。

3. `DCN / DCN-v2`：显式高阶交叉。  
Cross 层经典形式：
$$
\mathbf{x}_{l+1}=\mathbf{x}_0(\mathbf{w}_l^\top\mathbf{x}_l)+\mathbf{b}_l+\mathbf{x}_l
$$
DCN-v2 在参数化和结构上更实用（如低秩/混合专家），用于大规模 LTR 更稳。

4. `DLRM`（工业大规模经典）：
- 稀疏离散特征经 embedding 表，稠密特征经 bottom MLP。
- 特征交互常用点积交互矩阵，再接 top MLP 输出概率。

适用场景：
- 特征工程成熟、样本规模大、业务追求稳定提升时，以上模型长期有效。

### 5.2.3 行为序列排序（DIN / DIEN / BST 类）

1. `DIN`（目标感知兴趣激活）：
- 给定目标物品 embedding $\mathbf{e}_{target}$，对历史行为 $\mathbf{e}_t$ 做注意力：
$$
\alpha_t=\operatorname{softmax}(a(\mathbf{e}_t,\mathbf{e}_{target}))
$$
$$
\mathbf{u}_{interest}=\sum_t \alpha_t\mathbf{e}_t
$$
- 与用户侧其他特征拼接后进入 MLP 输出 CTR。

2. `DIEN`（兴趣演化）：
- 先用 GRU 编码兴趣序列，再用注意力门控更新，突出与目标相关的兴趣迁移。
- 常见更新思想：用注意力值调制 GRU 更新门，抑制无关行为对状态的污染。

3. `BST/Transformer 排序`：
- 用 Transformer 建模长序列依赖，在行为序列较长时优于 RNN。

关键点：
- 序列特征若处理不好，排序模型容易“记住热门而不是记住用户”。
- 序列窗口、行为类型权重、时间衰减是效果上限的关键三件事。

### 5.2.4 多任务排序（ESMM / MMoE / PLE）

推荐系统通常不只优化 CTR，还要兼顾 CVR、时长、互动深度等目标。

1. `ESMM`（解决 CVR 样本选择偏差）：
- 建模：
$$
P(\text{CTCVR}|x)=P(\text{CTR}|x)\cdot P(\text{CVR}|\text{click},x)
$$
- 训练时直接学习 CTR 与 CTCVR 两个可观测目标，避免只在点击样本上学 CVR 导致偏差。

2. `MMoE`：
- 多个共享 expert，任务特定 gate 学习组合权重：
$$
\mathbf{h}^{(t)}=\sum_{k=1}^{K}g_k^{(t)}(x)\cdot \operatorname{Expert}_k(x)
$$
- 适合任务相关但不完全一致的场景。

3. `PLE`：
- 在共享专家与任务专家间分层抽取，减少任务冲突与“跷跷板效应”。

实战要点：
- 多任务必须先定义清楚“主目标”和“护栏目标”。
- 线上看板至少同时看：CTR、CVR、GMV/时长、投诉率/不感兴趣率。

### 5.2.5 排序目标函数与校准

1. `Pointwise`：BCE/MSE，训练稳定，实现简单。
2. `Pairwise`：BPR、hinge，直接学习“正样本比分负样本高”。
3. `Listwise`：ListMLE/LambdaRank，直接优化列表质量（NDCG 等）。

示例（BCE）：
$$
\mathcal{L}_{BCE}=-\frac{1}{N}\sum_{n=1}^{N}\big[y_n\log \hat{y}_n+(1-y_n)\log(1-\hat{y}_n)\big]
$$

上线校准常用：
- `Platt scaling`：对 logit 再做一层 sigmoid 标定。
- `Isotonic regression`：非参数单调校准，适合分布复杂场景。

---

## 5.3 重排模型（Re-ranking）

重排在“已拿到排序分”后做列表级优化，核心是平衡相关性与业务约束。

### 5.3.1 多样性重排（MMR / xQuAD）

1. `MMR`：
$$
\operatorname*{argmax}_{i\in C\setminus S}\left[\lambda\cdot Rel(i)-(1-\lambda)\cdot\max_{j\in S}Sim(i,j)\right]
$$
- $Rel(i)$：排序模型相关性分。
- $Sim(i,j)$：内容相似度。
- $\lambda$ 越大越偏相关性，越小越偏多样性。

2. `xQuAD`：按子主题覆盖增量选 item，可显式控制“主题曝光覆盖率”。

落地建议：
- MMR 参数应分场景（首页、关注流、视频详情页）单独调。
- 多样性不是越强越好，过强会伤点击。

### 5.3.2 约束优化重排（多目标配额与公平）

把重排写成约束优化问题：
$$
\max_{\pi}\;\mathbb{E}_{i\sim\pi}[R(i)]\quad
\text{s.t.}\;\mathbb{E}_{i\sim\pi}[C_m(i)]\ge b_m,\;m=1,...,M
$$

常用拉格朗日形式：
$$
\mathcal{L}(\pi,\lambda)=\mathbb{E}[R(i)]+\sum_m\lambda_m\big(\mathbb{E}[C_m(i)]-b_m\big)
$$
在线通过调节 $\lambda_m$ 控制约束满足率（如商业位、作者公平、新内容占比）。

### 5.3.3 学习式重排（Listwise Neural Re-ranker）

输入不再是单条样本，而是“候选列表 + 列表统计特征”：
- item 特征：预测 CTR/CVR、类目、时效、质量分
- list 特征：当前类目覆盖、作者覆盖、已放入内容分布

输出方式：
- 逐步选取（greedy policy）
- 或一次性输出全列表得分再排序

常见训练目标：
- 直接用 NDCG 近似优化
- 或用强化学习目标优化长期奖励

### 5.3.4 探索型重排（Bandit / RL）

1. `UCB`：
$$
\operatorname{UCB}_a=\hat{\mu}_a+c\sqrt{\frac{\ln t}{n_a}}
$$
- $\hat{\mu}_a$ 是当前收益估计，$n_a$ 是被选次数。
- 第二项是探索奖励，冷门内容会被适度探索。

2. `Thompson Sampling`：
- 为每个臂维护后验分布（如 Beta 分布），采样后选最大值。
- 工程上在稀疏反馈场景往往比固定 epsilon-greedy 稳定。

3. `RL 重排`（如 SlateQ 思路）：
- 目标从单次点击扩展到长期价值（留存、复访、生命周期价值）。
- 关键难点是奖励延迟与离线评估偏差。

### 5.3.5 重排离线评估与反事实校正

重排策略不能只看在线 A/B，也要做离线风险评估。

1. `IPS`（Inverse Propensity Scoring）：
$$
\hat{V}_{IPS}=\frac{1}{N}\sum_{n=1}^{N}\frac{\pi(a_n|x_n)}{\mu(a_n|x_n)}r_n
$$
- $\mu$ 是日志策略，$\pi$ 是新策略。
- 当重要性权重过大时方差会爆炸，需要裁剪或 SNIPS。

2. `Doubly Robust`：结合直接模型与 IPS，通常更稳健。

---

## 5.4 代表性与前沿模型（建议重点掌握）

下面按“工业代表性 + 前沿方向”给一份学习优先级。

### 5.4.1 工业代表性模型（高优先级）

1. `YouTube DNN 双塔召回（2016）`  
意义：奠定了“大规模向量召回 + ANN”工业范式。

2. `Wide&Deep / DeepFM（2016-2017）`  
意义：建立了“显式低阶交叉 + 深度高阶泛化”的主流排序框架。

3. `DIN / DIEN（2018-2019）`  
意义：把“用户历史行为”从静态聚合升级为目标感知、随时间演化的兴趣建模。

4. `BERT4Rec / SASRec（2018-2019）`  
意义：序列推荐进入 Transformer 时代，长依赖建模能力明显增强。

5. `LightGCN（2020）`  
意义：图协同过滤的高效化范式，结构简洁、效果稳定。

6. `DCN-v2 / DLRM（2019-2020）`  
意义：在大规模工业排序中实现“可部署、可扩展、可复用”的高性能交叉建模。

### 5.4.2 前沿方向模型（2024+）

1. `生成式检索推荐（DSI / Generative Retrieval）`  
关键词：把检索改写为序列生成；适合超大规模候选库与语义泛化。  
难点：ID 生成稳定性、延迟与可控性。

2. `HSTU（ICML 2024）`  
关键词：超大规模序列建模与生成式推荐范式；强调大数据、大模型下的序列行为理解。  
启示：工业推荐模型正在从“单任务单塔”走向“统一序列建模 + 多任务”。

3. `LLM 化推荐（P5 等）`  
关键词：把推荐任务表述为语言任务（prompt + generation）。  
现状：在可解释交互、多任务统一上潜力大，但成本和稳定性仍是规模化门槛。

### 5.4.3 学习顺序建议

1. 先吃透经典：`ItemCF + 双塔 + DeepFM/DCN + DIN + MMoE/ESMM + MMR`。
2. 再上序列与图：`SASRec/BERT4Rec + LightGCN`。
3. 最后看前沿：`Generative Retrieval + HSTU + LLM4Rec`，重点理解“为什么难上线”。

---

## 5.5 论文与资料索引（按模型学习）

以下是本章建议优先阅读的一手资料（论文/官方技术文档）：

- Deep Neural Networks for YouTube Recommendations（RecSys 2016）
- Wide & Deep Learning for Recommender Systems（2016）
- DeepFM: A Factorization-Machine based Neural Network for CTR Prediction（2017）
- Deep Interest Network for Click-Through Rate Prediction（KDD 2018）
- Deep Interest Evolution Network for CTR Prediction（AAAI 2019）
- Self-Attentive Sequential Recommendation（SASRec, 2018）
- BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations（2019）
- Deep Learning Recommendation Model for Personalization and Recommendation Systems（DLRM, 2019）
- Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems（DCN-v2, 2020）
- LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation（SIGIR 2020）
- Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate（KDD 2018）
- Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts（KDD 2018）
- Recommender Systems with Generative Retrieval（2023）
- Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations（ICML 2024）
- Recommendation as Language Processing (P5)（2022）

提示：
- 读论文时不要只看模型图，重点看训练目标、采样策略、线上约束和消融实验。
- 工业落地时，“数据和系统”通常比“模型名字”更决定最终收益。
