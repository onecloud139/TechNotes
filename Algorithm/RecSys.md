# 一、推荐系统基本知识

## 术语与缩写

- `CTR`（Click-Through Rate）：点击率，通常是 `点击次数/曝光次数`。
- `CVR`（Conversion Rate）：转化率，通常是 `转化次数/点击次数` 或按业务口径定义。
- `CTCVR`：点击后转化概率的整体口径，常用于 ESMM。
- `WT`（Watch Time）：观看时长/停留时长指标。
- `REV`（Revenue）：收益指标（如广告收益、成交收益）。
- `AUC`：ROC 曲线下面积，衡量正负样本区分能力。
- `GAUC`：按用户或场景分组后加权的 AUC，更贴近推荐场景。
- `LogLoss`：对概率预测质量敏感的交叉熵损失。
- `MRR`：首个相关结果倒数排名的平均值。
- `DCG/NDCG`：考虑位置折损的列表质量指标，NDCG 是归一化后的 DCG。
- `PSI`（Population Stability Index）：衡量训练分布与线上分布漂移程度。
- `KL`（Kullback-Leibler Divergence）：衡量两个分布差异的非对称距离。
- `Embedding`：把离散特征（用户ID、物品ID、类目ID）映射成稠密向量。
- `ANN`（Approximate Nearest Neighbor）：近似最近邻向量检索。
- `HNSW`：图结构 ANN 索引，检索速度快、召回高。
- `IVF-PQ`：先聚类分桶（IVF）再乘积量化（PQ）压缩检索。
- `Hard Negative`：模型容易混淆为正样本的困难负样本。
- `Pointwise/Pairwise/Listwise`：分别按单样本、样本对、列表定义训练目标。
- `Calibration`（校准）：让预测分数与真实概率在数值上更一致。
- `Propensity`：日志策略下样本被曝光的概率，反事实评估要用。
- `IPS/SNIPS/DR`：反事实评估方法。IPS 无偏高方差，SNIPS 是归一化 IPS，DR 结合直接法与 IPS。
- `DM`（Direct Method）：先训练回报模型再估计策略价值，方差低但易有偏。
- `UCB/Thompson Sampling`：Bandit 探索利用算法，平衡短期收益与探索。
- `MoE/MMoE/PLE`：多专家多任务框架，用门控分配共享与任务专属能力。
- `AUGRU`：用注意力值调制 GRU 更新门的序列建模单元（DIEN 常见）。
- `trie`（前缀树）：约束生成式解码合法路径，避免生成非法 ID。


## 1.1 推荐系统的本质：受约束的在线决策

推荐系统不是离线分类器，而是在线决策系统。每次请求都要在有限时延内输出一个列表，并且同时满足用户目标、业务目标和系统约束。

  - 1.1.1 问题形式化

    给定用户 $u$、上下文 $c$、候选集 $\mathcal{I}$，输出列表 $\pi=(i_1,\dots,i_K)$：

    $$
    \pi^* = \arg\max_{\pi \in \Pi(\mathcal{I})}\;\mathbb{E}[R(\pi\mid u,c)]
    $$

    其中 $R$ 是列表收益函数，通常是多目标组合：

    $$
    R = w_{ctr}\cdot CTR + w_{cvr}\cdot CVR + w_{wt}\cdot WT + w_{rev}\cdot REV
    $$

    如果你只优化单一指标（比如 CTR），经常会牺牲长期留存或内容生态健康。

  - 1.1.2 为什么是“列表优化”不是“单项预测”

    即使单个 item 的点击概率估计很准，也不代表整页体验好。因为列表还涉及：

    - 同质化（10 个类似内容）
    - 曝光公平（头部垄断）
    - 新颖性（探索不足）
    - 业务约束（广告位、库存、合规）

    所以推荐的最终优化对象是“页面级结果”，不是单 item 分数。

  - 1.1.3 三类核心约束

    - 时延约束：通常毫秒级预算，不能超时。
    - 稳定性约束：流量峰值不崩、结果分布不剧烈抖动。
    - 合规约束：内容安全、隐私、行业监管要求。

    三类约束不是并列口号，而是可量化的工程目标。常见拆解方式：

    - 时延约束要落到分层预算：

    $$
    T_{total}=T_{gateway}+T_{recall}+T_{ranking}+T_{rerank}+T_{serialize}
    $$

      其中真正可优化的是召回/排序/重排三段时延，且线上通常盯 `P95/P99` 而不是均值。
    - 稳定性约束要有抖动口径：例如主指标按小时环比不超过阈值、关键路由空结果率不超过阈值。
    - 合规约束要前置而非事后：违规过滤应在召回后或排序前就介入，避免末端兜底成本过高。

## 1.2 分阶段架构与模块边界

推荐系统通用链路：

```text
召回（Recall） -> 排序（Ranking） -> 重排（Re-ranking）
```

  - 1.2.1 每层职责（必须边界清晰）

    - 召回：从千万/亿级物品中快速取几百~几千候选，目标是“高覆盖 + 低时延”。
    - 排序：在候选集合上做更准确的效用估计，目标是“相关性和收益”。
    - 重排：做列表级约束优化，目标是“体验、生态、商业平衡”。

    每层都要定义“可交付物”：

    - 召回层交付“候选池质量”：覆盖率、空召回率、路由互补性。
    - 排序层交付“单条估计质量”：概率质量、前列精度、可校准性。
    - 重排层交付“页面级质量”：多样性、供给公平、策略命中率。

  - 1.2.2 输入输出契约（工程里必须定义）

    每层都要有稳定契约：

    - 输入字段版本（user/context/feature schema）
    - 输出字段定义（score、source、explain、trace_id）
    - 异常行为（超时、空结果、降级）

    如果契约不清楚，线上问题会很难排查。建议把契约写成可校验 schema（例如 proto/avro/json schema），并在发布时做兼容性检查：

    - 向后兼容：新增字段不能破坏老服务解析。
    - 向前兼容：可选字段缺失时要有默认语义。
    - 版本治理：`schema_version` 必须随重大语义变更递增。

  - 1.2.3 为什么不能“一层做完”

    - 全库精排太贵：复杂模型无法全量逐项打分。
    - 召回和排序的优化目标不同：召回更偏覆盖，排序更偏精度。
    - 重排处理的是列表结构问题，排序分数本身无法完全表达。

    从复杂度看也不现实：若全库物品规模是 $|\mathcal{I}|$，复杂精排代价近似 $O(|\mathcal{I}|)$；分层后代价接近：

    $$
    O(|\mathcal{I}|) \xrightarrow{\text{召回截断}} O(K_{recall}) \xrightarrow{\text{排序截断}} O(K_{rank})
    $$

    其中 $K_{recall}\ll|\mathcal{I}|$，$K_{rank}\ll K_{recall}$。分层本质是把不可计算问题变成可计算问题。

## 1.3 数据体系、特征体系与样本构造

这部分是全链路共享地基，做错了后面都不稳。

  - 1.3.1 事件日志最小单元

    推荐日志建议至少包含：

    ```text
    (request_id, user_id, item_id, action, timestamp, position, scene, device, source)
    ```

    常见动作链：

    $$
    \text{曝光} \rightarrow \text{点击} \rightarrow \text{深度行为} \rightarrow \text{转化}
    $$

  - 1.3.2 时间口径

    至少区分三类时间：

    - event_time：行为真实发生时间
    - process_time：日志入仓时间
    - feature_time：特征快照时间

    训练时必须满足：

    $$
    feature\_time \le label\_time
    $$

    否则就是信息泄漏（未来信息穿越）。

  - 1.3.3 标签构造

    标签不是“直接拿动作”，而是要定义窗口和归因规则。

    - 观测窗口：例如用最近 14 天训练。
    - 标签窗口：例如曝光后 24 小时点击记正样本。
    - 延迟标签：转化和留存需要回填。

    建议同时固定三套规则并写进口径文档：

    - 归因规则：last-touch、first-touch 或位置加权归因，不能混用。
    - 去重规则：同一用户-物品在窗口内是“首次行为”还是“多次行为累积”。
    - 负样本规则：曝光未点是否全纳入，是否对超热门位做降采样。

    否则不同实验之间样本语义不一致，离线结果无法横向比较。

  - 1.3.4 负样本来源

    常见负样本：

    - 曝光未点（最常用）
    - 随机负采样（缓解类不平衡）
    - 难负样本（提升区分能力）

    实际训练常组合使用，并记录来源字段，便于调参与解释。

    常见混合采样配比可写成：

    $$
    \mathcal{N}=\mathcal{N}_{expo}\cup \mathcal{N}_{rand}\cup \mathcal{N}_{hard},\quad
    r_{expo}:r_{rand}:r_{hard}=a:b:c
    $$

    其中 $a,b,c$ 需要按场景调节。短视频通常 hard negative 占比更高，电商可更重曝光未点样本。

  - 1.3.5 特征一致性

    全链路要保证“离线训练特征”和“线上推理特征”一致。

    常见问题：

    - 离线用全量统计，线上只有实时窗口
    - 字段填充规则不同
    - 类别编码字典版本不一致

    这类问题会导致“离线很好，线上掉线”。

  - 1.3.6 样本加权与再平衡

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

    其中 $\Delta t_i$ 是样本距当前时间的间隔。

  - 1.3.7 数据质量清单

    每次训练前建议固定检查：

    - 主键完整性：`request_id/user_id/item_id` 缺失率
    - 事件时序：`event_time` 乱序比例
    - 标签异常：CTR/CVR 是否突变
    - 特征异常：空值率、极值率、枚举爆炸
    - 分布漂移：训练窗口 vs 线上实时窗口 PSI/KL

    建议给每项设置触发阈值和处理动作，例如：

    - 主键缺失率 > 0.5%：停止训练并告警。
    - 关键特征空值率日环比翻倍：进入灰度保护。
    - PSI > 0.25：强制做分群评估后再发布。

## 1.4 分数语义、归一化与校准

“有分数”不等于“分数可直接比较”。

  - 1.4.1 三类分数语义

    - 检索分（召回）：相似度/距离，跨路通常不可比。
    - 预测分（排序）：概率或效用估计。
    - 策略分（重排）：相关性 + 约束项组合。

    三类分数常见关系是：

    $$
    score_{final}=g\big(score_{rank},\,constraints,\,diversity,\,business\big)
    $$

    因此“排序分高”不一定“最终名次高”。这不是模型错，而是策略层显式干预的结果。线上排障时必须同时看三层分数与变换日志。

  - 1.4.2 跨路分数归一化

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

  - 1.4.3 概率校准

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

  - 1.5.1 离线指标（定义要统一）

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

  - 1.5.2 在线实验（A/B）

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

    $$
    \hat p=\frac{click_A+click_B}{n_A+n_B}
    $$

    再根据 $z$ 值计算 p-value 判断显著性。

    效果提升建议同时报告相对提升：

    $$
    Lift=\frac{metric_B-metric_A}{metric_A}
    $$

    并给出置信区间，而不是只报单点值。

  - 1.5.3 反事实评估（Off-policy）

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

  - 1.6.1 主要偏差

    - 曝光偏差：仅曝光样本可观测。
    - 位置偏差：前排更容易被看到。
    - 选择偏差：购买往往以点击为前提。
    - 流行度偏差：热门获得更多曝光，进一步变热。

    位置偏差常见拆分：

    $$
    P(click)=P(exam)\cdot P(relevance)
    $$

  - 1.6.2 探索机制

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
    \text{random action}, & \text{with prob } \epsilon\
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

  - 1.6.3 反馈闭环管理

    模型上线后会改变曝光分布，进而改变下一轮训练数据。要做：

    - 数据回放和版本对齐
    - 分群监控（新老用户、头尾内容）
    - 周期性校验“是否越来越单一”

    建议加两个固定监控量：

    - 曝光熵（越低越同质）：

    $$
    H=-\sum_{k}p_k\log p_k
    $$

    - 头部集中度（如 Top1% 曝光占比）

## 1.7 统一符号与口径

- $u$：用户（user）
- $i$：物品（item）
- $c$：上下文（context）
- $\mathcal{I}$：全量候选库
- $\mathcal{S}(u,c)$：召回候选集
- $\pi$：最终展示列表
- $K$：展示长度（如 topK）

统一口径建议：

- CTR 口径：`click / exposure`
- CVR 口径：明确是 `purchase / click` 还是 `purchase / exposure`
- 新物品定义：按“上线天数 <= T”统一
- 长尾定义：按曝光分位（如后 80%）统一


# 二、召回层（Recall）

召回层的核心是“用最少计算找出尽可能多的高潜候选”。召回做得好，排序才有发挥空间。

## 2.1 召回层目标、输出与时延预算

  - 2.1.1 目标三角

    召回通常要平衡三件事：

    - 覆盖率（Recall）
    - 时延（Latency）
    - 成本（Compute/Storage）

    任何一项极端优化都会伤另外两项。

    更实用的做法是把“三角”变成可监控的三条曲线：

    - `Recall@N` 随候选深度增长曲线（收益）
    - `P95/P99` 随候选深度增长曲线（时延）
    - 实例 CPU/内存与索引体积曲线（成本）

    上线参数应选择“拐点区间”，而不是单点最优。

  - 2.1.2 输出定义

    召回输出不只是 item 列表，建议至少包含：

    ```text
    (item_id, route, raw_score, route_rank, reason, request_id)
    ```

    这样后面才可做融合解释、故障定位、策略审计。

    建议再补两个字段：

    - `feature_ts`：该候选分数对应的特征时间戳，便于排查“旧特征导致的误召回”。
    - `index_version`：命中的索引版本，便于灰度和回滚时定位问题。

  - 2.1.3 分路预算

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

## 2.2 召回通路框架

  - 2.2.1 协同过滤类通路

    核心思想：利用历史共现。

    - UserCF：相似用户迁移偏好
    - ItemCF：相似物品扩展

    典型相似度形式（ItemCF）：

    $$
    sim(i,j)=\frac{\sum_{u\in U_{ij}}w_u}{\sqrt{|U_i||U_j|}}
    $$

    其中 $U_{ij}$ 为同时交互过 $i,j$ 的用户集合。

  - 2.2.2 向量近邻类通路

    核心思想：把 user/item 映射到同一向量空间做近邻检索。

    输入/输出与框架：

    - 输入：用户特征 $x_u$、物品特征 $x_i$。
    - 输出：向量相似度分 $s(u,i)$ 与 TopK 候选。
    - 框架：先编码再检索，常见打分是点积：

    $$
    s(u,i)=\mathbf{u}^\top\mathbf{v}_i,\quad
    \mathbf{u}=f(x_u),\ \mathbf{v}_i=g(x_i)
    $$

    在线通常流程：

    ```text
    实时算 user 向量 -> ANN 检索 -> 返回 topK
    ```

    常见 ANN 方案：HNSW、IVF-PQ 等。关键是“召回率-时延”调参而不是单看检索速度。

  - 2.2.3 内容语义类通路

    核心思想：基于文本、标签、图像、音频等内容特征。适用场景：

    - 新物品冷启动
    - 行为数据稀疏
    - 强语义匹配需求

    内容通路常见打分形式：

    $$
    score_{content}(u,i)=\cos(\mathbf{e}^{text}_u,\mathbf{e}^{text}_i)+\eta\cdot \cos(\mathbf{e}^{vision}_u,\mathbf{e}^{vision}_i)
    $$

    其中 $\eta$ 是多模态权重。若业务是纯文本（例如资讯），可去掉视觉项。

    工程上要防“语义漂移”：内容 embedding 模型升级后，历史索引与新向量空间不一致，会导致召回突降。通常需要双索引灰度切换。

  - 2.2.4 热门/趋势/规则类通路

    用途：

    - 新用户兜底
    - 系统异常兜底
    - 活动/运营规则注入

    风险：热门通路权重过大时，会压制个性化和长尾探索。

    趋势分常见计算可写成：

    $$
    trend(i)=\frac{click(i,t_1,t_2)}{exposure(i,t_1,t_2)+\epsilon}\cdot freshness(i)
    $$

    其中 $freshness(i)$ 可按发布时间指数衰减。这样可以避免“旧爆款长期霸榜”。

## 2.3 多路融合、去重、截断

  - 2.3.1 融合前必须做的标准化

    因为跨路 raw score 通常不可比，需先归一化或改用 rank 融合。

    输入/输出与框架：

    - 输入：多路召回候选及其路由分数 `score_r(i)`。
    - 输出：统一可比较的融合分 `score(i)`，用于截断进入排序层。
    - 框架：先做分数标准化，再做加权融合（线性或 rank 融合）。

    常见融合：

    - 线性加权融合：

    $$
    score(i)=\sum_r w_r\cdot norm(score_r(i))
    $$

    - Rank 融合（RRF）：

    $$
    score(i)=\sum_r\frac{w_r}{k+rank_r(i)}
    $$

  - 2.3.2 去重策略

    去重时不要丢信息，建议保留：

    - source routes
    - 每路分数/名次
    - multi-hit 次数
    - 最终保留原因

    这样后续可以做“多路命中加权”和归因分析。

  - 2.3.3 场景化融合

    不同场景应动态配权：

    - 新用户：热门 + 内容权重高
    - 老用户：行为/向量权重高
    - 强时效场景：趋势/新鲜通路权重高

    一种常见的动态加权写法：

    $$
    w_r(u,c)=\mathrm{softmax}(h_r(\phi(u,c)))
    $$

    其中 $\phi(u,c)$ 是用户与上下文特征，$h_r$ 是第 $r$ 条路由的权重网络。这样权重不是人工常数，而是随用户状态变化。

  - 2.3.4 通路重叠与互补性分析

    通路之间既可能互补，也可能重复。建议固定监控路由重叠度（Jaccard）：

    $$
    J(A,B)=\frac{|A\cap B|}{|A\cup B|}
    $$

    解释方式：

    - 重叠过高：通路功能重复，浪费预算
    - 重叠过低：可能覆盖不同人群，但也可能质量不稳定

    建议同时看“独占命中贡献”，即某一路独有候选对最终点击/转化的贡献占比。

  - 2.3.5 截断策略

    每路截断深度建议按“边际收益”确定：当追加候选对最终指标增益趋近于 0 时停止扩深。可用离线曲线观察：`N=50/100/200/400` 时 Recall@N、后续精排增益、时延成本三者的拐点。可把边际收益显式写成：

    $$
    \Delta G(N)=G(N+\delta)-G(N)
    $$

    当  $\Delta G(N)<\epsilon$  且时延增量仍明显时，应停止扩深。这里  $G$  可取最终 CTR、GMV 或综合收益。

## 2.4 召回评估与诊断体系

  - 2.4.1 离线评估

    至少分三层看：

    - 总体：Recall@K、HitRate@K
    - 结构：新物品覆盖率、长尾覆盖率
    - 归因：每路独立增益、重叠度、互补性

    $$
    HitRate@K=\frac{1}{|U|}\sum_{u\in U}\mathbf{1}(|TopK_u\cap Rel_u|>0)
    $$

    $$
    Coverage@K=\frac{\left|\bigcup_{u\in U} TopK_u\right|}{|\mathcal{I}|}
    $$

    如果只看 Recall@K，容易忽视“全站内容是否被有效触达”。

  - 2.4.2 在线评估

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

  - 2.4.3 问题排查手册

    常见问题 -> 常见原因：

    - 召回突然下降：索引更新失败、上游特征空值、路由开关变更
    - 结果同质化：热门路过强、融合权重失衡、去重策略粗糙
    - 新内容进不来：增量索引延迟、内容路权重太低

    建议固定“三步排查顺序”，避免盲调：

    1. 先看系统层：路由超时率、索引版本、上游特征缺失率。
    2. 再看策略层：路由权重是否有异常变更、截断深度是否被误调低。
    3. 最后看数据层：新内容入库延迟、内容质量过滤是否误伤。

# 三、排序层（Ranking）

排序层是“候选上的精细效用估计器”，它决定了页面前几位内容的质量上限。

## 3.1 排序任务定义与目标拆解

给定候选集 $\mathcal{S}(u,c)$，学习评分函数：

$$
s_{uic}=f(u,i,c)
$$

$$
\pi(u,c)=\mathrm{sort}_{i\in\mathcal{S}(u,c)}(s_{uic})
$$

输入/输出与框架：

- 输入：召回候选集 $\mathcal{S}(u,c)$，以及用户特征、物品特征、上下文特征。
- 输出：每个候选的排序分 $s_{uic}$ 和最终排序列表 $\pi(u,c)$。
- 框架：编码特征得到候选分数，再按分数排序。常见实现是“单塔打分器”或“含交叉特征的打分网络”。

排序层目标通常拆为三类：

- 相关性目标：用户是否会点击/消费
- 价值目标：是否转化、收益多大
- 稳定目标：跨场景分布稳定、可校准

## 3.2 特征与样本框架

  - 3.2.1 特征分层

    推荐排序常用“分层特征”思维：

    - 用户层：画像、长期兴趣、短期会话
    - 物品层：质量、时效、内容信号
    - 上下文层：时间、设备、场景、入口
    - 交互层：用户与物品匹配关系

    建议每层再区分“静态特征”和“动态特征”：

    - 静态特征：更新慢、稳定性高（如类目、作者等级）。
    - 动态特征：更新快、信息量大（如近 5 分钟热度、会话序列）。

    动态特征延迟是线上排序效果的核心瓶颈，常见问题不是模型不够强，而是动态特征到达太慢。

  - 3.2.2 训练样本构造

    关键是定义好“样本空间”与“负样本语义”：

    - 曝光未点负样本：最贴近真实决策边界
    - 随机负样本：扩大覆盖
    - 难负样本：增强排序前列区分能力

    样本还要做：

    - 时间切分（防穿越）
    - 分桶均衡（防头部类压制）
    - 去重去噪（刷量与异常流量）

  - 3.2.3 延迟反馈处理

    转化/留存存在延迟，常见做法：

    - 截断窗口训练（牺牲部分样本）
    - 标签回填机制（延后训练）
    - 多任务拆分（短期目标 + 长期代理目标）

    延迟标签常见处理口径：

    - 训练时使用“成熟标签窗口”（例如曝光后 T+7 才纳入 CVR）
    - 线上监控使用“实时代理目标”（例如深度点击/加购）
    - 周期性用成熟标签回溯修正模型偏差

  - 3.2.4 多目标训练目标（框架）

    多任务常写成加权和：

    $$
    \mathcal{L}=\lambda_{ctr}\mathcal{L}_{ctr}+\lambda_{cvr}\mathcal{L}_{cvr}+\lambda_{wt}\mathcal{L}_{wt}
    $$

    权重可固定，也可按任务不确定性或梯度大小动态调整。上线时必须同步评估“主目标提升是否伤害护栏目标”。动态调权常见形式（不确定性加权示意）：

    $$
    \mathcal{L}=\sum_t\left(\frac{1}{2\sigma_t^2}\mathcal{L}_t+\log \sigma_t\right)
    $$

    其中 $\sigma_t$ 可学习，表示任务噪声水平。噪声大的任务会被自动降权，减轻任务冲突。

## 3.3 排序范式与损失函数

  - 3.3.1 Pointwise

    单样本监督，最常见是 BCE：

    输入/输出与框架：

    - 输入：单条样本 $(u,i,c,y)$，其中 $y\in\{0,1\}$。
    - 输出：单条概率 $\hat y$。
    - 框架：对每个候选独立打分，不显式比较同页候选之间关系。

    $$
    \mathcal{L}_{BCE}=-y\log\hat y-(1-y)\log(1-\hat y)
    $$

    优点：简单稳定；缺点：对列表前列质量不够敏感。

    实操里通常配合负采样比控制，例如 `1:4`、`1:10`。若负样本过多不重加权，会导致模型学到“全预测低概率”也能拿到不错损失值。

  - 3.3.2 Pairwise

    学习正负样本相对顺序，典型形式：

    输入/输出与框架：

    - 输入：同一上下文下的样本对 $(i^+,i^-)$。
    - 输出：相对顺序约束（希望 $s^+>s^-$）。
    - 框架：不直接学绝对概率，而是学“谁应该排在前面”。

    $$
    \mathcal{L}_{pair}= -\log\sigma(s^+ - s^-)
    $$

    优点：直接优化顺序；缺点：样本构造和采样策略更敏感。

    Pairwise 的关键在“配对质量”：  
    同曝光上下文下的 `(点击, 未点击)` 对价值最高；跨场景随机构对噪声大，可能训练出错误排序偏好。

  - 3.3.3 Listwise

    直接面向列表指标优化（如 NDCG 近似），更贴近线上体验，但训练复杂度更高。

    输入/输出与框架：

    - 输入：同一请求下的候选列表及其标签/收益。
    - 输出：整页列表的排序结果。
    - 框架：直接优化列表质量目标（前几位更重要），而非单条或样本对。

    排序范式选择建议：

    - 数据规模小、先求稳：Pointwise
    - 更关注前列顺序：Pairwise
    - 强调列表质量与页面体验：Listwise

    实践中常见组合：先 Pointwise 训练稳态模型，再用 Pairwise/Listwise 做增量精调。如果直接端到端 Listwise 训练，建议先离线做小流量回放验证;它对 batch 组织、位置偏差校正、显存成本都更敏感，工程复杂度明显高于 Pointwise。

## 3.4 排序评估、校准与一致性

  - 3.4.1 离线评估

    常用组合：

    - AUC（区分能力）
    - LogLoss/Brier（概率质量）
    - NDCG@K/MRR（前列排序质量）

    建议加两类“稳定性评估”：

    - 分群稳定性：新老用户、不同入口、不同类目是否同向提升。
    - 时间稳定性：按天或小时切片，防止只在某些时段有效。

  - 3.4.2 校准评估

    如果排序分数要用于收益决策、预算控制，必须看校准。

    常见检查：

    - Reliability Diagram
    - ECE
    - 分群校准（新老用户/类目/入口）

    很多线上事故来自“排序正确但概率失真”。例如模型能排对前后顺序，但把 0.05 预测成 0.20，会直接放大预算分配与竞价错误。

  - 3.4.3 离线在线一致性

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

输入/输出与框架：

- 输入：排序层输出列表、候选相关性分、业务约束（多样性/配额/风险）。
- 输出：满足约束后的最终展示列表。
- 框架：在相关性基线上叠加约束收益与惩罚项，再做列表级选择。

  - 常见可操作约束

    - 类目上限：某类最多 $m$ 个
    - 作者间隔：同作者最小间隔位数
    - 头部抑制：头部内容单页上限
    - 新品保底：新内容最少曝光配额

    约束配置建议分级：

    - 全局级：全站统一（如合规约束）。
    - 场景级：首页、详情页、搜索页分别配置。
    - 人群级：新用户、低活跃用户可用更强探索约束。

## 4.3 重排策略框架

  - 4.3.1 相关性-多样性平衡

    代表性框架（MMR 思路）：

    $$
    \arg\max_{i\in C\setminus S}[\lambda\,Rel(i)-(1-\lambda)\max_{j\in S}Sim(i,j)]
    $$

    输入/输出与框架：

    - 输入：候选集合 $C$、已选集合 $S$、相关性函数 $Rel$、相似度函数 $Sim$。
    - 输出：下一条入选内容 $i^*$（迭代构建整页）。
    - 框架：每一步都平衡“高相关”与“低重复”，避免同质化刷屏。

    含义：每一步选“既相关、又不太重复”的 item。参数   $\lambda$ 的调节建议：

    - 高意图场景（搜索、详情页相关推荐）：  $\lambda$ 取大，优先相关性。
    - 信息流场景（首页推荐）：  $\lambda$ 适当降低，提升多样性与探索。

  - 4.3.2 配额与公平框架

    - 按类目/创作者/商家做配额
    - 分群看曝光占比是否健康
    - 防止“强者愈强”导致生态退化

    供给侧集中度可用 Gini 粗衡量：

    $$
    Gini = \frac{\sum_{i=1}^{n}\sum_{j=1}^{n}|x_i-x_j|}{2n\sum_{i=1}^{n}x_i}
    $$

    其中 $x_i$ 可取作者或商家的曝光量。Gini 越高，说明曝光越集中。

  - 4.3.3 商业策略框架

    典型组合分：

    $$
    score_{rerank}=a\cdot rel+b\cdot value-c\cdot risk
    $$

    关键不是公式本身，而是“参数随场景动态化”。可把 `risk` 再拆成：

    $$
    risk=\gamma_1\cdot complaint+\gamma_2\cdot violation+\gamma_3\cdot churn\_proxy
    $$

    输入/输出与框架（学习式版本）：

    - 输入：用户上下文 + 候选列表（含排序分、内容特征、约束特征）。
    - 输出：每个候选的重排分 `rerank_score` 或直接输出新顺序。
    - 框架：可用轻量 MLP/GBDT，也可用 cross-encoder 结构对“用户-候选-列表上下文”联合编码后输出重排分。

    这样业务团队和风控团队可以分别维护子项权重，避免单一风险分不可解释。

## 4.4 重排评估与上线控制

  - 4.4.1 离线评估

    - 列表质量：NDCG@K
    - 多样性：ILD、类目覆盖率
    - 新颖性：新内容占比、长尾占比
    - 生态指标：作者/商家曝光分布

    ILD 一个常见定义：

    $$
    ILD=\frac{2}{K(K-1)}\sum_{a<b}(1-sim(i_a,i_b))
    $$

  - 4.4.2 在线评估

    - 用户体验：时长、负反馈率、返回率
    - 业务收益：CTR/CVR/GMV/广告收益
    - 生态健康：供给侧曝光 Gini、头尾比

    线上看板建议按“主指标 + 护栏指标 + 诊断指标”组织：

    - 主指标：CTR、时长、GMV（按场景选一到两个）。
    - 护栏指标：投诉率、拉黑率、内容安全命中率。
    - 诊断指标：约束命中率、重排幅度分布、回滚触发次数。

  - 4.4.3 上线节奏与回滚

    - 从小流量灰度开始
    - 先开软约束，再逐步加硬约束
    - 预设回滚触发条件（护栏触发即回滚）

    建议上线分三段：

    1. 影子评估：仅记录不生效，观察约束命中与分数漂移。
    2. 小流量灰度：验证主指标与护栏是否同时稳定。
    3. 全量放开：保留一键回滚和参数热修复能力。

# 五、主流模型介绍

统一符号约定：
- $u$：用户（user）
- $i,j$：物品（item）
- $c$：上下文（context）
- $x_u,x_i$：用户/物品原始特征（离散+连续）
- $\mathbf{e}$：embedding 向量
- $s(u,i)$：模型给用户 $u$ 与物品 $i$ 的相关性分数
- $\hat y$：预测概率（如 CTR 概率）
- $\mathcal{N}$：负样本集合
- $K$：候选或列表长度（TopK）

## 5.1 召回模型（Candidate Generation）

召回层目标：从海量库快速拿到高潜候选，强调覆盖与时延，不直接追求最终概率最准。

1. 协同过滤族（UserCF / ItemCF / Swing）
- 输入：
  - 用户历史行为集合 $I_u$（用户 $u$ 看过/点过/买过的物品集合）
  - 全体用户-物品交互图（或共现矩阵）
- 输出：
  - 对每个候选物品 $i$ 的召回分 $score(u,i)$
- 核心公式：
  - 用户相似度（UserCF）：
  $$
  \mathrm{sim}(u,v)=\frac{|I_u\cap I_v|}{\sqrt{|I_u||I_v|}}
  $$
  - ItemCF 打分：
  $$
  \mathrm{score}(u,i)=\sum_{j\in I_u}\mathrm{sim}(i,j)\cdot w_{u,j}
  $$
  - Swing：
  $$
  \mathrm{swing}(i,j)=\sum_{u,v\in U_i\cap U_j,\,u<v}\frac{1}{\alpha+|I_u\cap I_v|}
  $$
- 符号说明：
  - $I_u$：用户 $u$ 历史交互集合；$U_i$：交互过物品 $i$ 的用户集合
  - $w_{u,j}$：用户 $u$ 对历史物品 $j$ 的行为权重（可含时间衰减）
  - $\alpha$：平滑参数，用于抑制极端活跃用户噪声
- 创新点：从“纯共现计数”变成“归一化相似度 + 噪声抑制”。
- 工程要点：必须做去热门惩罚与时间衰减，否则长尾很难召回。

2. 双塔召回（DSSM / YouTube DNN）
- 输入：
  - 用户侧特征 $x_u$（画像、序列、上下文）
  - 物品侧特征 $x_i$（ID、类目、内容特征）
- 输出：
  - 用户向量 $\mathbf{u}$、物品向量 $\mathbf{v}_i$，以及匹配分 $s(u,i)$
- 核心公式：
  $$
  \mathbf{u}=f_\theta(x_u),\quad \mathbf{v}_i=g_\phi(x_i),\quad s(u,i)=\mathbf{u}^\top\mathbf{v}_i
  $$
  $$
  \mathcal{L}=-\log\frac{\exp(s(u,i^+)/\tau)}{\sum_{i\in\{i^+\}\cup\mathcal{N}}\exp(s(u,i)/\tau)}
  $$
- 符号说明：
  - $f_\theta,g_\phi$：用户塔/物品塔网络
  - $i^+$：正样本物品；$\mathcal{N}$：负样本集合；$\tau$：温度参数
- 创新点：把召回转成向量检索，训练与 ANN 检索天然对齐。
- 工程要点：Hard Negative 质量直接决定增益上限；索引更新频率要与内容更新频率匹配。

3. 多兴趣双塔（MIND / ComiRec）
- 输入：
  - 用户行为序列 embedding：$\{\mathbf{e}_1,\dots,\mathbf{e}_T\}$
  - 候选物品向量 $\mathbf{v}_i$
- 输出：
  - 用户多兴趣向量集合 $\{\mathbf{z}^{(1)}_u,\dots,\mathbf{z}^{(K)}_u\}$
  - 召回分 $score(u,i)$
- 大体框架（先理解流程，再看公式）：
  - 第一步：把用户历史行为“分组”为 $K$ 个兴趣簇（例如篮球簇、美妆簇）。
  - 第二步：每个兴趣簇聚合成一个兴趣向量 $\mathbf{z}^{(k)}_u$。
  - 第三步：候选物品 $i$ 与这 $K$ 个兴趣分别匹配，取最大匹配分作为该物品召回分。
- 核心公式：
  - 行为到兴趣的分配权重：
  $$
  c_{tk}=\frac{\exp(b_{tk})}{\sum_{k'=1}^{K}\exp(b_{tk'})}
  $$
  - 第 $k$ 个兴趣向量（加权聚合）：
  $$
  \mathbf{z}^{(k)}_u=\mathrm{squash}\left(\sum_{t=1}^{T} c_{tk}\mathbf{e}_t\right)
  $$
  - 候选物品最终匹配分：
  $$
  score(u,i)=\max_{k\in\{1,\dots,K\}}\mathbf{z}_u^{(k)\top}\mathbf{v}_i
  $$
- 符号说明（对应上面三步）：
  - $T$：用户历史行为长度；$K$：兴趣向量个数（可理解为“兴趣槽位数”）
  - $\mathbf{e}_t$：第 $t$ 个历史行为的向量表示
  - $b_{tk}$：第 $t$ 个行为对第 $k$ 个兴趣槽的原始相关分（越大越可能分到该兴趣）
  - $c_{tk}$：softmax 后的归一化权重，表示“行为 $t$ 属于兴趣 $k$ 的概率权重”
  - $\mathbf{z}^{(k)}_u$：用户在第 $k$ 个兴趣方向上的向量
  - $\mathbf{v}_i$：候选物品 $i$ 的向量
  - `squash`：向量压缩函数，保持方向信息并限制向量长度
- 创新点（和单兴趣双塔的区别）：
  - 单兴趣模型只有一个用户向量，容易把多主题偏好混在一起。
  - 多兴趣模型是“一个用户多个向量”，能让同一用户在不同主题上分别匹配。
- 工程要点（线上怎么跑）：
  - 常见做法是每个兴趣向量各取一批 TopN 候选，再合并去重。
  - $K$ 过小会混兴趣，$K$ 过大又会增加检索成本，通常需要按时延预算调参。

4. 序列召回（GRU4Rec / SASRec / BERT4Rec）
- 输入：
  - 按时间排序的行为序列 $S=(i_1,i_2,...,i_T)$
- 输出：
  - 用户当前兴趣状态向量（或下一物品分布）
- 核心公式：
  - GRU4Rec：
  $$
  \mathbf{h}_t=(1-\mathbf{z}_t)\odot\mathbf{h}_{t-1}+\mathbf{z}_t\odot\tilde{\mathbf{h}}_t
  $$
  - SASRec 注意力：
  $$
  \mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
  $$
  - BERT4Rec 掩码损失：
  $$
  \mathcal{L}_{mask}=-\sum_{t\in\mathcal{M}}\log p(i_t\mid S_{\setminus t})
  $$
- 符号说明：
  - $\mathbf{h}_t$：时刻 $t$ 的隐藏状态；$\mathbf{z}_t$：更新门
  - $Q,K,V$：query/key/value 矩阵；$d$：向量维度
  - $\mathcal{M}$：mask 位置集合；$S_{\setminus t}$：去掉位置 $t$ 的上下文
- 创新点：显式建模兴趣随时间迁移与长依赖关系。
- 工程要点：序列长度、行为类型权重、时间间隔特征通常比“换更大模型”更关键。

5. 图召回（LightGCN）
- 输入：
  - 用户-物品二分图 $\mathcal{G}$，节点初始 embedding
- 输出：
  - 多层传播后的用户/物品 embedding，用于打分 $s(u,i)$
- 核心公式：
  $$
  \mathbf{e}^{(k+1)}_u=\sum_{i\in\mathcal{N}(u)}\frac{1}{\sqrt{|\mathcal{N}(u)||\mathcal{N}(i)|}}\mathbf{e}^{(k)}_i
  $$
  $$
  \mathbf{e}^{(k+1)}_i=\sum_{u\in\mathcal{N}(i)}\frac{1}{\sqrt{|\mathcal{N}(i)||\mathcal{N}(u)|}}\mathbf{e}^{(k)}_u
  $$
  $$
  \mathbf{e}_u=\sum_{k=0}^{K}\alpha_k\mathbf{e}^{(k)}_u,
  \quad
  \mathcal{L}_{BPR}=-\sum\log\sigma\big(s(u,i^+)-s(u,i^-)\big)
  $$
- 符号说明：
  - $\mathcal{N}(u)$：用户 $u$ 的邻居物品集合；$\mathcal{N}(i)$：物品 $i$ 的邻居用户集合
  - $\alpha_k$：第 $k$ 层传播权重
  - $i^+,i^-$：正/负样本物品
- 创新点：去掉传统 GCN 的变换层和激活层，只保留协同传播。
- 工程要点：增量更新与热点节点稳定性是主要难点。

6. 生成式召回（DSI / Generative Retrieval / HSTU 路线）
- 输入：
  - 用户上下文 $x_u$（历史序列、画像、场景）
- 输出：
  - 物品 ID token 序列 $y=(y_1,...,y_T)$（对应最终物品）
- 核心公式：
  $$
  p(y\mid x_u)=\prod_{t=1}^{T}p(y_t\mid y_{<t},x_u),
  \quad
  \mathcal{L}=-\sum_{t=1}^{T}\log p(y_t\mid y_{<t},x_u)
  $$
- 符号说明：
  - $y_t$：第 $t$ 个生成 token；$y_{<t}$：前缀序列
  - $T$：token 序列长度
- 创新点：把“检索问题”改写成“条件生成问题”。
- 工程要点：需要前缀树（trie）约束解码，防止生成非法 ID 路径。

## 5.2 排序模型（Ranking）

排序层目标：对“召回候选”做更精细的效用估计。这里最关键的是：输入特征是谁的、怎么聚合成最终分数。

1. 线性与因子分解基线（LR / FM / FFM）
- 输入特征（按来源拆开）：
  - 用户侧：用户ID、年龄段、长期兴趣标签、历史活跃度
  - 物品侧：物品ID、类目、作者/商家、内容质量分
  - 上下文侧：时间段、设备、入口位、网络环境
  - 交互侧：用户与物品的历史关系（如是否看过该作者、近7天同类点击率）
- 聚合方式：
  - 先做特征编码（离散特征 one-hot/embedding，连续特征标准化）
  - 再拼接成统一向量 $\mathbf{x}$
- 输出：点击/转化概率 $\hat y$（或排序分）
- 核心公式：
  - LR（线性聚合）：
  $$
  \hat y=\sigma(\mathbf{w}^\top\mathbf{x}+b)
  $$
  - FM（二阶交互聚合）：
  $$
  \hat y=w_0+\sum_i w_ix_i+\sum_{i<j}\langle\mathbf{v}_i,\mathbf{v}_j\rangle x_ix_j
  $$
  - FFM（按 field 感知交互）：
  $$
  \hat y=w_0+\sum_i w_ix_i+\sum_{i<j}\langle\mathbf{v}_{i,f_j},\mathbf{v}_{j,f_i}\rangle x_ix_j
  $$
- 符号说明：
  - $f_i$：特征 $i$ 所属 field（如 user/item/context）
  - $\mathbf{v}_{i,f_j}$：特征 $i$ 在与 field $f_j$ 交互时使用的向量
- 大体框架：`多源特征 -> 拼接 -> 线性/二阶交互 -> 概率`

2. Wide&Deep 与 DeepFM（“手工交叉 + 自动交叉”）
- 输入特征（按分支）：
  - Wide 分支：人工设计的交叉特征 $\phi(x)$（如 `user_id × category_id` 的哈希交叉）
  - Deep 分支：各 field embedding + 连续特征
- 聚合方式：
  - Deep 分支先拼接：
  $$
  \mathbf{h}_0=[\mathbf{e}_1\|\mathbf{e}_2\|\cdots\|\mathbf{e}_F\|\mathbf{d}]
  $$
  其中 $\mathbf{e}_f$ 是第 $f$ 个离散 field 的 embedding，$\mathbf{d}$ 是连续特征向量
  - 经过 MLP：
  $$
  \mathbf{h}_l=\mathrm{ReLU}(W_l\mathbf{h}_{l-1}+\mathbf{b}_l)
  $$
- 输出：$\hat y$
- 核心公式：
  - Wide&Deep：
  $$
  \hat y=\sigma\big(\mathbf{w}_{wide}^\top\phi(x)+\mathbf{w}_{deep}^\top\mathbf{h}_L+b\big)
  $$
  - DeepFM：
  $$
  \hat y=\sigma\big(y_{FM}+y_{DNN}\big)
  $$
- 符号说明：
  - $y_{FM}$：由共享 embedding 计算的低阶交互分
  - $y_{DNN}$：由 MLP 学到的高阶非线性交互分
- 大体框架：`用户/物品/上下文特征 -> 两路（显式交叉 + 深度交叉）-> 分数融合`

3. DCN-v2 与 DLRM（工业大规模主力）
- 输入特征：
  - 稀疏离散 field（user_id、item_id、类目、作者等）
  - 稠密连续 field（统计特征、时效特征、质量特征）
- 聚合方式（DCN-v2）：
  - 先构造基础输入向量 $\mathbf{x}_0$
  - 通过 cross 层显式构造高阶交叉：
  $$
  \mathbf{x}_{l+1}=\mathbf{x}_l+\mathbf{x}_0\odot(W_l\mathbf{x}_l+\mathbf{b}_l)
  $$
  - 用低秩近似降低参数：$W_l\approx U_lV_l^\top$
- 聚合方式（DLRM）：
  - 稠密特征先过 bottom MLP 得到 $\mathbf{d}'$
  - 稀疏特征查 embedding 得到 $\{\mathbf{e}_1,...,\mathbf{e}_F\}$
  - 做 pairwise 点积交互：
  $$
  z_{ab}=\langle\mathbf{t}_a,\mathbf{t}_b\rangle,\quad \mathbf{t}\in\{\mathbf{d}',\mathbf{e}_1,...,\mathbf{e}_F\}
  $$
  - 再把交互结果送入 top MLP
- 输出：概率或排序分
- 大体框架：`多field特征 -> 显式交叉/点积交互 -> 顶层网络输出`

4. 行为序列排序（DIN / DIEN）
- 输入特征（谁参与注意力）：
  - 用户历史行为序列 $\{\mathbf{e}_1,...,\mathbf{e}_T\}$
  - 当前候选物品向量 $\mathbf{e}_{target}$
  - 其他静态用户特征与上下文特征（作为并行输入）
- 聚合方式（DIN）：
  - 用候选物品作为 query，对历史行为做注意力加权：
  $$
  \alpha_t=\mathrm{softmax}(a(\mathbf{e}_t,\mathbf{e}_{target})),\quad
  \mathbf{u}_{interest}=\sum_t\alpha_t\mathbf{e}_t
  $$
  - 最终打分输入常为：
  $$
  [\mathbf{u}_{interest}\|\mathbf{e}_{target}\|\mathbf{p}_u\|\mathbf{c}]
  $$
  其中 $\mathbf{p}_u$ 是用户静态画像，$\mathbf{c}$ 是上下文向量
- 聚合方式（DIEN）：
  - 先用 GRU 建模兴趣演化，再用 AUGRU 用注意力调制更新门：
  $$
  \mathbf{u}'_t=\alpha_t\cdot\mathbf{u}_t,
  \quad
  \mathbf{h}_t=(1-\mathbf{u}'_t)\odot\mathbf{h}_{t-1}+\mathbf{u}'_t\odot\tilde{\mathbf{h}}_t
  $$
- 输出：候选条件下的排序分
- 大体框架：`历史序列 + 当前候选 -> 注意力/演化聚合 -> 候选相关兴趣分`

5. 多任务排序（ESMM / MMoE / PLE）
- 输入特征：
  - 共享输入 $x_{shared}$：用户、物品、上下文的公共特征
  - 可选任务特有输入 $x_t$：仅对某任务有意义的特征
- 聚合方式（ESMM）：
  - 共享底层后，分别输出 CTR 分支和 CVR 分支：
  $$
  P(CTCVR\mid x)=P(CTR\mid x)\cdot P(CVR\mid click,x)
  $$
- 聚合方式（MMoE）：
  - 多个 expert 先提取共享表示，每个任务用 gate 加权：
  $$
  \mathbf{h}^{(t)}=\sum_{k=1}^{K}g_k^{(t)}(x)\cdot Expert_k(x)
  $$
- 聚合方式（PLE）：
  - 同时使用共享专家和任务专家，任务表示可写为：
  $$
  \mathbf{h}^{(t)}=\sum_{k\in\mathcal{S}}g_{k}^{(t)}E_k^{shared}(x)+\sum_{m\in\mathcal{T}_t}g_{m}^{(t)}E_m^{task}(x)
  $$
- 输出：多任务分数（CTR/CVR/时长等）
- 大体框架：`共享特征抽取 -> 任务特定聚合 -> 多目标联合输出`

6. 损失函数与校准（不是“只看一个 loss”）
- 输入：
  - Pointwise：单条样本 $(u,i,c,y)$
  - Pairwise：同请求下的正负对 $(i^+,i^-)$
  - Listwise：同请求下的候选列表
- 输出：
  - 可优化训练目标 + 可用于决策的校准概率
- 核心公式：
  - BCE（Pointwise）
  $$
  \mathcal{L}_{BCE}=-\frac{1}{N}\sum_n\big[y_n\log\hat y_n+(1-y_n)\log(1-\hat y_n)\big]
  $$
  - BPR（Pairwise）
  $$
  \mathcal{L}_{BPR}=-\sum\log\sigma\big(s(u,i^+)-s(u,i^-)\big)
  $$
  - ListMLE（Listwise）
  $$
  \mathcal{L}_{ListMLE}=-\log\prod_{r=1}^{n}\frac{\exp(s_{\pi_r})}{\sum_{j=r}^{n}\exp(s_{\pi_j})}
  $$
- 大体框架：`样本组织方式（单条/成对/成列表）决定训练目标，输出分数再做校准用于线上决策`。

## 5.3 重排模型（Re-ranking）

重排层输入是“排序后的列表”，重点是列表特征聚合，不是再做一遍独立 CTR 预测。

1. MMR 与 xQuAD（多样性重排）
- 输入特征是谁的：
  - 用户/请求侧：当前请求意图 $q$
  - 候选 item 侧：相关性分 $Rel(i)$（来自排序层）
  - item-item 侧：相似度 $Sim(i,j)$（来自内容 embedding、类目、作者等）
- 聚合方式：
  - 迭代构建列表，已选集合 $S$ 会影响下一步得分
- 核心公式：
  - MMR：
  $$
  i^*=\arg\max_{i\in C\setminus S}\left[\lambda Rel(i)-(1-\lambda)\max_{j\in S}Sim(i,j)\right]
  $$
  - xQuAD：
  $$
  score(i)=\lambda P(i\mid q)+(1-\lambda)\sum_z P(z\mid q)P(i\mid z)\prod_{j\in S}(1-P(j\mid z))
  $$
- 大体框架：`排序分做主信号 + 已选列表做去重/覆盖聚合 -> 迭代出最终列表`

2. 约束优化重排（配额/公平/商业）
- 输入特征是谁的：
  - item 特征：类目、作者、广告标记、新鲜度、风险分
  - list 特征：当前已选集合的类目占比、作者占比、广告占比等
- 聚合方式：
  - 把 item 属性在列表维度聚合成约束指标：
  $$
  C_m(\pi)=\frac{1}{K}\sum_{k=1}^{K} g_m(i_k)
  $$
  其中 $g_m(i_k)$ 是第 $m$ 个约束的指示函数/得分函数
- 核心公式：
  $$
  \max_{\pi}\;\mathbb{E}[R],
  \quad
  s.t.\;\mathbb{E}[C_m]\ge b_m
  $$
  $$
  \mathcal{L}(\pi,\lambda)=\mathbb{E}[R]+\sum_m\lambda_m(\mathbb{E}[C_m]-b_m)
  $$
- 大体框架：`排序分 + 列表约束聚合 -> 拉格朗日加权 -> 选出满足约束的列表`

3. 学习式重排（Neural Re-ranker / Cross-Encoder）
- 输入特征是谁的：
  - 用户特征 $\mathbf{u}$、上下文特征 $\mathbf{c}$
  - 候选 item 特征 $\mathbf{i}_k$
  - 列表上下文特征（已选项池化、类目直方图、位置特征）
- 聚合方式（两种常见）：
  - 轻量 MLP 方式：
  $$
  \mathbf{l}_{k}=\mathrm{Pool}(\{\mathbf{i}_j\}_{j\in S_{k-1}}),
  \quad
  score_k=MLP([\mathbf{u}\|\mathbf{c}\|\mathbf{i}_k\|\mathbf{l}_k\|pos_k])
  $$
  - Cross-Encoder 方式（更重）：
  $$
  \mathbf{h}_k=Transformer([u\_tokens;item\_tokens;list\_tokens]),
  \quad
  score_k=\mathbf{w}^\top\mathbf{h}_{k,[CLS]}
  $$
- 输出：重排分或直接新顺序
- 大体框架：`用户/候选/列表上下文联合编码 -> 输出 rerank 分 -> 重新排序`

4. Bandit / RL 重排
- 输入特征是谁的：
  - 状态 $s_t$：用户画像 + 会话特征 + 当前列表摘要
  - 候选动作 $a$：可选 item 或策略动作
- 列表摘要常见聚合：
  $$
  \mathbf{l}_t=\mathrm{mean}(\{\mathbf{i}_j\}_{j\in S_t})
  $$
  或用类目计数直方图/作者覆盖率向量
- 核心公式：
  - UCB：
  $$
  UCB_a=\hat\mu_a+c\sqrt{\frac{\ln t}{n_a}}
  $$
  - Thompson：
  $$
  \theta_a\sim Beta(\alpha_a,\beta_a),\quad a^*=\arg\max_a\theta_a
  $$
  - RL 目标：
  $$
  J(\pi)=\mathbb{E}_{\pi}\left[\sum_{t=1}^{T}\gamma^{t-1}r_t\right]
  $$
- 大体框架：`状态聚合 -> 选动作 -> 更新状态与奖励 -> 持续学习`

5. 反事实离线评估（IPS / SNIPS / DR）
- 输入特征是谁的：
  - 日志上下文 $x_n$（用户+上下文+候选信息）
  - 日志动作 $a_n$、回报 $r_n$
  - 日志策略概率 $\mu(a_n\mid x_n)$ 与新策略概率 $\pi(a_n\mid x_n)$
- 聚合方式：
  - 按样本重要性权重 $w_n=\pi/\mu$ 做加权汇总
- 核心公式：
  - IPS：
  $$
  \hat V_{IPS}=\frac{1}{N}\sum_{n=1}^{N}\frac{\pi(a_n\mid x_n)}{\mu(a_n\mid x_n)}r_n
  $$
  - SNIPS：
  $$
  \hat V_{SNIPS}=\frac{\sum_n w_nr_n}{\sum_n w_n}
  $$
  - DR：
  $$
  \hat V_{DR}=\frac{1}{N}\sum_n\left[\hat q(x_n,\pi)+\frac{\pi(a_n\mid x_n)}{\mu(a_n\mid x_n)}(r_n-\hat q(x_n,a_n))\right]
  $$
- 大体框架：`日志样本 -> 权重校正 -> 离线估计新策略价值`。
## 5.4 代表性与前沿模型（详细版）

这一节只做“模型级总览”，避免和 5.1-5.3 重复铺陈：
- 5.1-5.3 讲链路内方法细节
- 5.4 讲代表模型的端到端输入、聚合框架、关键公式和输出

1. YouTube DNN 双塔（代表：工业向量召回范式）
- 输入特征是谁的：
  - 用户侧 $x_u$：用户画像、最近行为、上下文（入口/时间）
  - 物品侧 $x_i$：物品ID、类目、内容特征
- 计算框架（怎么聚合）：
  - 用户塔把 $x_u$ 编码为向量 $\mathbf{u}$
  - 物品塔把 $x_i$ 编码为向量 $\mathbf{v}_i$
  - 用点积聚合匹配分，并用 ANN 做 TopK 检索
- 关键公式：
  $$
  \mathbf{u}=f_\theta(x_u),\quad \mathbf{v}_i=g_\phi(x_i),\quad s(u,i)=\mathbf{u}^\top\mathbf{v}_i
  $$
  $$
  \mathcal{L}=-\log\frac{\exp(s(u,i^+)/\tau)}{\sum_{j\in\{i^+\}\cup\mathcal{N}}\exp(s(u,j)/\tau)}
  $$
- 输出：
  - 召回候选 TopK 及其匹配分（供排序层使用）

2. DeepFM（代表：排序里的自动特征交叉）
- 输入特征是谁的：
  - 用户/物品/上下文多域离散特征 + 连续统计特征
- 计算框架（怎么聚合）：
  - 各 field 先 embedding
  - 同一份 embedding 同时进入 FM 分支（低阶交叉）与 DNN 分支（高阶交叉）
  - 两分支分数相加后过 sigmoid
- 关键公式：
  $$
  \hat y=\sigma(y_{FM}+y_{DNN})
  $$
  $$
  y_{FM}=w_0+\sum_i w_ix_i+\sum_{i<j}\langle\mathbf{v}_i,\mathbf{v}_j\rangle x_ix_j
  $$
- 输出：
  - 每个候选的 CTR 概率或排序分

3. DIN / DIEN（代表：候选感知兴趣建模）
- 输入特征是谁的：
  - 用户历史行为序列 $\{\mathbf{e}_1,...,\mathbf{e}_T\}$
  - 当前候选向量 $\mathbf{e}_{target}$
  - 静态用户特征 $\mathbf{p}_u$、上下文特征 $\mathbf{c}$
- 计算框架（怎么聚合）：
  - DIN：用 target 作为 query，对历史序列做注意力加权
  - DIEN：在 DIN 基础上加入兴趣演化（GRU + 注意力门控）
- 关键公式：
  $$
  \alpha_t=\mathrm{softmax}(a(\mathbf{e}_t,\mathbf{e}_{target})),\quad
  \mathbf{u}_{interest}=\sum_t\alpha_t\mathbf{e}_t
  $$
  $$
  \mathbf{u}'_t=\alpha_t\cdot\mathbf{u}_t,
  \quad
  \mathbf{h}_t=(1-\mathbf{u}'_t)\odot\mathbf{h}_{t-1}+\mathbf{u}'_t\odot\tilde{\mathbf{h}}_t
  $$
- 输出：
  - 候选条件下的兴趣表示与排序分（同一用户对不同候选分数可显著不同）

4. LightGCN（代表：图协同召回）
- 输入特征是谁的：
  - 用户-物品交互二分图 $\mathcal{G}$
  - 用户与物品初始 embedding
- 计算框架（怎么聚合）：
  - 沿图邻居做多层传播聚合
  - 各层 embedding 加权求和作为最终表示
- 关键公式：
  $$
  \mathbf{e}^{(k+1)}_u=\sum_{i\in\mathcal{N}(u)}\frac{1}{\sqrt{|\mathcal{N}(u)||\mathcal{N}(i)|}}\mathbf{e}^{(k)}_i,
  \quad
  \mathbf{e}^{(k+1)}_i=\sum_{u\in\mathcal{N}(i)}\frac{1}{\sqrt{|\mathcal{N}(i)||\mathcal{N}(u)|}}\mathbf{e}^{(k)}_u
  $$
  $$
  \mathbf{e}_u=\sum_{k=0}^{K}\alpha_k\mathbf{e}^{(k)}_u
  $$
- 输出：
  - 用户/物品最终图表示，用于召回打分或作为排序特征

5. DCN-v2 / DLRM（代表：工业大规模排序骨干）
- 输入特征是谁的：
  - 稀疏离散域（user/item/context 各 field）
  - 稠密连续域（统计、时效、质量等）
- 计算框架（怎么聚合）：
  - DCN-v2：显式 cross 层逐层构造高阶交叉，并用低秩/专家结构控参
  - DLRM：稀疏 embedding 与稠密向量做点积交互，再进 top MLP
- 关键公式：
  $$
  \mathbf{x}_{l+1}=\mathbf{x}_l+\mathbf{x}_0\odot(W_l\mathbf{x}_l+\mathbf{b}_l),\quad W_l\approx U_lV_l^\top
  $$
  $$
  z_{ab}=\langle\mathbf{t}_a,\mathbf{t}_b\rangle,\quad \mathbf{t}\in\{\mathbf{d}',\mathbf{e}_1,...,\mathbf{e}_F\}
  $$
- 输出：
  - 线上排序分（CTR/CVR/综合价值分）

6. Cross-Encoder Re-ranker（代表：重排联合建模）
- 输入特征是谁的：
  - 用户特征/文本 token
  - 候选 item 内容 token
  - 列表上下文 token（已选项摘要、位置信息）
- 计算框架（怎么聚合）：
  - 把用户、候选、列表上下文拼接成同一序列
  - 用 Transformer 联合编码后读取 [CLS] 向量作为重排分
- 关键公式：
  $$
  \mathbf{h}_k=Transformer([u\_tokens;item\_tokens;list\_tokens]),
  \quad
  score_k=\mathbf{w}^\top\mathbf{h}_{k,[CLS]}
  $$
- 输出：
  - rerank 分数或新顺序（强调列表级约束下的最终可展示结果）

7. 生成式推荐（DSI / Generative Retrieval / HSTU 路线，前沿）
- 输入特征是谁的：
  - 用户上下文 $x_u$（历史序列、用户状态、场景）
- 计算框架（怎么聚合）：
  - 不走“向量检索 + 排序”传统两段式
  - 直接自回归生成物品 token 序列（可配 trie 约束）
- 关键公式：
  $$
  p(y\mid x_u)=\prod_{t=1}^{T}p(y_t\mid y_{<t},x_u),
  \quad
  \mathcal{L}=-\sum_{t=1}^{T}\log p(y_t\mid y_{<t},x_u)
  $$
- 输出：
  - 物品 ID（或层级 token）序列，再映射为最终候选/结果

