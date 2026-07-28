# DeepSeek-V4 技术笔记

> 资料状态：以 DeepSeek 官方 V4 模型卡为架构事实来源；CSA/HCA 的具体缓存组织与部署细节参考 vLLM 对公开权重的技术说明。DeepSeek V4 于 2026-04 发布。

## 1. 模型定位与规模

DeepSeek V4 延续 DeepSeekMoE 与 Multi-Token Prediction（MTP）路线，但把重心转向 **百万 token 长上下文的缓存与注意力计算成本**。该系列包括两个模型：

| 型号 | 总参数 | 每 token 激活参数 | 上下文 |
| --- | ---: | ---: | ---: |
| DeepSeek-V4-Pro | 1.6T | 49B | 1M token |
| DeepSeek-V4-Flash | 285B | 13B | 1M token |

官方提供 Non-think、Think High 与 Think Max 三种推理模式。它们是服务侧的推理深度设置，不应与模型架构或训练阶段混为一谈。

## 2. 混合注意力：CSA + HCA + DSA + 局部窗口

V4 的核心更新是将长上下文问题拆成两个部分处理：

1. **KV cache 的容量**：CSA（Compressed Sparse Attention）沿序列维度压缩 KV；HCA（Heavily Compressed Attention）采用更高压缩率的稠密注意力。
2. **注意力计算量**：DSA（DeepSeek Sparse Attention）先通过索引选择相关 token，再仅对 Top-$k$ 位置进行主要注意力计算。
3. **局部细节与因果性**：短滑动窗口在未压缩 token 上工作，补回压缩边界附近的局部信息。

vLLM 的公开实现说明了两种压缩层：`c4a` 约以 4 倍压缩缓存，并配合 DSA；`c128a` 约以 128 倍压缩缓存。两者与 128-token 局部滑动窗口配合，避免高压缩层在早期 token 无法读取局部因果上下文的问题。

这不是传统的线性注意力：它仍保留内容寻址的注意力，只是对缓存状态和候选 token 做压缩/稀疏选择。

DSA 先检索、后精算：

$$
s_{ij}=q_i^\top\tilde k_j,\quad\mathcal I_i=\operatorname{TopK}_j(s_{ij},k),\qquad
o_i=\sum_{j\in\mathcal I_i\cup\mathcal W_i}\operatorname{softmax}_j(q_i^\top k_j/\sqrt d)v_j.
$$

远程访问由内容检索决定，局部窗口 $\mathcal W_i$ 保留近邻细节。

## 3. CSA/HCA 的工程含义

可以把 V4 的长上下文机制理解为三层缓存：

| 层次 | 保存内容 | 作用 |
| --- | --- | --- |
| 短窗口 | 最近的原始 token KV | 保留局部精细信息 |
| 压缩注意力缓存 | 多 token 聚合后的 KV | 以较小状态表示较远历史 |
| 稀疏索引 | 候选位置或索引状态 | 在长历史中选择需要精算的 token |

压缩会带来位置编码问题。共享 Key/Value 后，注意力输出会携带绝对旋转位置；实现需要在输出端应用 inverse RoPE，恢复只依赖相对位置的表示。这个细节是将 KV 共享与 RoPE 兼容的关键。

C4/C128 可从分块汇聚理解：

$$
\bar k_m=\sum_{r=0}^{R-1}a_{m,r}k_{mR+r},\qquad
\bar v_m=\sum_{r=0}^{R-1}a_{m,r}v_{mR+r},\quad R\in\{4,128\}.
$$

这说明压缩缓存，不代表公开压缩器的逐项实现。

## 4. mHC：流形约束的 Hyper-Connections

V4 引入 manifold-constrained Hyper-Connections（mHC）。官方描述为：将残差映射约束在双随机矩阵构成的 Birkhoff polytope 上。

- **直观目的**：让跨分支/跨层的残差信号组合保持更稳定，同时不把组合限制为简单恒等映射。
- **为什么是双随机约束**：行列和均为 1 的非负矩阵可以被看作“受控重分配”，避免残差通路在深层堆叠中任意放大或塌缩。
- **与普通残差的关系**：它仍是优化信号传播的残差设计，并不替代注意力或 MoE。

mHC 的约束可概括为

$$
H'=AH,\qquad A_{ij}\ge0,\quad A\mathbf1=\mathbf1,\quad A^\top\mathbf1=\mathbf1.
$$

双随机约束限制残差信号任意放大或坍缩。

## 5. MTP 与 Muon

- **MTP**：延续 DeepSeek-V3 的多 token 预测路线。它既作为训练辅助目标，也可在推理时提供 speculative decoding 的草稿 token。
- **Muon**：V4 官方模型卡列为训练优化器之一。Muon 对隐藏层矩阵更新进行近似正交化；它属于训练时的参数优化，不是推理注意力算法。

把这两点分开很重要：MTP 优化解码吞吐，Muon 优化训练收敛，CSA/HCA/DSA 优化长上下文注意力与缓存。

MTP 的训练目标为 $\mathcal L_{\rm MTP}=-\sum_{s=1}^{S}\lambda_s\sum_t\log p_\theta(x_{t+s}\mid x_{\le t},s)$。它给 speculative decoding 提供草稿；Muon 是训练优化器，例如 $\Delta W_t=-\eta\operatorname{Orth}(B_t)$，不是注意力算子。

## 6. 部署与阅读建议

- V4 属于异构注意力堆叠：不同层的压缩倍率与局部窗口状态不同，KV cache 分页、前后端分离 prefill/decode、prefix cache 都需要支持多种状态布局。
- 部署时还应区分模型权重精度、注意力缓存精度和索引缓存精度；它们可分别使用不同格式。
- 评估百万上下文时，除了精度指标，还应记录单请求 KV cache、prefill 延迟、每输出 token 时间（TPOT）和批量大小。

## 参考资料

- [DeepSeek V4 官方模型卡](https://www.deepseek.com/en/transparency/)
- [vLLM：DeepSeek V4 注意力与实现说明](https://vllm.ai/blog/2026/04/24/deepseek-v4/)


## 7. 扩展：从缓存层级到端到端推理

V4 的长上下文可以按四条路径理解：最近 token 走未压缩局部窗口；中远程 token 走 C4/C128 压缩缓存；DSA 索引器先选 Top-k 候选；候选再进入精确注意力。其目标是在容量、候选召回和局部保真之间分层取舍，而不只是减少 FLOPs。

prefill 主要构建压缩缓存和索引，decode 则主要承担每步查询、候选读取、局部窗口与 MoE 通信。因此部署应分开记录首 token 延迟、TPOT、索引耗时、候选数和 KV 占用。压缩模型的关键质量指标还应包括远程证据召回率：关键定义或工具结果没有被检索到时，后续精确注意力无法补救。

mHC、MTP、Muon 分别服务于深层信号稳定、草稿解码和训练收敛，它们不等同于缓存压缩。比较系统时应避免把所有收益都归因于某一个注意力变体。
