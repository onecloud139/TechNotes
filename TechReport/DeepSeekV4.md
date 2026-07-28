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

## 3. CSA/HCA 的工程含义

可以把 V4 的长上下文机制理解为三层缓存：

| 层次 | 保存内容 | 作用 |
| --- | --- | --- |
| 短窗口 | 最近的原始 token KV | 保留局部精细信息 |
| 压缩注意力缓存 | 多 token 聚合后的 KV | 以较小状态表示较远历史 |
| 稀疏索引 | 候选位置或索引状态 | 在长历史中选择需要精算的 token |

压缩会带来位置编码问题。共享 Key/Value 后，注意力输出会携带绝对旋转位置；实现需要在输出端应用 inverse RoPE，恢复只依赖相对位置的表示。这个细节是将 KV 共享与 RoPE 兼容的关键。

## 4. mHC：流形约束的 Hyper-Connections

V4 引入 manifold-constrained Hyper-Connections（mHC）。官方描述为：将残差映射约束在双随机矩阵构成的 Birkhoff polytope 上。

- **直观目的**：让跨分支/跨层的残差信号组合保持更稳定，同时不把组合限制为简单恒等映射。
- **为什么是双随机约束**：行列和均为 1 的非负矩阵可以被看作“受控重分配”，避免残差通路在深层堆叠中任意放大或塌缩。
- **与普通残差的关系**：它仍是优化信号传播的残差设计，并不替代注意力或 MoE。

## 5. MTP 与 Muon

- **MTP**：延续 DeepSeek-V3 的多 token 预测路线。它既作为训练辅助目标，也可在推理时提供 speculative decoding 的草稿 token。
- **Muon**：V4 官方模型卡列为训练优化器之一。Muon 对隐藏层矩阵更新进行近似正交化；它属于训练时的参数优化，不是推理注意力算法。

把这两点分开很重要：MTP 优化解码吞吐，Muon 优化训练收敛，CSA/HCA/DSA 优化长上下文注意力与缓存。

## 6. 部署与阅读建议

- V4 属于异构注意力堆叠：不同层的压缩倍率与局部窗口状态不同，KV cache 分页、前后端分离 prefill/decode、prefix cache 都需要支持多种状态布局。
- 部署时还应区分模型权重精度、注意力缓存精度和索引缓存精度；它们可分别使用不同格式。
- 评估百万上下文时，除了精度指标，还应记录单请求 KV cache、prefill 延迟、每输出 token 时间（TPOT）和批量大小。

## 参考资料

- [DeepSeek V4 官方模型卡](https://www.deepseek.com/en/transparency/)
- [vLLM：DeepSeek V4 注意力与实现说明](https://vllm.ai/blog/2026/04/24/deepseek-v4/)

## 7. 公式化补充：稀疏检索、压缩缓存与训练目标

DSA 可以看成两阶段注意力。先用便宜的索引键 $\tilde k_j$ 选择候选，再在候选集合上进行精确 softmax：

$$
s_{ij}=q_i^\top\tilde k_j,\qquad
\mathcal I_i=\operatorname{TopK}_j(s_{ij},k),
$$

$$
o_i=\sum_{j\in\mathcal I_i\cup\mathcal W_i}
\operatorname{softmax}_j\left(\frac{q_i^\top k_j}{\sqrt d}\right)v_j.
$$

$\mathcal W_i$ 是局部滑窗。于是远程访问由内容打分决定、近邻细节由未压缩窗口保证；主注意力从全部 $L$ 个位置缩小到约 $k+|\mathcal W_i|$ 个位置。上式是 DSA 的机制抽象，不是官方索引网络的逐项实现。

C4/C128 的压缩可从分块汇聚的视角理解：

$$
\bar k_m=\sum_{r=0}^{R-1}a_{m,r}k_{mR+r},\qquad
\bar v_m=\sum_{r=0}^{R-1}a_{m,r}v_{mR+r},\quad R\in\{4,128\}.
$$

这是解释“多个 token 以一个压缩状态保存”的概念式，不代表实际压缩器一定是线性加权。压缩降低缓存容量，但也可能模糊边界和局部因果关系，因此 V4 同时需要短滑窗和稀疏精算。

mHC 的约束集合可记为

$$
H'=AH,\qquad A_{ij}\ge0,\quad A\mathbf1=\mathbf1,\quad A^\top\mathbf1=\mathbf1.
$$

也就是残差信号在双随机矩阵内重分配：行、列和为 1 限制了总量的任意放大或坍缩。这说明其稳定化动机，不应误读为官方的完整参数化。

MTP 的训练目标则可写成

$$
\mathcal L_{\rm MTP}=-\sum_{s=1}^{S}\lambda_s\sum_t
\log p_\theta(x_{t+s}\mid x_{\le t},s).
$$

它既提供未来 token 的辅助监督，也能在推理时生成 speculative decoding 草稿；速度收益最终取决于草稿接受率，而不是预测步数本身。
