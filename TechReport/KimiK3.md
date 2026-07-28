# Kimi K3 技术笔记

> 资料状态：基于 Moonshot AI 于 2026-07 发布的官方模型仓库、Kimi Linear 技术报告与 Attention Residuals 论文整理。除非特别说明，参数均指 Kimi K3 主模型。

## 1. 一页概览

Kimi K3 是面向长程编程、知识工作和智能体任务的开权重原生多模态 MoE 模型。它的核心不是单一新算子，而是把 **Kimi Delta Attention（KDA）**、**Gated MLA**、**Attention Residuals（AttnRes）** 与 **Stable LatentMoE** 组合成一个兼顾长上下文、推理成本和跨层信息流的体系。

| 项目 | 官方配置 |
| --- | --- |
| 总参数 / 激活参数 | 2.8T / 104B |
| 层数 | 93（其中 1 层 Dense） |
| 注意力层 | 69 KDA + 24 Gated MLA |
| 注意力隐藏维度 / 头数 | 7168 / 96 |
| MoE | 896 路由专家，16 个激活，另有 2 个共享专家 |
| 上下文窗口 | 1,048,576 token |
| 模态 | 文本、图像；使用 MoonViT-V2 视觉编码器 |
| 量化训练 | 从 SFT 阶段起使用 MXFP4 权重与 MXFP8 激活的 QAT |

## 2. 序列混合：KDA 与 Gated MLA 的层间混合

### KDA：固定状态的线性注意力

KDA 是对 Gated DeltaNet 的扩展。标准全注意力需要保存长度随上下文增长的 KV cache；KDA 则为每个头维护固定大小的状态 $S_t$，通过写入门 $\beta_t$ 和**逐通道**的遗忘门 $\alpha_t$ 更新这个状态：

$$
S_t = (I - \beta_t k_t k_t^\top)\operatorname{Diag}(\alpha_t)S_{t-1} + \beta_t k_t v_t^\top,
\qquad o_t = S_t^\top q_t
$$

相比每头一个标量遗忘率，$\operatorname{Diag}(\alpha_t)$ 允许每个特征通道有独立的记忆寿命。这既提供隐式的位置/时序偏置，也让模型能更精细地删除过时关联。训练时可采用分块并行算法，解码时则只更新常数大小状态。

### 为什么仍保留 MLA

KDA 的线性状态对极长序列很省内存，但它是受限容量的循环记忆；Gated MLA 则仍提供全局、内容寻址式的信息交互，并通过潜变量压缩降低 KV cache 开销。K3 在层级上混用二者：让 KDA 覆盖大多数层以控制长上下文成本，再周期性插入 Gated MLA 保留全局交互能力。

这个设计也说明 KDA 和 MLA 的关系：**KDA 改变序列混合范式，MLA 压缩全局注意力的缓存**，二者并非同类替代方案。

## 3. 深层信息流：Attention Residuals（AttnRes）

AttnRes 是针对深层 PreNorm Transformer 中“残差信号被逐层稀释”的改造。其核心思想是让注意力层能以可学习权重访问跨层历史表示，而不只接收相邻层的单一路径残差。

- **目标**：使不同深度的输出幅度和梯度分布更均衡，改善深网络中的信息与梯度传播。
- **与普通残差的区别**：普通残差主要是 $h_l + f_l(h_l)$ 的相邻层捷径；AttnRes 增加跨层可学习的残差组合。
- **与 KDA 的分工**：KDA 解决 token 维度的长程状态管理；AttnRes 解决 layer 维度的信号传播。两者互补。

## 4. Stable LatentMoE：高稀疏度下的专家路由

K3 的 MoE 在每个 token 上只激活 16/896 个路由专家，同时始终经过 2 个共享专家。这样总参数可以扩大到 2.8T，而单 token 的有效计算保持在 104B 级别。

需要区分两件事：

- **路由稀疏性**主要减少 FFN 的激活计算；
- **KDA/MLA 混合**主要降低长上下文的注意力状态与带宽压力。

两者分别优化不同瓶颈，不能把“激活参数少”误解为模型整体不需要大规模通信或内存。

## 5. 长上下文、原生视觉与部署

- **1M 上下文**：KDA 的固定状态避免在每层都线性增长 KV cache；Gated MLA 则在必要层保留全局信息能力。
- **原生视觉**：MoonViT-V2 将图像编码为可与语言表示联合处理的视觉 token；K3 的公开说明将其定位为文本与图像的原生多模态模型。
- **量化感知训练**：从 SFT 开始采用 MXFP4/MXFP8，目的是让权重格式与部署硬件更一致，而不是只在推理末端做离线量化。
- **思维历史保留**：官方 API 要求在多轮对话、工具调用时将上一轮完整 assistant 消息回传，包括 `reasoning_content` 与 `tool_calls`；只回传最终文本会破坏其训练时的上下文格式。

## 6. 学习时应抓住的主线

1. K3 不是“纯线性注意力模型”，而是 KDA 与 Gated MLA 的层级混合模型。
2. 其长上下文效率来自 KDA 的固定状态、MLA 的 KV 压缩和量化/内核工程的共同作用。
3. AttnRes 针对深度方向的训练与表征传播；Stable LatentMoE 针对 FFN 参数规模与激活成本。
4. 看基准时应同时记录 agent harness、工具、推理档位和上下文管理策略，不能将不同配置的分数直接横比。

## 参考资料

- [Kimi K3 官方仓库与配置](https://github.com/MoonshotAI/Kimi-K3)
- [Kimi Linear / KDA 技术报告](https://arxiv.org/abs/2510.26692)
- [Attention Residuals](https://arxiv.org/abs/2603.15031)
- [FlashKDA 内核](https://github.com/MoonshotAI/FlashKDA)

## 7. 公式化补充：从 KDA 到 AttnRes 的两条记忆轴

KDA 的状态更新

$$
S_t=(I-\beta_tk_tk_t^\top)\operatorname{Diag}(\alpha_t)S_{t-1}
+\beta_tk_tv_t^\top,\qquad o_t=S_t^\top q_t
$$

可以分成“遗忘—覆盖—读取”三步理解。$\operatorname{Diag}(\alpha_t)$ 先逐通道衰减旧记忆；$I-\beta_tk_tk_t^\top$ 再沿当前 key 的方向修正已有状态，避免同一内容被无界累积；$\beta_tk_tv_t^\top$ 写入当前键值对。由于 $S_t$ 的尺寸与序列长度无关，decode 的记忆容量由状态维度而非上下文长度控制。

AttnRes 则作用于深度而不是时间。普通残差为

$$
h_l=h_{l-1}+f_{l-1}(h_{l-1}),
$$

而 AttnRes 从历史层输出中选择残差路径：

$$
h_l=\alpha_{0\to l}h_1+\sum_{i=1}^{l-1}\alpha_{i\to l}f_i(h_i),\qquad
\alpha_{i\to l}=
\frac{\exp(q_l^\top\operatorname{RMSNorm}(k_i))}
{\sum_{j=0}^{l-1}\exp(q_l^\top\operatorname{RMSNorm}(k_j))}.
$$

这些权重和为 1，因此不是把所有历史层无条件相加，而是根据当前层需要进行凸组合。KDA 处理“该记住哪些历史 token”，AttnRes 处理“该走哪些历史 layer 路径”。

对于 K3 的 $16/896$ 稀疏专家，路由可用下面的通用表达理解：

$$
\mathcal E_t=\operatorname{TopK}(\operatorname{softmax}(W_rx_t),16),\qquad
y_t=E_{\rm shared}(x_t)+\sum_{e\in\mathcal E_t}\tilde p_{t,e}E_e(x_t).
$$

该式解释计算稀疏性，并非 Stable LatentMoE 未公开细节的完整复现。
