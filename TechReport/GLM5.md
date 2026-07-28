# GLM-5 / GLM-5.2 技术笔记

> 资料状态：GLM-5 的完整技术报告发表于 2026-02；GLM-5.2 是后续面向长程任务的更新，官方模型卡明确列出其新增的 1M 上下文、IndexShare 与 MTP 改进。本笔记将二者分开记录。

## 1. GLM-5：从通用编码到 Agentic Engineering

GLM-5 的目标是把“给出代码建议”的 vibe coding 推向可长程执行、反思和修复的软件工程智能体。模型以 MoE、长上下文、稀疏注意力和异步 Agent RL 为主轴。

| 项目 | GLM-5 技术报告配置 |
| --- | --- |
| 总参数 / 激活参数 | 744B / 40B |
| Transformer 层数 | 80 |
| MoE | 256 专家 |
| 基础模型训练 token | 28.5T |
| 中期训练上下文 | 从 4K 逐步扩展至 200K |
| 主注意力优化 | MLA、DSA（DeepSeek Sparse Attention） |
| 草稿解码 | 共享参数的 3 层 MTP |

## 2. 注意力与 Muon Split

### DSA：内容驱动的稀疏注意力

DSA 不使用固定滑动窗口来决定远程可见性，而是先根据内容选择重要 token，再进行细粒度稀疏注意力计算。报告强调其优势是：可从稠密模型继续训练得到，避免从头训练长上下文稀疏架构的成本；同时保留内容驱动的远程依赖选择。

### MLA-256 与 Muon Split

GLM-5 采用 MLA 来降低 KV cache，但报告发现常规 MLA 与 Muon 的组合在小型 latent KV 维度下可能不如 GQA。为此提出 **Muon Split**：将注意力上投影矩阵按头切分，再分别执行 Muon 的矩阵正交化。

- **作用**：让不同注意力头能够以不同尺度更新，稳定注意力 logits。
- **MLA-256**：通过增大单头维度、减少头数，在保持训练计算与参数量大致不变的前提下降低解码时的点积成本。
- **边界**：Muon Split 是优化器参数分组/矩阵切分策略，不是新注意力公式。

## 3. MTP：共享参数的多 token 预测

普通 MTP 若要预测后续多个 token，通常需要多个预测层，会线性增加参数和 KV 状态。GLM-5 在训练中让 3 个 MTP 层共享参数，以较小额外状态换取更长的 speculative decoding 接受长度。

这体现了一个常见工程原则：MTP 的价值不仅是辅助训练损失，也直接影响推理阶段草稿 token 的可接受比例和端到端吞吐。

## 4. 训练流水线：长上下文与异步 Agent RL

GLM-5 将训练划分为：

1. **预训练**：在大规模语料中尽早提高代码与推理占比。
2. **中期训练**：将窗口从 4K 扩展至 200K，并增加长上下文 agent 数据。
3. **后训练**：按 Reasoning RL → Agentic RL → General RL 的顺序推进。
4. **跨阶段蒸馏**：On-Policy Cross-Stage Distillation 用于减轻推理能力在后续通用/智能体训练中的遗忘。

其异步 RL 基础设施将 rollout 生成与训练解耦，使长轨迹采样不必被同步训练步骤阻塞；这是 Agent RL 能在更大规模环境中运行的系统前提。

## 5. GLM-5.2：面向 1M 上下文的增量更新

GLM-5.2 是 GLM-5.1 之后的长程任务更新。官方模型卡的重点是：

- **稳定 1M token 上下文**：面向长时程 coding 与工具工作流。
- **IndexShare**：让连续 4 层稀疏注意力复用同一 indexer；官方称在 1M 上下文时可将每 token FLOPs 降低约 2.9 倍。
- **MTP 改进**：官方称 speculative decoding 的接受长度最多提高 20%。
- **多档思考强度**：让服务侧在效果、时延和成本间调节。
- **开放权重**：官方模型卡标注为 MIT 许可；页面列出的权重文件总规模为 753B。与 GLM-5 报告的 744B/40B 配置不同，引用时应注明版本和来源，不应混用数值。

## 6. 学习时的对照

| 机制 | 主要解决的问题 | 不应混淆为 |
| --- | --- | --- |
| DSA | 长序列注意力计算 | 固定窗口稀疏注意力 |
| MLA / MLA-256 | KV cache 与解码成本 | 线性注意力 |
| Muon Split | 注意力投影矩阵训练稳定性 | 推理阶段优化 |
| MTP | speculative decoding 草稿质量 | 仅训练辅助任务 |
| IndexShare | 跨层稀疏索引计算复用 | 共享 KV cache |

## 参考资料

- [GLM-5 技术报告](https://arxiv.org/abs/2602.15763)
- [GLM-5.2 官方模型卡](https://huggingface.co/zai-org/GLM-5.2)
- [IndexCache / IndexShare 论文](https://arxiv.org/abs/2603.12201)

## 8. 公式化补充：DSA、Muon Split 与 IndexShare

DSA 的关键是把“索引”与“精算”分开：

$$
r_{ij}=g(q_i,\tilde k_j),\qquad
\mathcal C_i=\operatorname{TopK}_j(r_{ij},k),
$$

$$
o_i=\sum_{j\in\mathcal C_i\cup\mathcal W_i}
\frac{\exp(q_i^\top k_j/\sqrt d)}
{\sum_{u\in\mathcal C_i\cup\mathcal W_i}\exp(q_i^\top k_u/\sqrt d)}v_j.
$$

$g$ 是低成本的内容索引函数；只有候选 $\mathcal C_i$ 和局部窗口 $\mathcal W_i$ 进入完整注意力。若 $k\ll L$，主注意力计算可由 $O(L^2d)$ 降至约 $O(Lkd)$ 加索引开销。该式是阅读报告用的抽象，不是 DSA indexer 的完整公开实现。

MLA 的缓存压缩可概括为

$$
c_t^{KV}=x_tW_D,\qquad
k_t=c_t^{KV}W_K,\quad v_t=c_t^{KV}W_V.
$$

即缓存较小的 $c_t^{KV}$，并在需要时上投影。Muon Split 进一步把注意力投影按头切分；若 $B_{t,h}$ 是第 $h$ 个头的动量块，则概念更新为

$$
\Delta W_{t,h}=-\eta\,\operatorname{Orth}(B_{t,h}),\qquad
W_t=W_{t-1}+\operatorname{Concat}_h(\Delta W_{t,h}).
$$

分别正交化的目的，是避免不同头在同一个大矩阵中相互干扰更新尺度。它是训练期优化器策略，不是推理期注意力层。

共享参数 MTP 可写为

$$
\mathcal L_{\rm MTP}=-\sum_{s=1}^{S}\lambda_s\sum_t
\log p_\phi(x_{t+s}\mid x_{\le t},s).
$$

而 IndexShare 的“每 4 层复用一次索引”可用简化关系表示：

$$
I_l=I_{\,4\lfloor(l-1)/4\rfloor+1}.
$$

后式只表达调度方式：相邻 4 层共享候选索引，并不表示它们共享完整注意力输出或 KV cache。
