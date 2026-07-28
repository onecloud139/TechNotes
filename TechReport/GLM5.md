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

DSA 的抽象为

$$
r_{ij}=g(q_i,\tilde k_j),\quad\mathcal C_i=\operatorname{TopK}_j(r_{ij},k),\qquad
o_i=\sum_{j\in\mathcal C_i\cup\mathcal W_i}\operatorname{softmax}_j(q_i^\top k_j/\sqrt d)v_j.
$$

当 $k\ll L$ 时，主注意力由 $O(L^2d)$ 近似降至 $O(Lkd)$ 加索引开销。

### MLA-256 与 Muon Split

GLM-5 采用 MLA 来降低 KV cache，但报告发现常规 MLA 与 Muon 的组合在小型 latent KV 维度下可能不如 GQA。为此提出 **Muon Split**：将注意力上投影矩阵按头切分，再分别执行 Muon 的矩阵正交化。

- **作用**：让不同注意力头能够以不同尺度更新，稳定注意力 logits。
- **MLA-256**：通过增大单头维度、减少头数，在保持训练计算与参数量大致不变的前提下降低解码时的点积成本。
- **边界**：Muon Split 是优化器参数分组/矩阵切分策略，不是新注意力公式。

MLA 的潜变量缓存为 $c_t^{KV}=x_tW_D,\ k_t=c_t^{KV}W_K,\ v_t=c_t^{KV}W_V$。Muon Split 的分头概念更新为 $\Delta W_{t,h}=-\eta\operatorname{Orth}(B_{t,h})$；它是训练期策略。

## 3. MTP：共享参数的多 token 预测

普通 MTP 若要预测后续多个 token，通常需要多个预测层，会线性增加参数和 KV 状态。GLM-5 在训练中让 3 个 MTP 层共享参数，以较小额外状态换取更长的 speculative decoding 接受长度。

这体现了一个常见工程原则：MTP 的价值不仅是辅助训练损失，也直接影响推理阶段草稿 token 的可接受比例和端到端吞吐。

共享参数 MTP 的目标为 $\mathcal L_{\rm MTP}=-\sum_{s=1}^{S}\lambda_s\sum_t\log p_\phi(x_{t+s}\mid x_{\le t},s)$，其价值在于提高草稿 token 的连续接受长度。

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

IndexShare 的每四层索引复用可写成

$$
I_l=I_{\,4\lfloor(l-1)/4\rfloor+1}.
$$

它只共享候选索引，不共享完整注意力输出或 KV cache。

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


## 7. 扩展：从训练配方到 Agent 系统

GLM-5 的机制发生在不同时间尺度：DSA/MLA 处理每次注意力与 decode 成本，MTP 处理连续生成吞吐，异步 Agent RL 处理分钟到小时级环境交互。它们不是一个统一算子，调优与观测指标也应分开。

DSA 的真正难点是索引召回而非单纯减小候选数。代码 agent 中，远程函数定义、测试报错或工具结果未被索引到时，主注意力再精确也无法恢复。IndexShare 复用索引可省 FLOPs，但隐含相邻层对关键位置判断相近的假设。

共享 MTP 的端到端价值取决于平均接受长度和验证开销；草稿很长但频繁被拒绝并不会加速。异步 RL 则需要处理策略版本滞后、环境非确定性与奖励延迟，对代码任务还要隔离执行环境，避免缓存或外部服务污染奖励。
