# Qwen3 / Qwen3.5 深度技术笔记

## 1. 架构：从 Dense 到极稀疏 MoE 

Qwen3 系列采用统一的 Transformer Decoder-only 架构作为基础，但在模型拓扑上提供两种截然不同的实现路径：传统 Dense（稠密激活）架构与创新的极稀疏 MoE（Mixture-of-Experts）架构。这种双轨设计并非简单的参数缩放，而是针对不同部署场景的计算效率与表达能力进行的深度权衡。

在 Dense 架构路径中，Qwen3 沿用并优化了前代验证有效的结构设计。以 Qwen3-32B 为例，模型包含 64 个 Transformer 层，隐藏层维度为 5120，注意力头采用 Grouped Query Attention（GQA）配置，设置 40 个查询头（Query Heads）与 8 个键值头（Key-Value Heads），这种 5:1 的压缩比例在保持推理速度的同时显著降低了 KV Cache 的内存占用。FFN 维度扩展至 13824，使用 SwiGLU 激活函数替代传统 ReLU，其门控机制（Gating Mechanism）通过额外的线性投影动态控制信息流。归一化层采用 RMSNorm（Root Mean Square Layer Normalization），在训练和推理前对输入进行缩放，公式为 $\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{Mean}(x^2) + \epsilon}} \odot \gamma$，其中 $\gamma$ 为可学习参数。位置编码使用 RoPE（Rotary Positional Embeddings），通过在查询和键向量上应用旋转矩阵注入相对位置信息。

MoE 架构则是 Qwen3-Next（即 Qwen3.5）的核心创新，其设计哲学是"总参数量极大，激活参数量极小"。以 Qwen3-Next-80B-A3B 为例，模型总参数量达 800 亿，但每个 token 仅激活 30 亿参数，激活比例约为 1:50，实现了大模型性能与小模型推理成本的帕累托最优。具体实现上，MoE 层包含 512 个路由专家（Routed Experts）和 1 个共享专家（Shared Expert），每个专家的 FFN 维度与 Dense 模型保持一致。关键创新在于路由策略：不同于传统 MoE 的"one-of-many"（从多个专家选一个），Qwen3-Next 采用"many-of-many"路由，每个 token 会被路由到 10 个专家进行处理。这种 Top-10 路由策略允许更丰富的特征组合，即使在极高稀疏度（50:1）下仍能保持性能不衰减。

负载均衡（Load Balancing）是 MoE 训练稳定性的关键。Qwen3 系列采用全局负载均衡损失（Global Load Balance Loss），定义辅助损失函数 $L_{aux} = \alpha \cdot \sum_{i=1}^{N} f_i \cdot P_i$，其中 $f_i$ 是路由到专家 $i$ 的 token 比例，$P_i$ 是路由器分配给专家 $i$ 的平均概率，$\alpha$ 为超参数（通常设为 0.01）。此外引入专家选择（Expert Choice）的变体机制，通过容量因子（Capacity Factor）限制每个专家处理的 token 数量，防止热门专家过载。共享专家（Shared Expert）机制强制所有 token 必须流经一个特定专家，该专家学习通用语言表示，减轻路由专家的负担。

注意力机制在长上下文建模中面临计算复杂度挑战。Qwen3-Next 采用混合注意力（Hybrid Attention）设计：在 48 层 Transformer 中交替使用传统 Grouped Query Attention（GQA）和线性注意力（Linear Attention）——具体为每 4 层使用 GQA，其余层使用 Gated Delta Networks 实现的线性注意力。标准自注意力的计算复杂度为 $O(n^2 \cdot d)$，其中 $n$ 为序列长度，$d$ 为维度；而线性注意力通过核技巧（Kernel Trick）将 softmax 注意力改写为 $\text{Attention}(Q,K,V) = \frac{\phi(Q)\phi(K)^T V}{\phi(Q)\phi(K)^T \mathbf{1}}$，其中 $\phi$ 为特征映射函数（如 ReLU 或多项式核），利用矩阵乘法的结合律将复杂度降至 $O(n \cdot d^2)$。Gated Delta Networks 在此基础上引入门控机制和残差连接，公式为 $h_t = \text{Gate}(x_t) \odot (x_t + \Delta(x_t))$，其中 $\Delta(x_t)$ 通过线性注意力的增量更新计算，使模型能够高效处理超过 26 万 token 的超长输入，同时保持对关键信息的精确聚焦而不会因长度增加而"遗忘"上下文。

位置编码的扩展是长上下文支持的技术基石。Qwen3 采用 ABF（Attention-Base Frequency）技术，将 RoPE 的基频（Base Frequency）从标准的 10000 提升至 1000000（即 $10^6$），公式为  $\theta_i = (10^6)^{-2i/d}$ ，其中 $i$ 为维度索引。高频基座使模型能更好区分长距离相对位置。同时引入 YARN（Yet another RoPE extensioN）技术，通过温度缩放（Temperature Scaling）和注意力缩放（Attention Scaling）修正长序列上的注意力分布，避免注意力熵（Attention Entropy）过低导致的注意力分散。Dual Chunk Attention（DCA）则针对超出预训练长度的序列，将长序列分块处理，通过块间注意力和块内注意力机制的结合，实现推理时 128K 甚至 260K+ token 的有效上下文窗口。

## 2. 预训练数据工程：36 万亿 token 的清洗与课程学习

Qwen3 的预训练语料总规模达 36 万亿（36 Trillion）token，覆盖 119 种自然语言和方言，包括英语、中文、欧洲主要语言以及低资源语言如斯瓦希里语、威尔士语等。数据处理并非简单的数据堆砌，而是经过严格质量筛选、毒性过滤、去重和课程学习（Curriculum Learning）的三阶段精细化工程。

数据清洗流程高度依赖模型自身的判别能力。团队使用 Qwen2.5-VL 视觉语言模型处理扫描版 PDF 文档和图文混排内容，通过 OCR 和版面分析提取结构化文本，相比传统基于规则的 PDF 解析器，准确率提升显著，特别是在处理学术论文的公式、表格和参考文献时。对于数学和代码数据，使用 Qwen2.5-Math 和 Qwen2.5-Coder 生成大规模合成数据（Synthetic Data），并通过质量评分模型过滤低质量样本。去重采用 MinHash-LSH 算法，在全局范围内移除重复段落和近似重复文档，防止模型记忆特定文本片段。

三阶段课程学习策略（Curriculum Learning）模拟人类的学习进程，从简单到复杂逐步提升数据难度。第一阶段（S1：通用基础训练）持续在超过 30 万亿 token 上进行训练，序列长度为 4096。数据配比经过精心设计：网页文本（过滤后的高质量 Common Crawl）占 50%，书籍和百科全书占 15%，学术论文占 10%，代码占 15%，多语言文本占 10%。这一阶段的数据侧重于语言流畅性、世界知识和基础逻辑，避免过早引入复杂推理任务导致模型陷入局部最优。

第二阶段（S2：推理能力强化）额外训练约 5 万亿高质量 token，序列长度仍为 4096。关键变化在于数据配比的重平衡：STEM（科学、技术、工程、数学）领域内容占比提升至 30%，代码数据提升至 25%，逻辑谜题和合成推理数据占 20%，而通用网页文本比例降至 25%。同时加速学习率衰减，采用余弦退火（Cosine Annealing）策略，将学习率从 1e-4 快速降至 1e-5，迫使模型在有限步骤内吸收高难度知识。此阶段特别注重数学证明的格式多样性（包括自然语言证明、形式化 Lean 证明等）和代码的跨语言覆盖（Python、C++、Java、JavaScript、Go 等）。

第三阶段（S3：长上下文扩展）使用数百亿 token 的高质量长文本数据，将序列长度扩展至 32768。数据来源包括长篇小说、技术文档、多轮对话记录和结构化长文本（如 JSON、XML、Markdown）。数据长度分布经过统计优化：75% 的文本长度在 16384-32768 token 之间，25% 在 4096-16384 token 之间，确保模型适应各种长度尺度。技术上，除了 ABF、YARN 和 DCA，还采用长度外推（Length Extrapolation）训练，通过逐步增加训练序列长度（从 4K 到 8K 到 32K），让模型逐步适应长距离依赖。

多语言处理并非简单的数据混合，而是考虑语言家族和书写系统的多样性。119 种语言被分为高资源语言（英语、中文等，各 1T+ token）、中资源语言（日语、韩语、德语、法语等，各 100B-500B token）和低资源语言（各 &lt;100B token）。对于低资源语言，采用回译（Back-translation）和跨语言迁移学习，利用高资源语言的知识提升低资源语言性能。特殊语言如阿拉伯语、希伯来语等从右至左书写的语言，以及泰语、缅甸语等无需空格分词的语言，都有专门的预处理流程。

## 3. 后训练对齐：从 SFT 到混合强化学习的四阶段流水线

后训练阶段的目标是将预训练模型转化为遵循人类指令、具备复杂推理能力且可控制推理深度的对齐模型。Qwen3 采用四阶段流水线：长思维链冷启动、推理强化学习、思维模式融合和通用强化学习，每个阶段都有明确的技术目标和优化策略。

**阶段一：长思维链冷启动（Long-CoT Cold Start）** 通过监督微调（SFT）为模型注入基本的逐步推理能力。训练数据包含 20 万+ 条高质量长思维链样本，覆盖数学推导（如 AMC 12、AIME 竞赛题的分步解答）、逻辑谜题（如骑士与骗子问题）、代码调试（多步骤错误定位与修复）和科学推理（物理、化学问题的链式思考）。数据构建采用拒绝采样（Rejection Sampling）配合人工审核：使用强模型（如 Qwen3-72B）生成多个候选推理路径，通过规则验证（如数学答案正确性、代码可执行性）筛选正确路径，再由人工标注者检查推理逻辑的连贯性和可读性。SFT 训练使用标准的因果语言建模损失，学习率为 1e-5，批量大小为 512，训练 3 个 epoch。此阶段后，模型学会使用"&lt;think&gt;"和"&lt;/think&gt;"标签包裹推理过程，形成规范的思维模式输出格式。

**阶段二：推理强化学习（Reasoning RL）** 针对数学和代码任务开展基于规则的强化学习（Rule-based RL）。不同于传统 RLHF 使用 Reward Model（RM）进行评分，Qwen3 采用更稳定的规则验证器（Rule-based Verifier）作为奖励信号来源。对于数学题，奖励函数定义为 $r = \mathbb{I}(answer\_correct) - 0.1 \times length\_penalty$，即答案正确得 1 分，并根据推理长度施加轻微惩罚防止冗长。对于代码题，使用单元测试（Unit Test）验证，通过所有测试用例得 1 分，部分通过按比例得分。算法上采用 GRPO（Group Relative Policy Optimization），这是一种 PPO 的变体，不需要单独训练 Critic 模型，而是通过采样一组候选答案（Group）的相对奖励来估计优势函数（Advantage），显著降低显存占用。训练时设置 KL 散度约束（KL Penalty）系数为 0.01，防止策略模型偏离 SFT 模型太远。此阶段在 AIME'24 和 LiveCodeBench 等基准上带来 10-15 个百分点的显著提升。

**阶段三：思维模式融合（Thinking Mode Fusion）** 是解决"一个模型，两种思维模式"的关键技术。训练数据为混合数据集，包含三类样本：带完整推理过程的问题（思维链数据，占 40%）、不带推理过程的直接回答（快速响应数据，占 40%）、以及显式指定模式的指令（如"请详细思考"或"请直接回答"，占 20%）。损失函数采用加权交叉熵，对推理部分的 token 施加更高权重（权重 1.2）以强化学习。技术挑战在于防止模式混淆（Mode Collapse），即模型在非思考任务上也过度生成推理过程。解决方案是引入模式标记（Mode Token）：在输入序列开头插入"&lt;thinking&gt;"或"&lt;not_thinking&gt;"标记（虽然实际实现可能通过系统提示或特殊 token 隐式控制），并通过注意力掩码（Attention Mask）确保模型在不同模式下的行为隔离。

"思考预算"（Thinking Budget）控制机制在此阶段建立。技术上通过在解码阶段设置最大生成长度限制，或在训练时使用动态长度裁剪（Dynamic Length Cropping）模拟不同预算约束。模型学会在有限预算内分配推理深度，例如当预算紧张时跳过验证步骤，预算充足时进行多路径验证。内部基准 ThinkFollow 包含 1000 条测试用例，评估模型根据用户意图（通过提示词如"快速回答"或"深入分析"）自动切换模式的能力，准确率从阶段二的 65% 提升至阶段四的 92%。

**阶段四：通用强化学习（General RL）** 将模型的能力从推理任务扩展到通用领域。训练数据覆盖 20 多个任务类别，包括指令遵循（Instruction Following，如格式转换、摘要提取）、工具使用（Tool Use，如函数调用、API 交互）、创意写作（Creative Writing，如故事续写、诗歌创作）、多轮对话（Multi-turn Dialogue）和安全对齐（Safety Alignment）。奖励模型采用混合策略：对于客观任务（如格式遵循）使用规则验证，对于主观任务使用奖励模型（Reward Model）评分。RM 基于 Qwen3-72B 初始化，在人类偏好数据（Human Preference Data）上训练，采用 Bradley-Terry 模型建模成对比较概率：$P(y_1 | g_t; y_2) = \sigma(r(x, y_1) - r(x, y_2))$，其中 $\sigma$ 为 sigmoid 函数。训练使用离线 PPO 变体，批次大小为 256，学习率为 5e-6，KL 系数设为 0.02。此阶段特别注重多语言泛化，确保 RM 的奖励信号在不同语言间保持一致，避免英语优化过度而其他语言退化。

## 4. 强到弱蒸馏：小模型的高效训练策略

对于较小规模的模型（0.6B、1.7B、4B 参数），完整的四阶段后训练成本过高且数据效率低下。Qwen3 采用强到弱蒸馏（Strong-to-Weak Distillation）策略，利用大模型（如 72B 或 235B）的知识高效训练小模型。

蒸馏分为两个阶段：离策略蒸馏（Off-policy Distillation）和在线策略蒸馏（On-policy Distillation）。离策略阶段，大模型作为教师生成高质量的 SFT 数据，包括长思维链和直接回答。关键技巧是多样化采样（Diverse Sampling）：通过调整温度参数（Temperature Scaling，T=0.7 到 T=1.2）和 Top-p 采样（Nucleus Sampling，p=0.9）生成多个候选答案，从中筛选最优解。对于数学题，保留所有正确且推理路径不同的答案；对于开放性问题，使用语义去重（Semantic Deduplication）保留多样性。小模型在这些数据上进行标准 SFT 训练。

在线策略阶段更为精细。小模型生成回答（Student Response），教师模型对同一问题生成回答（Teacher Response），通过最小化两者的分布差异（Distribution Matching）进行优化。损失函数结合标准语言建模损失和 KL 散度损失：$L = L_{ce}(y_{student}, y_{true}) + \beta \cdot D_{KL}(P_{student} || P_{teacher})$，其中 $\beta$ 为蒸馏温度系数（通常设为 0.5-1.0）。进一步采用逆向 KL 散度（Reverse KL）$D_{KL}(P_{teacher} || P_{student})$，这迫使学生模型覆盖教师模型的所有可能输出，避免过度自信（Over-confidence）。结果显示，经过蒸馏的 Qwen3-4B 在 AIME'24 上得分 28.5，超过直接 RL 训练的同等规模模型（得分约 15），且训练成本仅为 1/10。

对于极小模型（0.6B 和 1.7B），还采用层共享（Layer Sharing）和嵌入层压缩（Embedding Compression）技术。层共享让相邻 Transformer 层共享参数，减少 30% 参数量但保持 95% 性能；嵌入层压缩将词汇表从 150K 压缩至 50K（针对特定语言），或采用因子化嵌入（Factorized Embedding，将大词汇表拆分为两个较小矩阵的乘积）减少内存占用。

## 5. 多模态统一架构：Omni 与 VL 的技术实现

Qwen3-Omni 是统一的多模态大模型，支持文本、图像、音频、视频四种模态的输入，以及文本和语音两种模态的输出。其核心架构创新是 Thinker-Talker MoE 设计，将感知、推理与生成解耦。

**Thinker 模块** 承担多模态感知与高层推理职责。对于图像输入，使用 ViT（Vision Transformer）编码器提取特征，将图像分割为 14x14 像素的 patch，投影到语言模型的嵌入空间。对于音频，采用 AuT（Audio Tokenizer）编码器，以 12.5Hz 的频率（即每 80ms 生成一个 token）将原始音频波形离散化为语义 token。对于视频，均匀采样 16-32 帧，每帧独立编码后通过时间维度拼接。所有模态特征通过模态适配器（Modality Adapter，多层 MLP）对齐到文本嵌入空间，输入到基于 MoE 的 Thinker 进行联合推理。Thinker 的 MoE 配置与语言模型类似，但专家数量可能根据模态调整。

**Talker 模块** 专门负责语音生成，其输入不再直接依赖 Thinker 的文本输出（传统 TTS 系统的做法），而是基于 Thinker 的隐状态（Hidden States）和音频上下文进行条件生成，实现文本内容与语音风格的解耦。Talker 采用多码本（Multi-codebook）自回归模型，使用 RVQ（Residual Vector Quantization）技术，通常设置 8 个码本，每个码本负责不同层次的声学细节（如基频、频谱包络、相位信息）。生成时使用 Grouped AR（Grouped Autoregressive）策略，第一个码本自回归生成，后续码本并行生成，平衡质量与速度。

为实现超低延迟的实时语音对话，Qwen3-Omni 在解码流程上深度优化。采用轻量级因果 ConvNet 声码器 Code2Wav 替代传统 Vocoder，直接从离散 token 生成波形，避免复杂的谱图转换。支持分块流式推理（Chunk-based Streaming）：当 Thinker 生成部分文本时，Talker 可立即开始合成对应的语音片段，实现首包延迟（First Packet Latency）低至 234ms（包括网络传输和处理时间）。模型在超过 500 万小时的语音数据上训练，涵盖有声书、播客、对话录音等，支持 119 种语言的文本交互、19 种语言的语音识别（ASR）和 10 种语言的高质量语音合成（TTS），并具备零样本语音克隆（Zero-shot Voice Cloning）能力，仅需 3-10 秒的参考音频即可模仿说话人音色。

**Qwen3-VL** 专注于视觉语言任务，引入多项架构创新。**Interleaved MRoPE（Multimodal Rotary Position Embeddings）** 解决视频理解中的时空位置编码问题。传统 RoPE 将位置信息视为一维序列，而 MRoPE 将位置编码分解为时间（Temporal）、高度（Height）、宽度（Width）三个维度，分别应用旋转矩阵。对于视频帧序列，第 $t$ 帧的第 $(h,w)$ 个 patch 的位置编码为三个维度编码的交错组合，使模型能同时感知时间先后和空间相对位置，避免视频帧的"位置混淆"问题。

**DeepStack 机制** 解决视觉编码器与语言模型之间的语义鸿沟。传统方法仅使用视觉编码器的最后一层特征，导致细粒度视觉信息（如小物体、纹理细节）丢失。DeepStack 将视觉编码器（如 ViT）中间层的多层级特征通过轻量级 MLP 投影器注入语言模型的对应层级。具体而言，若视觉编码器有 24 层，语言模型有 48 层，则将视觉编码器的第 6、12、18、24 层输出分别注入语言模型的第 12、24、36、48 层。这种层级对齐（Layer-wise Alignment）在不增加序列长度（即不增加计算量）的前提下，显著提升了小物体检测和 OCR 的准确率。

**显式时间戳建模** 针对视频理解中的时序定位任务。不同于复杂的时间编码器，Qwen3-VL 采用显式文本时间戳（Explicit Text Timestamp），在视频帧特征前插入可学习的时间嵌入（如"&lt;time_00:01:23&gt;"），将时间信息作为文本 token 的一部分输入语言模型。这种方法简化了架构，同时使模型能精确到秒级定位视频事件，支持"描述第三分钟的内容"或"找出狗出现的时段"等时序推理任务。

## 6. 部署优化与工程实践

Qwen3-Next 的稀疏 MoE 架构在部署时需要特殊的工程优化以发挥硬件潜力。在 NVIDIA Blackwell 架构 GPU 上，利用第五代 NVLink（提供 1.8TB/s 的 GPU 直连带宽）可显著降低专家路由的通信延迟。采用专家并行（Expert Parallelism）策略，将 512 个专家分布到多个 GPU 上，通过 All-to-All 通信在 forward 过程中交换 token。配合 FP8（8-bit Floating Point）量化，在保持模型精度的同时，将激活值和权重压缩至 8 位，相比 FP16 减少 50% 显存占用，提升 20-30% 的推理吞吐量。

Cerebras 的 wafer-scale 引擎（CS-3）针对 Qwen3-32B 实现了实时推理优化。通过将模型权重静态存储在晶圆级芯片的 44GB SRAM 中，消除 HBM 带宽瓶颈，实现推理延迟低至 1.2 秒，比同类推理模型在 GPU 上的部署快 60 倍。这种架构特别适合需要极低延迟的交互式应用。

对于长上下文部署，采用 Dynamic Sparse Attention 技术，在序列长度超过 8K 时自动启用局部注意力（Local Attention）与全局注意力（Global Attention）的混合模式，将计算复杂度从 $O(n^2)$ 降至 $O(n \cdot w)$，其中 $w$ 为局部窗口大小（通常 4K）。配合 vLLM 的 PagedAttention 内存管理，支持单卡部署 128K 上下文模型。
