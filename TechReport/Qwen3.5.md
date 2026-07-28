# Qwen3.5 / Qwen3.6 技术笔记

> 资料状态：Qwen3.5 是原生多模态基础模型族；Qwen3.6 与其共享模型类型和混合注意力骨架。未公开训练细节不作推断。

## 1. 系列定位

Qwen3.5 从训练开始交错处理文本、图像和视频 token；文本骨干复用 Qwen3-Next 的线性注意力 decoder，视觉塔复用 Qwen3-VL 编码器。

| 系列 | 代表档位 | 定位 |
| --- | --- | --- |
| Qwen3.5 Dense | 9B、27B | 原生多模态、混合注意力 |
| Qwen3.5 MoE | 35B-A3B 等 | 稀疏 FFN 容量 |
| Qwen3.6 | 后续 Dense / MoE | agentic coding、推理历史 |

## 2. 3:1 混合注意力

每四个块使用三层 Gated DeltaNet 和一层 Gated Attention：

$$
3\times\text{Gated DeltaNet}\rightarrow1\times\text{Gated Attention}.
$$

递推路径可用抽象式理解：

$$
S_t=\alpha_t\odot S_{t-1}+\beta_tk_tv_t^\top-\gamma_tk_t(S_{t-1}^\top k_t)^\top,\qquad o_t=S_t^\top q_t.
$$

固定状态降低长历史成本，全局注意力周期性恢复精确内容寻址；该式不是未公开实现的逐项公式。

## 3. MoE、GQA 与 3.6

35B-A3B 公开配置为 40 层、256 专家，每 token 激活 8 个路由专家和 1 个共享专家；Gated Attention 使用 16 个 Q 头和 2 个 KV 头。

$$
\mathcal E_t=\operatorname{TopK}(\operatorname{softmax}(W_rx_t),8),\qquad
y_t=E_{\rm shared}(x_t)+\sum_{e\in\mathcal E_t}\tilde p_{t,e}E_e(x_t).
$$

MoE 降低 FFN 激活计算，GQA 降低 KV cache；二者针对不同瓶颈。

## 4. 多模态、MTP 与 Agent

多模态 RoPE 可概括为

$$
R_{\rm mm}(t,h,w)=R_t(t)\oplus R_h(h)\oplus R_w(w).
$$

MTP 目标为

$$
\mathcal L_{\rm MTP}=-\sum_{s=1}^{S}\lambda_s\sum_t\log p_\theta(x_{t+s}\mid x_{\le t},s).
$$

Qwen3.6 的标准窗口为 262,144 token，可扩展至约 1.01M。多轮 agent 仍受预算约束：

$$
|C_{\rm system}|+|C_{\rm task}|+|C_{\rm tool}|+|C_{\rm reasoning}|\le B.
$$

保留 reasoning history 不能替代工具输出摘要和文件引用定位。

## 5. 部署

- 混合注意力缺少专用 kernel 时会退回更慢、更占内存的 PyTorch 路径。
- 视觉/视频、GQA cache、MoE 路由与 MTP 需要推理框架共同支持。
- Qwen3.5 与 Qwen3.6 在 Transformers 中共用模型类型；迁移时须核对 chat template、thinking 字段与窗口配置。

## 参考资料

- [Qwen3.5 Transformers 文档](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
- [Qwen3.5-27B 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-27B)
- [Qwen3.5-35B-A3B 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
- [Qwen3.6-35B-A3B 官方模型卡](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)

## 7. 扩展：多模态输入链路与部署选择

Qwen3.5 的原生多模态不是只在末端拼接视觉特征：图像和视频先转为视觉 token，再与文本一起进入混合注意力 decoder。这让模型能在生成中反复访问视觉局部，但视觉 token 数也会直接占用上下文预算；长视频、高清图像与工具日志同时出现时，需要主动压缩输入。

Dense 档位适合延迟更可预测、部署简单的场景；MoE 档位提供更大容量，但引入路由与 all-to-all 通信。Qwen3.6 面向 coding workflow 的意义，也要放在 harness 中理解：工具结果、文件片段和失败尝试应带引用摘要保存，不能无界累积。

评测时建议拆开测视觉理解、文本推理、函数调用和跨轮状态保持；长上下文还应报告视觉 token 占比、prefill 时间、KV 占用和连续工具调用成功率。
