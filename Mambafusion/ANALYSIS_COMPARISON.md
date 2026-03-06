# Mambafusion 相对原始 MambaFusion 的模型改动核对（基于代码 diff）

## 对比方式
- 上游基线仓库：`https://github.com/AutoLab-SAI-SJTU/MambaFusion.git`
- 本地对比目录：`/workspace/MambaFusion/Mambafusion`
- 采用 `git diff --no-index` 与 `diff -rq` 对关键文件做逐项对比。

## 结论概览（仅列与“门控融合 / 缺失模态 dropout / proxy”直接相关）
1. **新增了结构化模态 dropout（相机/激光）**，用于训练阶段随机“缺失”单模态输入，并保证两者不会同时被置零。实现于 `ModalityDropout`。
2. **新增了全局门控融合（GlobalGatedFusion）**，以样本级权重 `alpha=[alpha_cam, alpha_lidar]` 对两模态特征进行加权后再拼接，而不是固定直接拼接。
3. **新增了对齐质量 proxy（AlignmentProxy）输入门控网络**：目前真正实现的是 `feat_cosine`；`none` 与 `geom_overlap` 当前都返回全零向量（即几何重叠 proxy 代码尚未实装）。
4. **在 ConvFuser 中加入了可配置开关**，默认都关闭，因此默认行为保持和原 baseline 一致；在配置文件中新增对应参数。

## 具体代码证据

### 1) Modality Dropout
- 训练时分别按 `p_cam`、`p_lidar` 伯努利采样保留掩码，并做逐样本逐模态特征清零。
- 额外逻辑：若一次采样出现两模态都被丢弃，则随机保留其中一个，避免“全空输入”。

数学形式可写为（逐样本）：
- `m_cam ~ Bernoulli(1-p_cam)`, `m_lidar ~ Bernoulli(1-p_lidar)`
- 若 `m_cam=m_lidar=0`，则强制改为 `(1,0)` 或 `(0,1)`
- `z_cam' = m_cam * z_cam`, `z_lidar' = m_lidar * z_lidar`

这属于常见的模态随机失活训练（与 dropout / stochastic depth 思路一致），理论动机是降低模型对单一模态的过拟合、提升缺失模态鲁棒性。

### 2) Global Gated Fusion
- 使用全局池化后的向量作为门控输入：
  - `pool_cam = mean_HW(z_cam)`
  - `pool_lidar = mean_HW(z_lidar)`
- 输入 MLP 后经 softmax 得到两模态权重 `alpha`，满足 `alpha_cam+alpha_lidar=1`。
- 输出是**加权后再通道拼接**：
  - `z_fused = concat(alpha_cam * z_cam, alpha_lidar * z_lidar)`

这在数学上是“输入自适应加权融合”（mixture-of-experts 风格的简化版，两专家门控），比固定拼接多了一步“可信度重加权”。

### 3) Alignment Proxy
- `feat_cosine` 模式下：
  - 先 1x1 投影到统一维度；
  - 全局池化得到向量 `v_cam, v_lidar`；
  - 构造 proxy 特征 `[cos(v_cam,v_lidar), log||v_cam||, log||v_lidar||]`。
- 其数学动机：
  - 余弦项表示跨模态语义方向一致性；
  - 两个能量项可粗略表示当前样本中每个模态激活强度，辅助门控分配权重。

### 4) ConvFuser 接入与兼容性
- `ConvFuser` 新增 `USE_MODALITY_DROPOUT / USE_GATED_FUSION / USE_ALIGNMENT_PROXY` 等配置。
- 默认关闭上述开关时仍走原路径（直接 `torch.cat`），因此可做干净 A/B。
- 当开启 gated fusion 时，将 `fusion_gate_alpha` 写入 `batch_dict`，便于后续统计分析。

## 合理性与当前实现边界

### 有数学依据且实现自洽的部分
- **Modality dropout**：伯努利掩码 + 非全空约束，目标明确，且有单元测试统计保留率与非空性。
- **Softmax 门控融合**：标准可微加权机制，权重可解释、训练稳定（归一化约束）。
- **feat_cosine proxy**：属于“弱监督质量信号”注入门控网络，公式明确，维度与数值范围可检验。

### 需要谨慎解读/尚未完全落地的点
- `ALIGNMENT_PROXY_MODE="geom_overlap"` 在当前代码里与 `none` 一样返回零向量，**尚未实现几何重叠 proxy**。
- 这套 gated fusion 是样本级全局权重（非空间位置级 gate），表达能力比像素/BEV位置级门控更弱，但更轻量。
- 在 `ConvFuser` 中，门控分支位于 `use_vmamba=False` 的分支；若训练主路径主要依赖 `vmamba` 分支，则这些改动不会直接生效。

## 可复现实验建议（避免“凭感觉”）
- 按配置做分阶段对照：
  1. baseline（全部关闭）
  2. + gated fusion
  3. + modality dropout
  4. + alignment proxy(feat_cosine)
- 重点同时看：NDS/mAP、缺失模态鲁棒性（人为置空 cam/lidar）、以及 `fusion_gate_alpha` 分布是否与场景难度一致。
