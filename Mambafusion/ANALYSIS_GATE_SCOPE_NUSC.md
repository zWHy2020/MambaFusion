# MambaFusion 主实验是否跑 vmamba 分支？以及与 gate 作用域相关的数学分析（nuScenes）

> 本文档回答两个问题：
> 1) 上游项目主实验是否启用了 `FUSER` 的 `vmamba` 分支；
> 2) 你本地 `gate_fusion` 为样本级全局权重（非空间位置级）时，对 nuScenes 指标“维持/超过上游”的潜在影响与数学依据。

---

## A. 可核查证据：上游主实验里到底哪个 vmamba 在启用

### A1) 训练入口使用的配置文件

上游 README 的主训练命令使用：
- `cfgs/mambafusion_models/mamba_fusion.yaml`

因此判断“主实验开关”应优先看该 yaml。

### A2) 在该 yaml 中，`MM_BACKBONE` 与 `FUSER` 的 vmamba 状态不同

在 `tools/cfgs/mambafusion_models/mamba_fusion.yaml`：
- `MM_BACKBONE.USE_VMAMBA_PRETRAIN: True`（开启）
- `FUSER` 中 `# USE_VMAMBA: True` 是注释行（即未显式开启）

这说明：
- 上游主实验明确启用了 **MM_BACKBONE 的 vmamba 预训练分支**；
- 但**没有在主配置中显式启用 FUSER 的 vmamba 分支**（默认值取代码里的 `False`）。

### A3) 代码默认值验证（关键）

`ConvFuser` 里：
- `self.use_vmamba = model_cfg.get('USE_VMAMBA', False)`
- `forward` 中仅当 `self.use_vmamba=True` 才走 `mamba_forward`，否则直接 `torch.cat`。

因此在主配置未设置 `USE_VMAMBA: True` 时，`FUSER` 的 vmamba 分支不会进入。

### A4) 结论（关于“主实验是否跑 vmamba 分支”）

严格按上游公开主配置与默认代码推断：
- **跑了 vmamba 相关能力**：在 `MM_BACKBONE` 的 `USE_VMAMBA_PRETRAIN=True`；
- **但主 FUSER 并未跑 `use_vmamba=True` 那条融合分支**（除非作者另有未公开覆盖配置/命令行 patch）。

这是一条“配置+默认值”层面的可验证结论，不涉及猜测。

---

## B. 你本地 gate 的作用域：确实是“样本级全局”，不是“空间级”

你本地 `GlobalGatedFusion` 的计算是：
1. `pool_cam = mean_{H,W}(Z_cam)`，`pool_lidar = mean_{H,W}(Z_lidar)`
2. `alpha = softmax(MLP([pool_cam, pool_lidar, ...]))`，`alpha∈R^2`
3. 输出 `concat(alpha_cam * Z_cam, alpha_lidar * Z_lidar)`

所以每个样本只有一对门控权重 `(alpha_cam, alpha_lidar)`，与位置 `(x,y)` 无关，属于样本级全局 gate。

---

## C. 与“空间位置级 gate”相比的数学差异

设 BEV 特征为 `Z_c(x,y), Z_l(x,y)`。

### C1) 你的全局 gate

\[
\alpha = \text{softmax}(g(\text{pool}(Z_c),\text{pool}(Z_l),\xi)),\quad
F_{global}(x,y)=\big[\alpha_c Z_c(x,y),\;\alpha_l Z_l(x,y)\big]
\]

其中 `\xi` 可包含 mask/proxy。

### C2) 空间 gate（对照）

\[
\alpha(x,y)=\text{softmax}(h(Z_c,Z_l)_{x,y}),\quad
F_{spatial}(x,y)=\big[\alpha_c(x,y)Z_c(x,y),\;\alpha_l(x,y)Z_l(x,y)\big]
\]

### C3) 函数类关系

- 全局 gate 是空间 gate 的特例：令 `\alpha(x,y)\equiv\alpha` 即可；
- 因此空间 gate 的表达上界不低于全局 gate；
- 但空间 gate 参数/自由度更高，有限样本下估计方差通常更高，泛化不必然更优。

这给出“为什么全局 gate 可能稳，但上限可能受限”的数学依据。

---

## D. 对 nuScenes “维持/超过上游指标”的影响判断（条件性）

上游 README/论文摘要给出 val NDS 约 75.0。针对你本地模型，结合上面证据可得：

1. **不存在结构性“必涨”保证**：
   样本级全局 gate 不等价于空间自适应 gate，不能仅凭结构断言超越。

2. **存在“维持或小幅提升”的合理机制**：
   当整帧中某一模态整体质量变差（雨夜、曝光、激光回波异常）时，全局 gate 能以低方差方式重分配模态权重；与模态 dropout 联合可提升缺失模态鲁棒性。

3. **也存在“上限受限”的场景**：
   nuScenes 常见局部异质退化（局部遮挡、局部远距稀疏），全局标量无法在不同位置做差异化权重，理论上会损失一部分最优性。

4. **还需注意你的实现生效分支**：
   你本地 `ConvFuser` 中 gated fusion 在 `use_vmamba=False` 分支生效；若训练时主要走 `use_vmamba=True`，则该改动不会直接贡献最终指标。

因此，是否“维持甚至超过上游”只能通过受控 A/B 报告，不应由结构直推。

---

## E. 额外实现边界（避免误读）

`AlignmentProxy` 里 `geom_overlap` 当前返回零向量，和 `none` 等价；目前真正提供信息的是 `feat_cosine`。解释 proxy 效果时应与此保持一致。

---

## F. 建议的最小验证协议（同预算）

1. Baseline（不开 gate/dropout/proxy）
2. + global gate
3. + global gate + modality dropout
4. + global gate + modality dropout + feat_cosine proxy

并增加两组鲁棒性验证：
- val 屏蔽 camera；
- val 屏蔽 lidar。

若 #3/#4 在标准 val 与缺失模态 val 上同时不劣于 baseline，才可更可靠地支持“全局 gate 对维持/超过上游指标有帮助”。
