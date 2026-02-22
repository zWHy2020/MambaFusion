# MambaFusion

## Stage-1: Modality Dropout + Gated Fusion + Alignment Proxy

在 `FUSER` 配置中新增了以下开关（默认关闭，保持 baseline 行为）：

```yaml
FUSER:
  NAME: ConvFuser
  USE_MODALITY_DROPOUT: False
  P_CAM: 0.1
  P_LIDAR: 0.05
  USE_GATED_FUSION: False
  USE_ALIGNMENT_PROXY: False
  ALIGNMENT_PROXY_MODE: "none"   # "none" | "feat_cosine" | "geom_overlap"
  GATE_USE_MASK: True
  GATE_HIDDEN_DIM: 128
  GATE_PROJ_DIM: 64
```

推荐初始超参：`P_CAM=0.1`, `P_LIDAR=0.05`, `ALIGNMENT_PROXY_MODE="feat_cosine"`。

### 对照实验配置

1. **baseline**
   - `USE_MODALITY_DROPOUT=False`
   - `USE_GATED_FUSION=False`
   - `USE_ALIGNMENT_PROXY=False`

2. **gated fusion only**
   - `USE_GATED_FUSION=True`
   - `USE_MODALITY_DROPOUT=False`
   - `USE_ALIGNMENT_PROXY=False`

3. **gated fusion + modality dropout**
   - `USE_GATED_FUSION=True`
   - `USE_MODALITY_DROPOUT=True`
   - `P_CAM=0.1`, `P_LIDAR=0.05`
   - `USE_ALIGNMENT_PROXY=False`

4. **gated fusion + modality dropout + alignment proxy(feat_cosine)**
   - `USE_GATED_FUSION=True`
   - `USE_MODALITY_DROPOUT=True`
   - `P_CAM=0.1`, `P_LIDAR=0.05`
   - `USE_ALIGNMENT_PROXY=True`
   - `ALIGNMENT_PROXY_MODE="feat_cosine"`

### 运行命令（示例）

```bash
python tools/train.py --cfg_file tools/cfgs/mambafusion_models/mamba_fusion.yaml
```

可基于同一配置文件分别切换上述开关进行 A/B 实验。
