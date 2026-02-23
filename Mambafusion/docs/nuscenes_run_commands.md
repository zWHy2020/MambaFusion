# NuScenes 训练/验证/测试运行指令（MambaFusion 本地改进版）

本文件仅基于当前仓库代码可验证出的入口脚本与配置项整理，不依赖外部猜测。

## 1. 数据准备（生成 infos + gt database）
在 `Mambafusion/` 根目录执行：

```bash
python -m pcdet.datasets.nuscenes.nuscenes_dataset \
  --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml \
  --func create_nuscenes_infos \
  --version v1.0-trainval \
  --with_cam \
  --with_cam_gt
```

> 该命令会生成当前主配置所需的 `nuscenes_infos_10sweeps_*_withimg.pkl` 与 `nuscenes_dbinfos_10sweeps_withvelo.pkl` 等文件。

## 2. 训练
### 单卡训练
```bash
python tools/train.py \
  --cfg_file tools/cfgs/mambafusion_models/mamba_fusion.yaml \
  --extra_tag nuscenes_mambafusion_local \
  --workers 8
```

### 多卡分布式训练（示例 8 卡）
```bash
bash tools/scripts/dist_train.sh 8 \
  --cfg_file tools/cfgs/mambafusion_models/mamba_fusion.yaml \
  --extra_tag nuscenes_mambafusion_local \
  --workers 8
```

## 3. 验证（val）
### 单卡验证
```bash
python tools/test.py \
  --cfg_file tools/cfgs/mambafusion_models/mamba_fusion.yaml \
  --ckpt output/cfgs/mambafusion_models/mamba_fusion/nuscenes_mambafusion_local/ckpt/checkpoint_epoch_10.pth \
  --extra_tag nuscenes_mambafusion_local \
  --eval_tag val_eval \
  --workers 8
```

### 多卡分布式验证（示例 8 卡）
```bash
bash tools/scripts/dist_test.sh 8 \
  --cfg_file tools/cfgs/mambafusion_models/mamba_fusion.yaml \
  --ckpt output/cfgs/mambafusion_models/mamba_fusion/nuscenes_mambafusion_local/ckpt/checkpoint_epoch_10.pth \
  --extra_tag nuscenes_mambafusion_local \
  --eval_tag val_eval \
  --workers 8
```

## 4. 测试集推理（test server 提交）
若要跑 NuScenes 测试集，请将数据版本切到 `v1.0-test`，并把测试 split / info 路径通过 `--set` 覆盖，例如：

```bash
python tools/test.py \
  --cfg_file tools/cfgs/mambafusion_models/mamba_fusion.yaml \
  --ckpt output/cfgs/mambafusion_models/mamba_fusion/nuscenes_mambafusion_local/ckpt/checkpoint_epoch_10.pth \
  --extra_tag nuscenes_mambafusion_local \
  --eval_tag test_infer \
  --workers 8 \
  --set DATA_CONFIG.VERSION v1.0-test \
        DATA_CONFIG.DATA_SPLIT.test test \
        DATA_CONFIG.INFO_PATH.test "[nuscenes_infos_10sweeps_test_withimg.pkl]"
```

此时会导出 `results_nusc.json`，且程序会提示 `No ground-truth annotations for evaluation`（test 集无标注）。

## 5. 你新增的门控融合 / 模态 dropout / proxy 开关
在当前仓库主配置 `mamba_fusion.yaml` 里，默认是关闭的。可在训练命令中用 `--set` 打开：

```bash
python tools/train.py \
  --cfg_file tools/cfgs/mambafusion_models/mamba_fusion.yaml \
  --extra_tag nuscenes_mambafusion_mm \
  --workers 8 \
  --set MODEL.FUSER.USE_MODALITY_DROPOUT True \
        MODEL.FUSER.P_CAM 0.1 \
        MODEL.FUSER.P_LIDAR 0.05 \
        MODEL.FUSER.USE_GATED_FUSION True \
        MODEL.FUSER.USE_ALIGNMENT_PROXY True \
        MODEL.FUSER.ALIGNMENT_PROXY_MODE feat_cosine
```

当前代码中可用值至少包含 `none`、`geom_overlap`、`feat_cosine`，其中真正计算特征代理的是 `feat_cosine`。
