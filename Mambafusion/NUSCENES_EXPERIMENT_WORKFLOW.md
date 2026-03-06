# Mambafusion 在 nuScenes 上的实验流程与指令（对齐上游 MambaFusion）

> 本文档只使用两类可核查信息：
> 1) 上游项目 `AutoLab-SAI-SJTU/MambaFusion` README 公布的实验流程与命令；
> 2) 你本地 `Mambafusion` 目录中的脚本/配置参数。
>
> 目标：给出你本地仓库可直接落地的 **完整实验流程 + 命令 + 参数注释**，避免臆测。

---

## 0) 目录约定

以下命令默认在仓库根目录执行：

```bash
cd /workspace/MambaFusion/Mambafusion
```

---

## 1) 环境安装（按上游流程）

上游 README 给出的核心安装步骤如下（CUDA 11.8 + torch2.1.0）：

```bash
conda create -n mambafusion python=3.8 -y
conda activate mambafusion

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install https://data.pyg.org/whl/torch-2.1.0%2Bcu118/torch_scatter-2.1.2%2Bpt21cu118-cp38-cp38-linux_x86_64.whl
pip install nuscenes-devkit==1.0.5

# repo develop
python setup.py develop
python mambafusion_setup.py develop

# custom ops
cd selective_scan && python setup.py develop && cd ..
cd mamba_diffv/mamba && python setup.py develop && cd ../..
```

### 注释
- `setup.py develop` 和两个子目录的 `develop` 用于注册自定义算子与包路径。
- 若环境中 PyTorch/CUDA 不匹配，通常会在编译自定义算子时报错。

---

## 2) 数据集准备（nuScenes trainval）

### 2.1 放置原始数据

目录结构需满足（与上游一致）：

```text
Mambafusion/
  data/
    nuscenes/
      v1.0-trainval/
        samples/
        sweeps/
        maps/
        v1.0-trainval/
```

### 2.2 生成 info 与 GT database（你本地代码支持的参数）

```bash
python -m pcdet.datasets.nuscenes.nuscenes_dataset \
  --func create_nuscenes_infos \
  --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml \
  --version v1.0-trainval \
  --with_cam \
  --with_cam_gt
```

可选：若机器内存足够，可开启共享内存加速数据读取：

```bash
python -m pcdet.datasets.nuscenes.nuscenes_dataset \
  --func create_nuscenes_infos \
  --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml \
  --version v1.0-trainval \
  --with_cam \
  --with_cam_gt \
  --share_memory
```

### 注释
- `--with_cam`：生成包含图像相关信息的 nuScenes info。
- `--with_cam_gt`：额外构建图像 GT database（用于采样增强相关流程）。
- `--share_memory`：把索引缓存到共享内存，速度更快但占用大。

---

## 3) 主实验训练（按上游 dist_train 入口）

上游主实验训练命令（4 卡）在你本地可直接对应为：

```bash
cd tools
bash scripts/dist_train.sh 4 \
  --cfg_file cfgs/mambafusion_models/mamba_fusion.yaml \
  --sync_bn \
  --pretrained_model ckpts/pretrained.pth \
  --logger_iter_interval 1000
```

### 注释（关键参数）
- `dist_train.sh 4`：用 4 GPU 启动 `torch.distributed.launch`。
- `--cfg_file`：主配置文件（MambaFusion 论文主实验对应配置）。
- `--sync_bn`：多卡同步 BN（分布式训练常用）。
- `--pretrained_model`：上游 README 建议先下载预训练权重并传入。
- `--logger_iter_interval 1000`：每 1000 iter 打一次日志。

### 输出位置
训练输出目录由 `tools/train.py` 规则决定，默认形如：

```text
Mambafusion/output/mambafusion_models/mamba_fusion/default/
```

其中 checkpoint 默认在该目录下 `ckpt/`。

---

## 4) 单 checkpoint 测试 / 验证

按上游流程：

```bash
cd tools
bash scripts/dist_test.sh 4 \
  --cfg_file cfgs/mambafusion_models/mamba_fusion.yaml \
  --ckpt /absolute/or/relative/path/to/checkpoint_epoch_XX.pth
```

### 注释
- `dist_test.sh` 同样走分布式入口，`NGPUS` 与可见卡数一致。
- `--ckpt` 建议写明确路径，避免工作目录变化导致找不到文件。

---

## 5) 你本地“门控融合/模态dropout/proxy”改动的实验开关

你本地在 `tools/cfgs/mambafusion_models/mamba_fusion.yaml` 中给 `FUSER` 新增了以下开关（默认关闭）：

- `USE_MODALITY_DROPOUT`
- `P_CAM`, `P_LIDAR`
- `USE_GATED_FUSION`
- `USE_ALIGNMENT_PROXY`
- `ALIGNMENT_PROXY_MODE`
- `GATE_USE_MASK`, `GATE_HIDDEN_DIM`, `GATE_PROJ_DIM`

### 推荐 A/B 运行方式

保持同一训练脚本，只改配置开关：

1. Baseline：全关
2. + Gated Fusion：`USE_GATED_FUSION=True`
3. + Dropout：再开 `USE_MODALITY_DROPOUT=True`
4. + Proxy：再开 `USE_ALIGNMENT_PROXY=True, ALIGNMENT_PROXY_MODE="feat_cosine"`

命令不变，仍是：

```bash
cd tools
bash scripts/dist_train.sh 4 --cfg_file cfgs/mambafusion_models/mamba_fusion.yaml --sync_bn --pretrained_model ckpts/pretrained.pth --logger_iter_interval 1000
```

> 仅通过改 yaml 来进行可复现对照，避免额外变量。

---

## 6) 常用单卡调试命令（快速 sanity check）

在你本地 `README` 里保留了单进程入口，可用于先排配置问题：

```bash
python tools/train.py --cfg_file tools/cfgs/mambafusion_models/mamba_fusion.yaml
```

### 注释
- 单卡调试更容易定位数据路径/依赖/算子编译问题。
- 真正论文复现实验建议回到分布式脚本。

---

## 7) 执行前检查清单（强烈建议）

1. `nvidia-smi`：确认 GPU 可见。
2. `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
3. `python -c "import pcdet"`：确认 develop 安装成功。
4. 检查 `data/nuscenes` 目录是否完整。
5. 检查 `ckpts/pretrained.pth` 是否存在（若使用预训练）。

---

## 8) 与上游流程的一致性说明

- 数据准备命令、训练命令、测试命令与上游 README 保持同构。
- 你本地新增内容主要是 `FUSER` 的可选开关，不改变默认 baseline 流程。
- 因此可先复现 baseline，再逐步打开新开关做增量实验。
