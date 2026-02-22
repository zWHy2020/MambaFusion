# scripts/inference_qwen.py
import torch
import torch.nn.functional as F
import argparse
import os
import sys
from PIL import Image
import numpy as np
import json
import glob
from transformers import AutoTokenizer
from tqdm import tqdm 

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.main_model_qwen import MultiModalPestDetectionWithQwen
import albumentations as A
from albumentations.pytorch import ToTensorV2

class QwenPredictor:
    """Qwen 模型推理器"""
    
    def __init__(self, checkpoint_path, qwen_path, data_root, use_hsi=True, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print(f"使用设备: {self.device}")
        print(f"加载检查点: {checkpoint_path}")
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 加载类别映射
        self.load_class_mapping(data_root)
        
        # 创建模型
        print("创建模型...")
        self.model = MultiModalPestDetectionWithQwen(
            num_classes=self.num_classes,
            qwen_path=qwen_path,
            use_hsi=use_hsi,
            use_lora=True
        )

        # 加载权重
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '')
            new_state_dict[new_key] = v
        
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese', trust_remote_code=True)
        self.rgb_transform = A.Compose([A.Resize(224, 224), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()])
        print("✓ 模型加载完成\n")
    
    def load_class_mapping(self, data_root):
        """加载类别映射"""
        mapping_path = os.path.join(data_root, 'class_mapping.json')
        print(f"从 {mapping_path} 加载类别映射...")
        
        if os.path.exists(mapping_path):
            with open(mapping_path, 'r', encoding='utf-8') as f:
                self.class_to_idx = json.load(f)
                self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        else:
            print(f"错误: 在 '{data_root}' 中找不到 class_mapping.json 文件！推理无法进行。")
            sys.exit(1)
        
        self.num_classes = len(self.idx_to_class)
        print(f"类别加载成功，共 {self.num_classes} 个类别。")

    def preprocess_rgb(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        transformed = self.rgb_transform(image=image)
        return transformed['image'].unsqueeze(0)
    
    def preprocess_hsi(self, hsi_path, use_hsi=True):
        if use_hsi and hsi_path and os.path.exists(hsi_path):
            hsi = np.load(hsi_path)
            hsi = (hsi - hsi.min()) / (hsi.max() - hsi.min() + 1e-8)
            hsi = torch.from_numpy(hsi).float()
            if hsi.dim() == 3 and hsi.shape[-1] > hsi.shape[0]:
                hsi = hsi.permute(2, 0, 1)
            return hsi.unsqueeze(0)
        else:
            return torch.zeros(1, 224, 64, 64)
    
    def preprocess_text(self, text):
        # --- 核心修改：使用与训练中文本遮蔽时相同的通用文本 ---
        if not text: 
            text = "一张需要分析的植物叶片图片。"
            
        encoded = self.tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        return encoded['input_ids'], encoded['attention_mask']
    
    @torch.no_grad()
    def predict(self, rgb_path, hsi_path=None, text=None, top_k=5):
        rgb = self.preprocess_rgb(rgb_path).to(self.device)
        hsi = self.preprocess_hsi(hsi_path, self.model.use_hsi).to(self.device)
        text_ids, text_mask = self.preprocess_text(text)
        text_ids, text_mask = text_ids.to(self.device), text_mask.to(self.device)
        
        outputs = self.model(rgb, hsi, text_ids, text_mask)
        logits = outputs['logits']
        probs = F.softmax(logits, dim=1)
        
        top_probs, top_indices = torch.topk(probs, min(top_k, self.num_classes), dim=1)
        top_probs, top_indices = top_probs.squeeze().cpu().numpy(), top_indices.squeeze().cpu().numpy()
        
        predictions = []
        if top_indices.ndim == 0:
            top_indices, top_probs = [top_indices], [top_probs]
            
        for idx, prob in zip(top_indices, top_probs):
            predictions.append({'class_id': int(idx), 'class_name': self.idx_to_class.get(int(idx), f'Unknown_{idx}'), 'confidence': float(prob)})
            
        return {'rgb_path': rgb_path, 'hsi_path': hsi_path, 'text': text, 'predictions': predictions, 'top_prediction': predictions[0]}
    
    def predict_batch(self, image_dir, output_file='predictions.json', pattern='*.jpg', text=None):
        image_paths = glob.glob(os.path.join(image_dir, pattern))
        print(f"找到 {len(image_paths)} 张图像")
        results = []
        for image_path in tqdm(image_paths, desc="批量推理"):
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            hsi_path = os.path.join(image_dir, base_name + '.npy')
            if not os.path.exists(hsi_path): hsi_path = None
            result = self.predict(image_path, hsi_path, text)
            results.append(result)
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        print(f"结果保存到: {output_file}")
        
        print("\n前5张图片预测结果示例:")
        for res in results[:5]:
            top = res['top_prediction']
            print(f"  - 文件: {os.path.basename(res['rgb_path'])}, 预测: {top['class_name']} (置信度: {top['confidence']:.4f})")
            
        return results

def parse_args():
    # ... (这部分代码无需修改，此处省略)
    parser = argparse.ArgumentParser(description='Qwen模型推理')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True, help='处理后的数据集根目录 (包含 class_mapping.json)')
    parser.add_argument('--qwen_path', type=str, default='./models/qwen2.5-7b')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--rgb_image', type=str, default=None)
    parser.add_argument('--hsi_image', type=str, default=None)
    parser.add_argument('--text', type=str, default=None)
    parser.add_argument('--image_dir', type=str, default=None)
    parser.add_argument('--pattern', type=str, default='*.jpg')
    parser.add_argument('--output', type=str, default='predictions.json')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--use_hsi', action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()

def main():
    # ... (这部分代码无需修改，此处省略)
    args = parse_args()
    predictor = QwenPredictor(checkpoint_path=args.checkpoint, qwen_path=args.qwen_path, data_root=args.data_root, device=args.device, use_hsi=args.use_hsi)
    if args.rgb_image:
        print("="*60 + "\n单张图像推理\n" + "="*60)
        result = predictor.predict(rgb_path=args.rgb_image, hsi_path=args.hsi_image, text=args.text, top_k=args.top_k)
        print(f"\n预测结果:\n图像: {args.rgb_image}\n\nTop-{args.top_k} 预测:")
        for i, pred in enumerate(result['predictions'], 1):
            print(f"  {i}. {pred['class_name']}: {pred['confidence']:.4f}")
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n结果保存到: {args.output}")
    elif args.image_dir:
        print("="*60 + "\n批量图像推理\n" + "="*60)
        results = predictor.predict_batch(image_dir=args.image_dir, output_file=args.output, pattern=args.pattern, text=args.text)
    else:
        print("请指定 --rgb_image 或 --image_dir 参数")

if __name__ == '__main__':
    main()