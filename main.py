#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文本匹配方法对比实验 - GPU优化版本 (RTX 4090)
修复了维度匹配问题
"""

import os
import json
import time
import random
import numpy as np
from typing import List, Dict
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from rank_bm25 import BM25Okapi
import jieba
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd


# ==================== 配置 ====================
@dataclass
class Config:
    data_dir: str = "C:/Users/PC/Desktop/data"
    train_file: str = "train.jsonl"
    val_file: str = "validation.jsonl"
    test_file: str = "test.jsonl"

    output_dir: str = "./results"
    batch_size: int = 128  # RTX 4090 大显存
    num_epochs: int = 5
    max_length: int = 32
    seed: int = 42

    # GPU配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True
    num_workers: int = 4

    # 神经网络参数
    embed_dim: int = 128
    hidden_dim: int = 256
    learning_rate: float = 0.001


config = Config()


# ==================== 工具函数 ====================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def load_data(data_path: str) -> List[Dict]:
    """加载JSONL数据，自动处理中文引号"""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    line = line.replace('“', '"').replace('”', '"')
                    line = line.replace('‘', "'").replace('’', "'")
                    line = line.replace('：', ':')
                    data.append(json.loads(line.strip()))
                except:
                    try:
                        line = line.replace('"', '').replace('"', '')
                        parts = line.split(',')
                        s1 = parts[0].split(':')[1].strip().strip('"')
                        s2 = parts[1].split(':')[1].strip().strip('"')
                        label = int(parts[2].split(':')[1].strip())
                        data.append({'sentence1': s1, 'sentence2': s2, 'label': label})
                    except:
                        continue
    return data


def tokenize_chinese(text: str) -> List[str]:
    return list(jieba.cut(text))


def evaluate(predictions, labels):
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='weighted'),
        'precision': precision_score(labels, predictions, average='weighted'),
        'recall': recall_score(labels, predictions, average='weighted')
    }


# ==================== GPU数据集 ====================
class TextMatchingDataset(Dataset):
    def __init__(self, data, word2idx, max_length=32):
        self.data = data
        self.word2idx = word2idx
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def encode_text(self, text):
        words = tokenize_chinese(text)
        ids = [self.word2idx.get(word, 1) for word in words]
        if len(ids) > self.max_length:
            ids = ids[:self.max_length]
        while len(ids) < self.max_length:
            ids.append(0)
        return ids

    def __getitem__(self, idx):
        item = self.data[idx]
        s1 = torch.tensor(self.encode_text(item['sentence1']), dtype=torch.long)
        s2 = torch.tensor(self.encode_text(item['sentence2']), dtype=torch.long)
        label = torch.tensor(item['label'], dtype=torch.long)
        return s1, s2, label


# ==================== 神经网络模型 (修复维度) ====================
class GPUTextMatcher(nn.Module):
    """GPU优化的文本匹配模型 - 修复维度问题"""

    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(0.3)

        # 双向LSTM
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=0.3
        )

        # 分类器 - 修正维度
        # 双向LSTM的隐藏维度是 hidden_dim * 2
        # 两个句子拼接后是 (hidden_dim * 2) * 2 = hidden_dim * 4
        lstm_output_dim = hidden_dim * 2  # 双向
        combined_dim = lstm_output_dim * 2  # 两个句子拼接

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, s1, s2):
        # 嵌入
        emb1 = self.embedding(s1)
        emb2 = self.embedding(s2)
        emb1 = self.embedding_dropout(emb1)
        emb2 = self.embedding_dropout(emb2)

        # LSTM编码
        _, (h1, _) = self.lstm(emb1)
        _, (h2, _) = self.lstm(emb2)

        # 获取最后一层的隐藏状态 (双向)
        # h1 shape: [num_layers * 2, batch, hidden_dim]
        # 取最后一层的前向和后向
        h1_last = torch.cat([h1[-2], h1[-1]], dim=1)  # [batch, hidden_dim * 2]
        h2_last = torch.cat([h2[-2], h2[-1]], dim=1)  # [batch, hidden_dim * 2]

        # 拼接两个句子的表示
        combined = torch.cat([h1_last, h2_last], dim=1)  # [batch, hidden_dim * 4]

        return self.classifier(combined)


# ==================== 方法1: 编辑距离 ====================
def run_edit_distance(test_data):
    print("\n" + "=" * 60)
    print("方法1: 编辑距离 (Edit Distance)")
    print("=" * 60)

    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]

    def similarity(s1, s2):
        dist = levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        return 1 - dist / max_len if max_len > 0 else 1.0

    labels = [item['label'] for item in test_data]
    start_time = time.time()
    predictions = []
    for item in tqdm(test_data, desc="  计算编辑距离"):
        sim = similarity(item['sentence1'], item['sentence2'])
        predictions.append(1 if sim >= 0.6 else 0)
    elapsed = time.time() - start_time

    metrics = evaluate(predictions, labels)
    metrics['time'] = elapsed
    print(f"  准确率: {metrics['accuracy']:.4f}")
    print(f"  F1分数: {metrics['f1']:.4f}")
    print(f"  耗时: {elapsed:.4f}s")
    return metrics


# ==================== 方法2: TF-IDF ====================
def run_tfidf(test_data):
    print("\n" + "=" * 60)
    print("方法2: TF-IDF")
    print("=" * 60)

    all_sentences = [item['sentence1'] for item in test_data] + [item['sentence2'] for item in test_data]
    vectorizer = TfidfVectorizer(tokenizer=tokenize_chinese, token_pattern=None, ngram_range=(1, 2), max_features=5000)
    vectorizer.fit(all_sentences)

    labels = [item['label'] for item in test_data]
    start_time = time.time()
    predictions = []
    for item in tqdm(test_data, desc="  计算TF-IDF"):
        vecs = vectorizer.transform([item['sentence1'], item['sentence2']])
        sim = cosine_similarity(vecs[0:1], vecs[1:2])[0][0]
        predictions.append(1 if sim >= 0.5 else 0)
    elapsed = time.time() - start_time

    metrics = evaluate(predictions, labels)
    metrics['time'] = elapsed
    print(f"  准确率: {metrics['accuracy']:.4f}")
    print(f"  F1分数: {metrics['f1']:.4f}")
    print(f"  耗时: {elapsed:.4f}s")
    return metrics


# ==================== 方法3: BM25 ====================
def run_bm25(test_data):
    print("\n" + "=" * 60)
    print("方法3: BM25")
    print("=" * 60)

    all_sentences = [item['sentence1'] for item in test_data] + [item['sentence2'] for item in test_data]
    tokenized_corpus = [tokenize_chinese(sent) for sent in all_sentences]
    bm25 = BM25Okapi(tokenized_corpus)

    labels = [item['label'] for item in test_data]
    start_time = time.time()
    predictions = []
    for item in tqdm(test_data, desc="  计算BM25"):
        query = tokenize_chinese(item['sentence1'])
        scores = bm25.get_scores(query)
        try:
            idx = all_sentences.index(item['sentence2'])
            sim = scores[idx] / (max(scores) + 1e-8)
        except:
            sim = 0.0
        predictions.append(1 if sim >= 0.3 else 0)
    elapsed = time.time() - start_time

    metrics = evaluate(predictions, labels)
    metrics['time'] = elapsed
    print(f"  准确率: {metrics['accuracy']:.4f}")
    print(f"  F1分数: {metrics['f1']:.4f}")
    print(f"  耗时: {elapsed:.4f}s")
    return metrics


# ==================== 方法4: 词向量 ====================
def run_word_vector(test_data):
    print("\n" + "=" * 60)
    print("方法4: 词向量 (Word Vector)")
    print("=" * 60)

    embed_dim = 100
    word_vectors = {}
    common_words = ['的', '了', '是', '我', '你', '他', '她', '这', '那', '有', '吗', '呢', '啊', '吧',
                    '什么', '怎么', '为什么', '可以', '已经', '一个', '没有', '不是', '就是', '在']
    for word in common_words:
        word_vectors[word] = np.random.randn(embed_dim)

    def get_embedding(text):
        words = tokenize_chinese(text)
        vecs = []
        for word in words:
            if word in word_vectors:
                vecs.append(word_vectors[word])
        if not vecs:
            return np.zeros(embed_dim)
        emb = np.mean(vecs, axis=0)
        return emb / (np.linalg.norm(emb) + 1e-8)

    labels = [item['label'] for item in test_data]
    start_time = time.time()
    predictions = []
    for item in tqdm(test_data, desc="  计算词向量"):
        v1 = get_embedding(item['sentence1'])
        v2 = get_embedding(item['sentence2'])
        sim = np.dot(v1, v2)
        predictions.append(1 if sim >= 0.5 else 0)
    elapsed = time.time() - start_time

    metrics = evaluate(predictions, labels)
    metrics['time'] = elapsed
    print(f"  准确率: {metrics['accuracy']:.4f}")
    print(f"  F1分数: {metrics['f1']:.4f}")
    print(f"  耗时: {elapsed:.4f}s")
    return metrics


# ==================== 方法5: GPU神经网络 ====================
def run_gpu_nn(train_data, val_data, test_data, config):
    """GPU优化的神经网络 - 修复所有问题"""
    print("\n" + "=" * 60)
    print("方法5: GPU神经网络 (GPU Neural Network)")
    print("=" * 60)

    # 检查GPU
    if torch.cuda.is_available():
        print(f"  ✅ 使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    else:
        print("  ⚠️  GPU不可用，使用CPU")

    # 构建词表
    print("  构建词表...")
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    all_texts = []
    for item in train_data + val_data + test_data:
        all_texts.append(item['sentence1'])
        all_texts.append(item['sentence2'])

    for text in tqdm(all_texts, desc="  构建词表"):
        for word in tokenize_chinese(text):
            if word not in word2idx:
                word2idx[word] = len(word2idx)

    vocab_size = len(word2idx)
    print(f"  词表大小: {vocab_size}")

    # 创建数据集
    print("  创建数据集...")
    train_dataset = TextMatchingDataset(train_data, word2idx, config.max_length)
    val_dataset = TextMatchingDataset(val_data, word2idx, config.max_length)
    test_dataset = TextMatchingDataset(test_data, word2idx, config.max_length)

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    # 创建模型
    print("  创建模型...")
    model = GPUTextMatcher(vocab_size, config.embed_dim, config.hidden_dim)
    model = model.to(config.device)

    # 多GPU支持
    if torch.cuda.device_count() > 1:
        print(f"  使用 {torch.cuda.device_count()} 个GPU")
        model = nn.DataParallel(model)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    scaler = GradScaler(enabled=config.use_amp and torch.cuda.is_available())

    # 训练
    print(f"  开始训练 (batch_size={config.batch_size}, epochs={config.num_epochs})...")
    start_time = time.time()
    best_val_acc = 0

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"  Epoch {epoch + 1}/{config.num_epochs}")

        for s1, s2, labels in progress_bar:
            s1 = s1.to(config.device, non_blocking=True)
            s2 = s2.to(config.device, non_blocking=True)
            labels = labels.to(config.device, non_blocking=True)

            optimizer.zero_grad()

            if config.use_amp and torch.cuda.is_available():
                with autocast():
                    logits = model(s1, s2)
                    loss = F.cross_entropy(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(s1, s2)
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(train_loader)
        scheduler.step()

        # 验证
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for s1, s2, labels in val_loader:
                s1 = s1.to(config.device, non_blocking=True)
                s2 = s2.to(config.device, non_blocking=True)
                logits = model(s1, s2)
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        print(f"  Epoch {epoch + 1} - Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('./results', exist_ok=True)
            torch.save(model.state_dict(), './results/best_model.pt')
            print(f"  ✅ 保存最佳模型 (Acc: {val_acc:.4f})")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    print(f"  训练完成，耗时: {elapsed:.2f}s")

    # 加载最佳模型
    if os.path.exists('./results/best_model.pt'):
        model.load_state_dict(torch.load('./results/best_model.pt'))

    # 测试
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for s1, s2, labels_batch in tqdm(test_loader, desc="  测试中"):
            s1 = s1.to(config.device, non_blocking=True)
            s2 = s2.to(config.device, non_blocking=True)
            logits = model(s1, s2)
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            labels.extend(labels_batch.numpy())

    metrics = evaluate(predictions, labels)
    metrics['time'] = elapsed
    print(f"  测试准确率: {metrics['accuracy']:.4f}")
    print(f"  测试F1分数: {metrics['f1']:.4f}")
    print(f"  总耗时: {elapsed:.2f}s")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


# ==================== 结果可视化 ====================
def plot_results(results):
    methods = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in methods]
    f1_scores = [results[m]['f1'] for m in methods]
    times = [results[m]['time'] for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(methods))
    width = 0.35
    bars1 = axes[0].bar(x - width / 2, accuracies, width, label='Accuracy', color='steelblue')
    bars2 = axes[0].bar(x + width / 2, f1_scores, width, label='F1 Score', color='coral')
    axes[0].set_xlabel('Methods')
    axes[0].set_ylabel('Scores')
    axes[0].set_title('Accuracy & F1 Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=45, ha='right')
    axes[0].legend()
    axes[0].set_ylim(0, 1)

    for bar in bars1:
        height = bar.get_height()
        axes[0].annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        axes[0].annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    axes[1].bar(methods, times, color='lightgreen')
    axes[1].set_xlabel('Methods')
    axes[1].set_ylabel('Time (seconds)')
    axes[1].set_title('Runtime Comparison')
    axes[1].tick_params(axis='x', rotation=45)

    for i, v in enumerate(times):
        axes[1].annotate(f'{v:.1f}s', xy=(i, v), xytext=(0, 3),
                         textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    os.makedirs('./results', exist_ok=True)
    plt.savefig('./results/comparison_plot.png', dpi=300, bbox_inches='tight')
    plt.show()


def save_results(results):
    os.makedirs('./results', exist_ok=True)

    with open('./results/results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    df_data = []
    for method, metrics in results.items():
        row = {'method': method}
        row.update(metrics)
        df_data.append(row)
    df = pd.DataFrame(df_data)
    df.to_csv('./results/results.csv', index=False, encoding='utf-8')

    print("\n📊 结果汇总:")
    print(df.to_string(index=False))


# ==================== 主函数 ====================
def main():
    print("\n" + "=" * 60)
    print("  文本匹配方法对比实验 - GPU优化版本")
    print("  包含: 编辑距离 | TF-IDF | BM25 | 词向量 | GPU神经网络")
    print("=" * 60)

    # GPU信息
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    else:
        print("  ⚠️  未检测到GPU，使用CPU")
    print(f"  设备: {config.device}")
    print(f"  Batch Size: {config.batch_size}")
    print("=" * 60)

    set_seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs('./results', exist_ok=True)

    print("\n📂 加载数据...")
    train_data = load_data(f"{config.data_dir}/{config.train_file}")
    val_data = load_data(f"{config.data_dir}/{config.val_file}")
    test_data = load_data(f"{config.data_dir}/{config.test_file}")

    print(f"  训练集: {len(train_data)} 条")
    print(f"  验证集: {len(val_data)} 条")
    print(f"  测试集: {len(test_data)} 条")

    if len(train_data) == 0:
        print("  ❌ 错误: 训练数据为空，请检查数据格式")
        return

    results = {}

    # 1. 编辑距离 (CPU)
    results['Edit Distance'] = run_edit_distance(test_data)

    # 2. TF-IDF (CPU)
    results['TF-IDF'] = run_tfidf(test_data)

    # 3. BM25 (CPU)
    results['BM25'] = run_bm25(test_data)

    # 4. 词向量 (CPU)
    results['Word Vector'] = run_word_vector(test_data)

    # 5. GPU神经网络
    results['GPU Neural Network'] = run_gpu_nn(train_data, val_data, test_data, config)

    print("\n" + "=" * 60)
    print("📊 实验结果汇总")
    print("=" * 60)
    save_results(results)
    plot_results(results)

    print("\n" + "=" * 60)
    print("✅ 实验完成！")
    print("  结果保存在 ./results/ 目录")
    print("=" * 60)


if __name__ == "__main__":
    main()