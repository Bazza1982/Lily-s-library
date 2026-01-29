"""Cross-Encoder Reranker 模块 - 使用 bge-reranker-large"""

import logging
from typing import List, Tuple, Optional
import torch

logger = logging.getLogger(__name__)

# 全局变量
_reranker = None

def load_reranker(model_name: str = "BAAI/bge-reranker-large", use_fp16: bool = True):
    """加载 reranker 模型"""
    global _reranker
    if _reranker is not None:
        return _reranker
    
    from FlagEmbedding import FlagReranker
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading reranker on {device}...")
    
    _reranker = FlagReranker(model_name, use_fp16=use_fp16, device=device)
    logger.info("Reranker loaded successfully")
    return _reranker

def rerank(
    query: str,
    chunks: List[dict],  # [{"text": ..., "payload": ..., "score": ...}, ...]
    top_k: int = 10,
    batch_size: int = 32
) -> List[dict]:
    """
    对 chunks 进行重排序
    
    Args:
        query: 查询文本
        chunks: 包含 text, payload, score 的 chunk 列表
        top_k: 返回前 k 个结果
        batch_size: 批处理大小
    
    Returns:
        重排序后的 chunks（包含 rerank_score）
    """
    if not chunks:
        return []
    
    reranker = load_reranker()
    
    # 准备 pairs
    pairs = [[query, chunk["text"]] for chunk in chunks]
    
    # 计算分数
    scores = reranker.compute_score(pairs, batch_size=batch_size)
    
    # 如果只有一个 pair，scores 可能是单个数字
    if not isinstance(scores, list):
        scores = [scores]
    
    # 添加 rerank 分数
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)
    
    # 按 rerank_score 降序排序
    sorted_chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    
    return sorted_chunks[:top_k]
