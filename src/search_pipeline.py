"""搜索管道 - 两阶段检索（Vector + Rerank）"""

import logging
from typing import List, Optional, Dict, Any

from vector_store import QdrantStore
from embedder import get_query_embedding, configure_embedder
from reranker import rerank
from embed_config import COLLECTION_NAME

logger = logging.getLogger(__name__)

class SearchPipeline:
    """两阶段搜索管道"""
    
    def __init__(self):
        self.store = QdrantStore()
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        initial_k: int = 50,
        section_types: Optional[List[str]] = None,
        use_rerank: bool = True
    ) -> List[Dict[str, Any]]:
        """
        执行两阶段搜索
        
        Args:
            query: 查询文本
            top_k: 最终返回数量
            initial_k: 第一阶段召回数量
            section_types: 过滤的 section 类型
            use_rerank: 是否使用 reranker
        
        Returns:
            搜索结果列表
        """
        # 确保 embedder 已配置
        configure_embedder()
        
        # Stage 1: Vector Search
        logger.info(f"Stage 1: Vector search for top {initial_k}")
        query_vector = get_query_embedding(query)
        
        filter_dict = None
        if section_types:
            filter_dict = {"section_type": section_types}
        
        initial_results = self.store.search(
            query_vector=query_vector,
            limit=initial_k,
            filter=filter_dict
        )
        
        if not initial_results:
            return []
        
        # 准备 chunks（需要获取文本内容）
        chunks = []
        for r in initial_results:
            # 读取原始文本
            file_path = r["payload"].get("file_path", "")
            text = self._load_chunk_text(file_path, r["payload"])
            
            chunks.append({
                "id": r["id"],
                "text": text,
                "payload": r["payload"],
                "vector_score": r["score"]
            })
        
        if not use_rerank:
            return chunks[:top_k]
        
        # Stage 2: Rerank
        logger.info(f"Stage 2: Reranking {len(chunks)} chunks")
        reranked = rerank(query, chunks, top_k=top_k)
        
        return reranked
    
    def _load_chunk_text(self, file_path: str, payload: dict) -> str:
        """加载 chunk 的原始文本"""
        from pathlib import Path
        from embed_config import DATA_DIR
        import json
        
        paper_name = payload.get('paper_name', '')
        section_type = payload.get('section_type', '')
        chunk_id = payload.get('chunk_id', 0)
        
        # 尝试从 sections 目录读取
        sections_dir = DATA_DIR.parent / "data" / "sections" / paper_name
        if sections_dir.exists():
            section_file = sections_dir / f"{section_type}.txt"
            if section_file.exists():
                try:
                    content = section_file.read_text(encoding="utf-8")
                    return content[:4000]  # 增加到 4000 字符
                except:
                    pass
        
        # 备用：从 chunks 文件读取
        chunks_dir = DATA_DIR.parent / "data" / "chunks" / paper_name
        if chunks_dir.exists():
            chunk_file = chunks_dir / f"{section_type}_{chunk_id}.txt"
            if chunk_file.exists():
                try:
                    return chunk_file.read_text(encoding="utf-8")
                except:
                    pass
        
        # 最后备用：返回元数据描述
        return f"[Paper: {paper_name} | Section: {section_type} | Chunk: {chunk_id}]"
