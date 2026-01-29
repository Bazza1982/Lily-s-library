"""本地向量化模块 - 使用 BAAI/bge-m3 模型

与 Gemini embedder 不同，此模块使用本地部署的 BGE-M3 模型进行向量化。
支持 GPU 加速 (CUDA) 和批量处理。

向量维度: 1024
最大序列长度: 8192 tokens
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

# 全局变量：缓存模型
_embedder = None

# BGE-M3 模型参数
BGE_M3_MODEL = "BAAI/bge-m3"
BGE_M3_DIMENSION = 1024
BGE_M3_MAX_LENGTH = 8192


class BGEEmbedderError(Exception):
    """BGE Embedder base exception."""
    pass


class ModelLoadError(BGEEmbedderError):
    """Model loading error."""
    pass


class EmbeddingError(BGEEmbedderError):
    """Embedding computation error."""
    pass


def load_embedder(
    model_name: str = BGE_M3_MODEL,
    use_fp16: bool = True,
) -> Any:
    """
    加载 BGE-M3 embedding 模型（带缓存）

    Args:
        model_name: 模型名称（默认 BAAI/bge-m3）
        use_fp16: 是否使用 FP16（GPU 时有效，可减少显存占用）

    Returns:
        BGEM3FlagModel 实例
    """
    global _embedder
    if _embedder is not None:
        return _embedder

    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError:
        raise ModelLoadError(
            "FlagEmbedding not installed. Run: pip install FlagEmbedding"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading BGE-M3 model on {device}...")

    try:
        _embedder = BGEM3FlagModel(
            model_name,
            use_fp16=use_fp16 and device == "cuda",
            device=device
        )
        logger.info(f"BGE-M3 loaded successfully (dimension={BGE_M3_DIMENSION})")
        return _embedder
    except Exception as e:
        raise ModelLoadError(f"Failed to load BGE-M3 model: {e}")


def get_device() -> str:
    """获取当前运行设备"""
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_dimension() -> int:
    """获取 BGE-M3 向量维度"""
    return BGE_M3_DIMENSION


def embed_texts(
    texts: List[str],
    batch_size: int = 32,
    max_length: int = BGE_M3_MAX_LENGTH,
    show_progress: bool = False
) -> List[List[float]]:
    """
    批量向量化文本

    Args:
        texts: 文本列表
        batch_size: 批处理大小
        max_length: 最大序列长度
        show_progress: 是否显示进度条

    Returns:
        向量列表（每个向量是 1024 维 float 列表）
    """
    if not texts:
        return []

    embedder = load_embedder()

    all_embeddings = []

    # 分批处理
    num_batches = (len(texts) + batch_size - 1) // batch_size
    iterator = range(0, len(texts), batch_size)

    if show_progress:
        iterator = tqdm(
            iterator,
            desc="Embedding",
            unit="batch",
            total=num_batches
        )

    for i in iterator:
        batch = texts[i:i + batch_size]
        try:
            # BGE-M3 encode 返回包含 dense_vecs 的字典
            result = embedder.encode(
                batch,
                batch_size=len(batch),
                max_length=max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False
            )

            # result['dense_vecs'] 是 numpy array，转为 list
            batch_embeddings = result['dense_vecs'].tolist()
            all_embeddings.extend(batch_embeddings)

        except Exception as e:
            logger.error(f"Error embedding batch at index {i}: {e}")
            raise EmbeddingError(f"Embedding failed at batch {i}: {e}")

    return all_embeddings


def embed_single(text: str, max_length: int = BGE_M3_MAX_LENGTH) -> List[float]:
    """
    向量化单个文本

    Args:
        text: 输入文本
        max_length: 最大序列长度

    Returns:
        向量（1024 维 float 列表）
    """
    embeddings = embed_texts([text], batch_size=1, max_length=max_length)
    return embeddings[0]


def embed_query(query: str, max_length: int = BGE_M3_MAX_LENGTH) -> List[float]:
    """
    向量化查询文本（用于搜索）

    注意: BGE-M3 不区分 document 和 query embedding，
    但保留此接口以保持与其他 embedder 的一致性。

    Args:
        query: 查询文本
        max_length: 最大序列长度

    Returns:
        向量（1024 维 float 列表）
    """
    return embed_single(query, max_length=max_length)


def process_paper_chunks(
    paper_dir: Path,
    batch_size: int = 32,
    max_length: int = BGE_M3_MAX_LENGTH
) -> Dict[str, Any]:
    """
    处理单个论文的所有 chunks 并生成向量

    Args:
        paper_dir: 论文 chunk 目录（如 data/chunks/Paper_Name/）
        batch_size: 批处理大小
        max_length: 最大序列长度

    Returns:
        {
            'paper_name': str,
            'chunks': [
                {'section': str, 'text': str, 'vector': list[float]},
                ...
            ]
        }
    """
    paper_name = paper_dir.name

    # 读取所有 txt 文件
    txt_files = sorted(paper_dir.glob("*.txt"))
    if not txt_files:
        logger.warning(f"No chunks found in {paper_dir}")
        return {'paper_name': paper_name, 'chunks': []}

    # 收集所有文本
    sections = []
    texts = []
    for txt_file in txt_files:
        section = txt_file.stem  # abstract, introduction, etc.
        try:
            text = txt_file.read_text(encoding='utf-8').strip()
            if text:
                sections.append(section)
                texts.append(text)
        except Exception as e:
            logger.warning(f"Error reading {txt_file}: {e}")

    if not texts:
        logger.warning(f"No valid text content in {paper_dir}")
        return {'paper_name': paper_name, 'chunks': []}

    # 批量向量化
    try:
        vectors = embed_texts(texts, batch_size=batch_size, max_length=max_length)
    except EmbeddingError as e:
        logger.error(f"Failed to embed paper {paper_name}: {e}")
        raise

    # 组装结果
    chunks = []
    for section, text, vector in zip(sections, texts, vectors):
        chunks.append({
            'section': section,
            'text': text,
            'vector': vector
        })

    return {
        'paper_name': paper_name,
        'chunks': chunks
    }


def save_vectors(
    result: Dict[str, Any],
    output_dir: str,
    include_text: bool = True,
    indent: Optional[int] = None
) -> Path:
    """
    保存向量到 JSON 文件

    Args:
        result: process_paper_chunks 的返回结果
        output_dir: 输出目录
        include_text: 是否包含原文（False 可减小文件体积）
        indent: JSON 缩进（None 表示紧凑格式）

    Returns:
        输出文件路径
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    paper_name = result['paper_name']
    output_file = output_path / f"{paper_name}.json"

    # 如果不包含原文，移除 text 字段
    if not include_text:
        result = {
            'paper_name': result['paper_name'],
            'chunks': [
                {'section': c['section'], 'vector': c['vector']}
                for c in result['chunks']
            ]
        }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=indent)

    return output_file


def load_vectors(vector_file: Path) -> Dict[str, Any]:
    """
    加载向量文件

    Args:
        vector_file: 向量 JSON 文件路径

    Returns:
        {
            'paper_name': str,
            'chunks': [
                {'section': str, 'text': str, 'vector': list[float]},
                ...
            ]
        }
    """
    with open(vector_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_all_vectors(vectors_dir: str) -> List[Dict[str, Any]]:
    """
    加载所有向量文件

    Args:
        vectors_dir: 向量目录

    Returns:
        向量数据列表
    """
    vectors_path = Path(vectors_dir)
    all_data = []

    for json_file in sorted(vectors_path.glob("*.json")):
        try:
            data = load_vectors(json_file)
            all_data.append(data)
        except Exception as e:
            logger.warning(f"Error loading {json_file}: {e}")

    return all_data
