"""多模态向量编码模型实现。

支持：
- CLIP 系列（AltCLIP 等）
- SigLIP 系列
- Qwen3-VL-Embedding
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoModel, AutoProcessor

from .base import BaseEmbeddingModel, EmbeddingOutput


MODEL_PRESETS = {
    # Qwen 系列
    "qwen3-vl-embedding-2b": "Qwen/Qwen3-VL-Embedding-2B",
    "qwen3-vl-embedding-8b": "Qwen/Qwen3-VL-Embedding-8B",

    # Chinese-CLIP 系列
    "chinese-clip-base": "OFA-Sys/chinese-clip-vit-base-patch16",
    "chinese-clip-large": "OFA-Sys/chinese-clip-vit-large-patch14",
    "chinese-clip-huge": "OFA-Sys/chinese-clip-vit-huge-patch14",

    # AltCLIP
    "altclip": "BAAI/AltCLIP",

    # Taiyi-CLIP
    "taiyi-clip-base": "IDEA-CCNL/Taiyi-CLIP-Roberta-102M-Chinese",
    "taiyi-clip-large": "IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese",

    # M-CLIP
    "mclip-xlm": "M-CLIP/XLM-Roberta-Large-Vit-B-16Plus",

    # Multilingual CLIP
    "clip-multilingual": "sentence-transformers/clip-ViT-B-32-multilingual-v1",

    # SigLIP
    "siglip-base": "google/siglip-base-patch16-256",
    "siglip2-base": "google/siglip2-base-patch16-256",
    "siglip-so400m": "google/siglip-so400m-patch14-384",
}


def resolve_model_name(model_name: str) -> str:
    """解析模型名称，支持预设别名。"""
    return MODEL_PRESETS.get(model_name, model_name)


def is_qwen3_vl_embedding_model(resolved_id: str) -> bool:
    """判断是否为 Qwen3-VL-Embedding 模型。"""
    return "qwen3-vl-embedding" in resolved_id.lower()


def is_qwen3_vl_embedding_checkpoint_dir(load_path: str | Path) -> bool:
    """根据本地 config.json 判断是否为 Qwen3VLForEmbedding 检查点。"""
    root = Path(load_path)
    if not root.is_dir():
        return False
    cfg_path = root / "config.json"
    if not cfg_path.is_file():
        return False
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    for name in cfg.get("architectures") or []:
        if isinstance(name, str) and "Qwen3VLForEmbedding" in name:
            return True
    return False


def _l2_normalize(array: np.ndarray) -> np.ndarray:
    """L2 归一化。"""
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return array / norms


class MultimodalEmbeddingModel(BaseEmbeddingModel):
    """多模态向量编码模型。

    自动检测模型类型：
    - CLIP/SigLIP: 使用 AutoModel + AutoProcessor
    - Qwen3-VL-Embedding: 使用专用的 Qwen3VLEmbedder
    """

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        trust_remote_code: bool = True,
        qwen_query_instruction: str | None = None,
    ) -> None:
        self.model_name = resolve_model_name(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._backend: str
        self._qwen: Any = None
        self._qwen_query_instruction = qwen_query_instruction

        # 检测是否为 Qwen3-VL-Embedding
        if is_qwen3_vl_embedding_model(self.model_name) or is_qwen3_vl_embedding_checkpoint_dir(
            self.model_name
        ):
            try:
                from .qwen3_vl_embedding import Qwen3VLEmbedder
            except ImportError as exc:
                raise ImportError(
                    "使用 Qwen3-VL-Embedding 需安装: pip install 'transformers>=4.57.0' qwen-vl-utils"
                ) from exc
            self._backend = "qwen3_vl"
            if self._qwen_query_instruction is None:
                self._qwen_query_instruction = (
                    "Retrieve images or text relevant to the user's query."
                )
            self._qwen = Qwen3VLEmbedder(
                model_name_or_path=self.model_name,
                device=str(self.device),
            )
            self.model = self._qwen.model
            self.processor = self._qwen.processor
            self.model.eval()
            return

        # CLIP/SigLIP 路径
        self._backend = "clip"
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=trust_remote_code,
        )
        self.model.to(self.device)
        self.model.eval()

    def _extract_feature_tensor(self, output: torch.Tensor | object) -> torch.Tensor:
        """从模型输出中提取特征张量。"""
        if isinstance(output, torch.Tensor):
            return output

        for attr in ("image_embeds", "text_embeds", "pooler_output"):
            value = getattr(output, attr, None)
            if isinstance(value, torch.Tensor):
                return value

        last_hidden_state = getattr(output, "last_hidden_state", None)
        if isinstance(last_hidden_state, torch.Tensor):
            return last_hidden_state[:, 0]

        raise TypeError(
            f"不支持的模型输出类型: {type(output).__name__}. "
            "请检查当前模型是否返回 image/text embedding。"
        )

    @torch.no_grad()
    def encode_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        max_length: int = 64,
    ) -> EmbeddingOutput:
        """编码文本为向量。"""
        if self._backend == "qwen3_vl":
            chunks: list[np.ndarray] = []
            text_list = list(texts)
            for start in range(0, len(text_list), batch_size):
                batch = text_list[start : start + batch_size]
                inputs: list[dict[str, Any]] = [
                    {
                        "text": t,
                        "instruction": self._qwen_query_instruction,
                    }
                    for t in batch
                ]
                embs = self._qwen.process(inputs)
                chunks.append(embs.detach().float().cpu().numpy())
            return EmbeddingOutput(
                embeddings=np.concatenate(chunks, axis=0),
                model_name=self.model_name,
            )

        # CLIP/SigLIP 路径
        batches: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            inputs = self.processor(
                text=batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            features = self.model.get_text_features(**inputs)
            features = self._extract_feature_tensor(features)
            features = features.detach().float().cpu().numpy()
            batches.append(_l2_normalize(features))

        return EmbeddingOutput(
            embeddings=np.concatenate(batches, axis=0),
            model_name=self.model_name,
        )

    @torch.no_grad()
    def encode_images(
        self,
        image_paths: Sequence[str],
        batch_size: int = 16,
        skip_bad_images: bool = False,
    ) -> EmbeddingOutput:
        """编码图片为向量。"""
        if self._backend == "qwen3_vl":
            batches: list[np.ndarray] = []
            valid_paths: list[str] = []
            paths = list(image_paths)
            for start in range(0, len(paths), batch_size):
                batch_paths = paths[start : start + batch_size]
                specs: list[dict[str, Any]] = []
                opened_paths: list[str] = []
                for path in batch_paths:
                    try:
                        with Image.open(path) as im:
                            im.load()
                    except (OSError, UnidentifiedImageError, ValueError) as exc:
                        if skip_bad_images:
                            print(
                                f"[skip] 无法读取图片: {path} ({type(exc).__name__}: {exc})"
                            )
                            continue
                        raise
                    specs.append({"image": str(Path(path).resolve())})
                    opened_paths.append(path)
                if not specs:
                    continue
                embs = self._qwen.process(specs)
                batches.append(embs.detach().float().cpu().numpy())
                valid_paths.extend(opened_paths)

            if not batches:
                if skip_bad_images:
                    dim = self.embedding_dim()
                    return EmbeddingOutput(
                        embeddings=np.zeros((0, dim), dtype=np.float32),
                        model_name=self.model_name,
                        image_paths=[],
                    )
                raise ValueError("没有成功编码任何图片，请检查图片路径或开启 skip_bad_images")

            return EmbeddingOutput(
                embeddings=np.concatenate(batches, axis=0),
                model_name=self.model_name,
                image_paths=valid_paths,
            )

        # CLIP/SigLIP 路径
        batches = []
        valid_paths: list[str] = []
        for start in range(0, len(image_paths), batch_size):
            batch_paths = list(image_paths[start : start + batch_size])
            images: list[Image.Image] = []
            opened_paths: list[str] = []
            for path in batch_paths:
                try:
                    image = Image.open(path)
                    image.load()
                    images.append(image.convert("RGB"))
                    opened_paths.append(path)
                except (OSError, UnidentifiedImageError, ValueError) as exc:
                    if skip_bad_images:
                        print(f"[skip] 无法读取图片: {path} ({type(exc).__name__}: {exc})")
                        continue
                    raise
            if not images:
                continue
            try:
                inputs = self.processor(images=images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                features = self.model.get_image_features(**inputs)
                features = self._extract_feature_tensor(features)
                features = features.detach().float().cpu().numpy()
                batches.append(_l2_normalize(features))
                valid_paths.extend(opened_paths)
            finally:
                for image in images:
                    image.close()

        if not batches:
            if skip_bad_images:
                dim = self.embedding_dim()
                return EmbeddingOutput(
                    embeddings=np.zeros((0, dim), dtype=np.float32),
                    model_name=self.model_name,
                    image_paths=[],
                )
            raise ValueError("没有成功编码任何图片，请检查图片路径或开启 skip_bad_images")

        return EmbeddingOutput(
            embeddings=np.concatenate(batches, axis=0),
            model_name=self.model_name,
            image_paths=valid_paths,
        )

    def embedding_dim(self, sample_text: str = "测试") -> int:
        """返回向量维度。"""
        return int(self.encode_texts([sample_text]).embeddings.shape[1])
