"""文档问答答案生成模型实现。

支持：
- Qwen2-VL
- Qwen2.5-VL
- Qwen3-VL
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

from .base import BaseGenerator


DEFAULT_SYSTEM_PROMPT = """你是一个中文文档问答助手。
请根据提供的文档图片回答问题。

要求：
1. 仅依据图片中的可见内容回答
2. 答案必须简洁，不要解释
3. 如果图片中没有答案，输出"无法确定"
4. 只输出最终答案，不要复述问题
"""


@dataclass
class GeneratorConfig:
    """Generator 配置。"""

    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    device: str | None = None
    max_new_tokens: int = 128
    temperature: float = 0.1
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = False
    min_pixels: int | None = None
    max_pixels: int | None = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    trust_remote_code: bool = True


class QwenVLGenerator(BaseGenerator):
    """Qwen-VL 系列生成模型。

    自动检测并支持：
    - Qwen2-VL
    - Qwen2.5-VL
    - Qwen3-VL
    """

    MODEL_CLASS_MAP = {
        "qwen2-vl": Qwen2VLForConditionalGeneration,
        "qwen2.5-vl": Qwen2_5_VLForConditionalGeneration,
        "qwen3-vl": Qwen3VLForConditionalGeneration,
    }

    def __init__(
        self,
        config: GeneratorConfig | None = None,
        normalizer: Any | None = None,
    ) -> None:
        """初始化 Generator。

        Args:
            config: 生成配置
            normalizer: 答案规范化器（可选）
        """
        self.config = config or GeneratorConfig()
        self.normalizer = normalizer

        self.device = self.config.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model, self.processor = self._load_model()
        self.model.eval()

    def _detect_model_type(self, model_name: str) -> str:
        """检测 Qwen-VL 模型类型。"""
        name_lower = model_name.lower()
        if "qwen3" in name_lower:
            return "qwen3-vl"
        elif "qwen2.5" in name_lower or "qwen2_5" in name_lower:
            return "qwen2.5-vl"
        elif "qwen2" in name_lower:
            return "qwen2-vl"
        return "qwen2.5-vl"

    def _load_model(self) -> tuple[Any, Any]:
        """加载模型和 processor。"""
        model_type = self._detect_model_type(self.config.model_name)
        model_class = self.MODEL_CLASS_MAP.get(model_type)

        if model_class is None:
            raise ValueError(f"Unsupported model type: {model_type}")

        processor = AutoProcessor.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )

        model = model_class.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map=self.device if self.device == "auto" else None,
        )

        if self.device != "auto":
            model = model.to(self.device)

        return model, processor

    def _build_messages(
        self,
        question: str,
        images: Sequence[Image.Image],
    ) -> list[dict[str, Any]]:
        """构建对话消息。"""
        system = self.config.system_prompt or DEFAULT_SYSTEM_PROMPT

        content = []
        for idx, img in enumerate(images, 1):
            content.append({"type": "text", "text": f"[文档{idx}]"})
            content.append({"type": "image", "image": img})

        content.append({"type": "text", "text": f"\n问题：{question}\n\n请直接输出答案："})

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]

        return messages

    @torch.no_grad()
    def generate(
        self,
        question: str,
        image_paths: Sequence[str | Path],
    ) -> dict[str, Any]:
        """生成答案。"""
        if not image_paths:
            return {
                "answer": "无法确定",
                "raw_answer": "",
                "is_uncertain": True,
                "num_images": 0,
            }

        images = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"[warning] Failed to load image {path}: {e}")
                continue

        if not images:
            return {
                "answer": "无法确定",
                "raw_answer": "",
                "is_uncertain": True,
                "num_images": 0,
            }

        messages = self._build_messages(question, images)

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=images,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        generation_config = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature if self.config.do_sample else None,
            "top_p": self.config.top_p if self.config.do_sample else None,
            "top_k": self.config.top_k if self.config.do_sample else None,
            "do_sample": self.config.do_sample,
        }
        generation_config = {k: v for k, v in generation_config.items() if v is not None}

        outputs = self.model.generate(**inputs, **generation_config)

        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        raw_answer = self.processor.decode(
            generated_ids, skip_special_tokens=True
        ).strip()

        # 规范化答案
        if self.normalizer:
            cleaned_answer = self.normalizer.clean_extracted_answer(raw_answer)
            is_uncertain = self.normalizer.is_uncertain_answer(cleaned_answer)
        else:
            cleaned_answer = raw_answer
            is_uncertain = False

        for img in images:
            img.close()

        return {
            "answer": cleaned_answer,
            "raw_answer": raw_answer,
            "is_uncertain": is_uncertain,
            "num_images": len(images),
        }

    def batch_generate(
        self,
        questions: Sequence[str],
        image_paths_list: Sequence[Sequence[str | Path]],
    ) -> list[dict[str, Any]]:
        """批量生成答案。"""
        results = []
        for question, image_paths in zip(questions, image_paths_list):
            result = self.generate(question, image_paths)
            results.append(result)
        return results
