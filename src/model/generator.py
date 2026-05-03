"""Document VQA answer generation models.

Supported generators:
- QwenVLGenerator: Qwen2-VL, Qwen2.5-VL, Qwen3-VL
- LLaVAGenerator: LLaVA-OneVision
- InternVLGenerator: InternVL2.5

Use GeneratorFactory.create(config) to auto-detect and instantiate the correct generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
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
    """Generator configuration."""

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
    # internvl_max_tiles: int = 12


# ---------------------------------------------------------------------------
# Qwen-VL Generator
# ---------------------------------------------------------------------------


class QwenVLGenerator(BaseGenerator):
    """Qwen-VL series generator (Qwen2-VL, Qwen2.5-VL, Qwen3-VL)."""

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
        self.config = config or GeneratorConfig()
        self.normalizer = normalizer

        self.device = self.config.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model, self.processor = self._load_model()
        self.model.eval()

    def _detect_model_type(self, model_name: str) -> str:
        name_lower = model_name.lower()
        if "qwen3" in name_lower:
            return "qwen3-vl"
        elif "qwen2.5" in name_lower or "qwen2_5" in name_lower:
            return "qwen2.5-vl"
        elif "qwen2" in name_lower:
            return "qwen2-vl"
        return "qwen2.5-vl"

    def _load_model(self) -> tuple[Any, Any]:
        model_type = self._detect_model_type(self.config.model_name)
        model_class = self.MODEL_CLASS_MAP.get(model_type)

        if model_class is None:
            raise ValueError(f"Unsupported Qwen-VL type: {model_type}")

        processor = AutoProcessor.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
            min_pixels=self.config.min_pixels,
            max_pixels=self.config.max_pixels,
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
        system = self.config.system_prompt or DEFAULT_SYSTEM_PROMPT

        content = []
        for idx, img in enumerate(images, 1):
            content.append({"type": "text", "text": f"[文档{idx}]"})
            content.append({"type": "image", "image": img})

        content.append({"type": "text", "text": f"\n问题：{question}\n\n请直接输出答案："})

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]

    @torch.no_grad()
    def generate(
        self,
        question: str,
        image_paths: Sequence[str | Path],
    ) -> dict[str, Any]:
        if not image_paths:
            return _uncertain_result()

        images = _load_images(image_paths)
        if not images:
            return _uncertain_result()

        messages = self._build_messages(question, images)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text], images=images, return_tensors="pt", padding=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        raw_answer = self._run_generation(inputs)

        for img in images:
            img.close()

        return self._normalize_result(raw_answer, len(images))

    @torch.no_grad()
    def batch_generate(
        self,
        questions: Sequence[str],
        image_paths_list: Sequence[Sequence[str | Path]],
    ) -> list[dict[str, Any]]:
        results = []
        for question, image_paths in zip(questions, image_paths_list):
            results.append(self.generate(question, image_paths))
        return results

    def _run_generation(self, inputs: dict[str, Any]) -> str:
        gen_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
        }
        if self.config.do_sample:
            gen_kwargs.update({
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
            })

        outputs = self.model.generate(**inputs, **gen_kwargs)
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(generated_ids, skip_special_tokens=True).strip()

    def _normalize_result(
        self, raw_answer: str, num_images: int,
    ) -> dict[str, Any]:
        if self.normalizer:
            cleaned = self.normalizer.clean_extracted_answer(raw_answer)
            is_uncertain = self.normalizer.is_uncertain_answer(cleaned)
        else:
            cleaned = raw_answer
            is_uncertain = False

        return {
            "answer": cleaned,
            "raw_answer": raw_answer,
            "is_uncertain": is_uncertain,
            "num_images": num_images,
        }


# ---------------------------------------------------------------------------
# LLaVA Generator
# ---------------------------------------------------------------------------


class LLaVAGenerator(BaseGenerator):
    """LLaVA-OneVision generator.

    Supports llava-onevision (llava-onevision-qwen2, etc.) via the transformers
    native ``LlavaOnevisionForConditionalGeneration`` class.
    """

    def __init__(
        self,
        config: GeneratorConfig | None = None,
        normalizer: Any | None = None,
    ) -> None:
        self.config = config or GeneratorConfig()
        self.normalizer = normalizer

        self.device = self.config.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model, self.processor = self._load_model()
        self.model.eval()

    def _load_model(self) -> tuple[Any, Any]:
        from transformers import LlavaOnevisionForConditionalGeneration

        processor = AutoProcessor.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )

        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
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
        system = self.config.system_prompt or DEFAULT_SYSTEM_PROMPT

        content: list[dict[str, Any]] = []
        for idx in range(len(images)):
            content.append({"type": "text", "text": f"[文档{idx + 1}]"})
            content.append({"type": "image"})

        content.append({"type": "text", "text": f"\n问题：{question}\n\n请直接输出答案："})

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]

    @torch.no_grad()
    def generate(
        self,
        question: str,
        image_paths: Sequence[str | Path],
    ) -> dict[str, Any]:
        if not image_paths:
            return _uncertain_result()

        images = _load_images(image_paths)
        if not images:
            return _uncertain_result()

        messages = self._build_messages(question, images)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        inputs = self.processor(
            text=text, images=images, return_tensors="pt", padding=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        raw_answer = self._run_generation(inputs)

        for img in images:
            img.close()

        return self._normalize_result(raw_answer, len(images))

    @torch.no_grad()
    def batch_generate(
        self,
        questions: Sequence[str],
        image_paths_list: Sequence[Sequence[str | Path]],
    ) -> list[dict[str, Any]]:
        return [self.generate(q, p) for q, p in zip(questions, image_paths_list)]

    def _run_generation(self, inputs: dict[str, Any]) -> str:
        gen_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
        }
        if self.config.do_sample:
            gen_kwargs.update({
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
            })

        outputs = self.model.generate(**inputs, **gen_kwargs)
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(generated_ids, skip_special_tokens=True).strip()

    def _normalize_result(
        self, raw_answer: str, num_images: int,
    ) -> dict[str, Any]:
        if self.normalizer:
            cleaned = self.normalizer.clean_extracted_answer(raw_answer)
            is_uncertain = self.normalizer.is_uncertain_answer(cleaned)
        else:
            cleaned = raw_answer
            is_uncertain = False

        return {
            "answer": cleaned,
            "raw_answer": raw_answer,
            "is_uncertain": is_uncertain,
            "num_images": num_images,
        }


# ---------------------------------------------------------------------------
# InternVL Generator  (official model.chat() API)
# ---------------------------------------------------------------------------


# ImageNet stats used by InternVL's official preprocessing
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Max number of 448×448 tiles per image (official default)
_MAX_TILES = 8  # lowered for multi-image DocVQA — fewer tiles = less VRAM per sample


def _find_closest_aspect_ratio(
    aspect_ratio: float, target_ratios: list[tuple[int, int]],
    width: int, height: int, image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        ratio_diff = abs(aspect_ratio - ratio[0] / ratio[1])
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = _MAX_TILES,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> list[Image.Image]:
    """Official InternVL dynamic-resolution tiling."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = list(
        dict.fromkeys(  # preserve order, deduplicate
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
    )

    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size,
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]

    resized = image.resize((target_width, target_height))
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    processed: list[Image.Image] = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed.append(resized.crop(box))

    if use_thumbnail and len(processed) != 1:
        processed.append(image.resize((image_size, image_size)))

    return processed


def _build_internvl_transform(input_size: int = 448):
    """Build the official InternVL image transform (Resize → ToTensor → Normalize)."""
    from torchvision.transforms import Compose, Normalize, Resize, ToTensor
    from torchvision.transforms.functional import InterpolationMode

    return Compose([
        ToTensor(),  # handles RGB conversion
        Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def _load_image_internvl(image: Image.Image | str | Path, max_num: int = _MAX_TILES) -> torch.Tensor:
    """Load and preprocess a single image for InternVL.

    Returns a tensor of shape ``(num_tiles, 3, 448, 448)``.
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    transform = _build_internvl_transform()
    tiles = _dynamic_preprocess(image, max_num=max_num)
    return torch.stack([transform(t) for t in tiles])


class InternVLGenerator(BaseGenerator):
    """InternVL2.5 generator using the official ``model.chat()`` API.

    Supports multiple document images via ``num_patches_list``.
    """

    def __init__(
        self,
        config: GeneratorConfig | None = None,
        normalizer: Any | None = None,
    ) -> None:
        self.config = config or GeneratorConfig()
        self.normalizer = normalizer

        self.device = self.config.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model, self.tokenizer = self._load_model()
        self.model.eval()

    def _load_model(self) -> tuple[Any, Any]:
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True,
        )

        target_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        model = AutoModel.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            dtype=target_dtype,
            use_flash_attn=True,
            device_map=self.device if self.device == "auto" else None,
        )
        if self.device != "auto":
            model = model.to(self.device)

        return model, tokenizer

    def _build_question(self, question: str, num_images: int) -> str:
        """Build the question string that ``model.chat()`` expects.

        Official format for multi-image with separate labels:
            Image-1: <image>
            Image-2: <image>
            {actual_question}
        """
        system = self.config.system_prompt or DEFAULT_SYSTEM_PROMPT

        lines = [system, ""]
        for i in range(1, num_images + 1):
            lines.append(f"[文档{i}]")
            lines.append(f"Image-{i}: <image>")
            lines.append("")
        lines.append(f"问题：{question}")
        lines.append("")
        lines.append("请直接输出答案：")
        return "\n".join(lines)

    @torch.no_grad()
    def generate(
        self,
        question: str,
        image_paths: Sequence[str | Path],
    ) -> dict[str, Any]:
        if not image_paths:
            return _uncertain_result()

        images = _load_images(image_paths)
        if not images:
            return _uncertain_result()

        num_images = len(images)

        # Official preprocessing: dynamic tiling + ImageNet normalization
        all_pv: list[torch.Tensor] = []
        for img in images:
            all_pv.append(_load_image_internvl(img))

        num_patches_list = [pv.shape[0] for pv in all_pv]
        pixel_values = torch.cat(all_pv, dim=0).to(
            device=self.model.device,
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        )

        prompt = self._build_question(question, num_images)

        gen_config = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
        }
        if self.config.do_sample:
            gen_config.update({
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
            })

        response = self.model.chat(
            self.tokenizer, pixel_values, prompt, gen_config,
            num_patches_list=num_patches_list if num_images > 1 else None,
        )

        for img in images:
            img.close()

        return self._normalize_result(response.strip(), num_images)

    @torch.no_grad()
    def batch_generate(
        self,
        questions: Sequence[str],
        image_paths_list: Sequence[Sequence[str | Path]],
    ) -> list[dict[str, Any]]:
        return [self.generate(q, p) for q, p in zip(questions, image_paths_list)]

    def _normalize_result(
        self, raw_answer: str, num_images: int,
    ) -> dict[str, Any]:
        if self.normalizer:
            cleaned = self.normalizer.clean_extracted_answer(raw_answer)
            is_uncertain = self.normalizer.is_uncertain_answer(cleaned)
        else:
            cleaned = raw_answer
            is_uncertain = False

        return {
            "answer": cleaned,
            "raw_answer": raw_answer,
            "is_uncertain": is_uncertain,
            "num_images": num_images,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class GeneratorFactory:
    """Factory that auto-detects the correct generator from the model name."""

    @staticmethod
    def create(
        config: GeneratorConfig,
        normalizer: Any | None = None,
    ) -> BaseGenerator:
        """Create a generator instance based on the model name in *config*.

        Detection rules (checked in order):
        - ``llava`` or ``llava-onevision`` → :class:`LLaVAGenerator`
        - ``internvl`` or ``internvl2``       → :class:`InternVLGenerator`
        - ``qwen``                             → :class:`QwenVLGenerator` (default)
        """
        name = config.model_name.lower()

        if "llava" in name:
            return LLaVAGenerator(config, normalizer)
        elif "internvl" in name:
            return InternVLGenerator(config, normalizer)
        elif "qwen" in name:
            return QwenVLGenerator(config, normalizer)
        else:
            raise ValueError(
                f"Cannot determine generator type for model: {config.model_name}. "
                f"Supported families: Qwen-VL, LLaVA-OneVision, InternVL."
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_img_context_token_id(tokenizer: Any) -> int | None:
    """Find the IMG_CONTEXT token id from an InternVL tokenizer.

    InternVL stores this in ``tokenizer.img_context_token_id`` or adds
    ``<IMG_CONTEXT>`` as a special token.  We try several common locations.
    """
    # 1. Direct attribute (some tokenizers set it)
    for attr in ("img_context_token_id", "image_token_id"):
        val = getattr(tokenizer, attr, None)
        if val is not None:
            return int(val)

    # 2. Known special token strings
    candidates = [
        "<IMG_CONTEXT>",
        "<img>",
        "<image>",
        "<|image|>",
        "<|vision_start|>",
    ]
    for tok in candidates:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid >= 0:
            return tid

    # 3. Added tokens vocabulary
    added = getattr(tokenizer, "added_tokens_encoder", {})
    for tok in candidates:
        if tok in added:
            return int(added[tok])

    return None


def _load_images(paths: Sequence[str | Path]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for p in paths:
        try:
            images.append(Image.open(p).convert("RGB"))
        except Exception as e:
            print(f"[warning] Failed to load image {p}: {e}")
            continue
    return images


def _uncertain_result() -> dict[str, Any]:
    return {
        "answer": "无法确定",
        "raw_answer": "",
        "is_uncertain": True,
        "num_images": 0,
    }
