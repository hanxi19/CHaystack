from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PLANNER_PROMPT = """你是中文文档检索 query 规划器。
请把用户问题改写成适合跨模态文档检索的多个子查询。

要求：
1. 保留能定位文档的实体、年份、机构、品牌、产品、金额、日期、型号、标题、术语等锚点。
2. 不要把待回答的答案直接写进检索 query。
3. retrieval_queries 应包含 2 到 4 条短查询，优先覆盖不同检索视角。
4. answer_slot 用英文短标签描述问题要问的答案类型，例如 author、date、price、phone、company、product、value、other。
5. 只输出 JSON，不要输出解释。

输出格式：
{
  "anchors": ["锚点1", "锚点2"],
  "answer_slot": "other",
  "retrieval_queries": ["检索子查询1", "检索子查询2"]
}
"""


@dataclass
class AnchorQueryPlan:
    original_question: str
    anchors: list[str]
    answer_slot: str
    retrieval_queries: list[str]
    raw_text: str = ""
    fallback: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clean_json_response(text: str) -> str:
    cleaned = text.strip()
    if "```" in cleaned:
        for block in cleaned.split("```"):
            block = block.strip()
            if block.lower().startswith("json"):
                block = block[4:].lstrip()
            if block.startswith("{"):
                cleaned = block
                break
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    return cleaned.strip()


def _normalize_text(text: Any) -> str:
    text = str(text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n，。；;、")


def _dedupe_keep_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        item = _normalize_text(item)
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


class AnchorQueryPlanner:
    """LLM-based planner that turns a question into anchor-oriented queries."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        max_new_tokens: int = 256,
        system_prompt: str = DEFAULT_PLANNER_PROMPT,
        trust_remote_code: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
        )
        self.model.to(self.device)
        self.model.eval()

    def plan(self, question: str, *, max_queries: int = 4) -> AnchorQueryPlan:
        question = _normalize_text(question)
        if not question:
            return self._fallback_plan(question, raw_text="")

        prompt = self._build_prompt(question)
        raw_text = self._generate(prompt)
        try:
            parsed = json.loads(_clean_json_response(raw_text))
            return self._parse_plan(question, parsed, raw_text, max_queries=max_queries)
        except Exception:
            return self._fallback_plan(question, raw_text=raw_text)

    def _build_prompt(self, question: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"用户问题：{question}"},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"{self.system_prompt}\n\n用户问题：{question}\nJSON："

    @torch.no_grad()
    def _generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _parse_plan(
        self,
        question: str,
        parsed: dict[str, Any],
        raw_text: str,
        *,
        max_queries: int,
    ) -> AnchorQueryPlan:
        anchors_raw = parsed.get("anchors") or []
        queries_raw = parsed.get("retrieval_queries") or parsed.get("queries") or []

        if not isinstance(anchors_raw, list):
            anchors_raw = []
        if not isinstance(queries_raw, list):
            queries_raw = []

        anchors = _dedupe_keep_order([str(item) for item in anchors_raw])[:8]
        queries = _dedupe_keep_order([str(item) for item in queries_raw])
        queries = [query for query in queries if query != question]

        anchor_query = " ".join(anchors[:6]).strip()
        all_queries = _dedupe_keep_order([anchor_query, *queries, question])
        all_queries = all_queries[: max(1, max_queries)]

        answer_slot = _normalize_text(parsed.get("answer_slot") or "other")
        answer_slot = re.sub(r"[^A-Za-z0-9_-]+", "_", answer_slot).strip("_").lower()
        if not answer_slot:
            answer_slot = "other"

        return AnchorQueryPlan(
            original_question=question,
            anchors=anchors,
            answer_slot=answer_slot,
            retrieval_queries=all_queries,
            raw_text=raw_text,
            fallback=False,
        )

    def _fallback_plan(self, question: str, *, raw_text: str) -> AnchorQueryPlan:
        return AnchorQueryPlan(
            original_question=question,
            anchors=[],
            answer_slot="other",
            retrieval_queries=[question] if question else [],
            raw_text=raw_text,
            fallback=True,
        )
