# CHaystack

<img width="1091" height="856" alt="系统图_png" src="https://github.com/user-attachments/assets/4f41ff13-8033-44d1-a1b1-051f147a7150" />

CHaystack is a benchmark for
Chinese document image question answering. It builds a visual document corpus
from multiple public datasets, indexes document images with a multimodal
embedding retriever, and evaluates RAG pipelines that retrieve relevant images
before generating answers with a vision-language model.

## Installation

```bash
pip install -r requirements.txt
```

## Build the Benchmark Corpus

Download and process the benchmark data with:

```bash
bash benchmark/build_corpus.sh
```

The script downloads the supported data sources into `benchmark/data/`, then
runs `benchmark/process.py` to organize images into `benchmark/data/image/`.

The download step can take a long time and depends on network availability. If a
single dataset download fails, fix the environment/network issue, rerun the
corresponding `benchmark/download_*.py` script, and then rerun
`benchmark/process.py`.

## Build Image Indexes

Before running RAG, build image indexes for the benchmark images:

```bash
bash scripts/build_index.sh
```

By default, this creates category-specific indexes under `data/indexes/`, such
as `data/indexes/benchmark_paper` and `data/indexes/benchmark_webpage`. Edit
`scripts/build_index.sh` to change the embedding model, device, batch size, or
which categories to index.

## Quick Start: Filter RAG

For a quick end-to-end run, use the phased Filter RAG pipeline:

```bash
python -m src.pipline.filter_rag \
  --phase all \
  --benchmark_root benchmark \
  --index_root data/indexes \
  --retriever_model qwen3-vl-embedding-2b \
  --filter_model Qwen/Qwen2.5-VL-3B-Instruct \
  --generator_model Qwen/Qwen2.5-VL-3B-Instruct \
  --candidate_k 15 \
  --top_k 10 \
  --limit 20 \
  --device cuda \
  --retrieval_cache data/output/phase1_retrieval.jsonl \
  --candidate_cache data/output/phase2_filtered.jsonl \
  --filter_cache_path data/output/filter_cache.json \
  --output_path data/output/filter_rag.jsonl
```

`--phase all` runs the three stages sequentially:

1. `retrieve`: retrieve candidate images from the index;
2. `filter`: use a vision-language model to keep question-relevant images;
3. `generate`: generate the final answer from the filtered candidates.

For larger experiments or ablations, you can also run:

```bash
bash scripts/run_filter_rag_ablation.sh
```

This script caches retrieval and filtering results so different generator
settings can reuse the same intermediate files.
