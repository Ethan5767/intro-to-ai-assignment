# MultiLexNorm 2026 — SKKU "Introduction to AI" Submission

> 🇺🇸 English (this file) · [🇰🇷 한국어 README](README.ko.md)

A reproducible pipeline for the **W-NUT 2026 MultiLexNorm 2** shared task: lexical normalisation of noisy social-media text across 17 languages. We train per-language byte- and subword-level seq2seq models (ByT5-small, mT5-small) in the ÚFAL token-to-token format from the 2021 winner, compare them against the MFR baseline, and pick the dev-best model per language for the final submission.

> Task page: <https://noisy-text.github.io/2026/multi-lexnorm.html>
> Dataset (gated): <https://huggingface.co/datasets/weerayut/multilexnorm2026-dev-pub>
> CodaBench: <https://www.codabench.org/competitions/14162/>

---

## What's in this repo

| Path | Purpose |
| --- | --- |
| `data_prep.py` | Convert raw sentences into ÚFAL per-word `<L> ... </L> <T> word </T> <R> ... </R>` examples |
| `train_seq2seq.py` | Fine-tune ByT5-small **or** mT5-small for one language (single shared script) |
| `predict_seq2seq.py` | Run a fine-tuned checkpoint on `validation` or `test`, reassemble per-sentence predictions |
| `run_mfr.py` | MFR baseline end-to-end: counts → predict on test → write `outputs/summary_mfr.csv` |
| `run_pipeline.py` | Driver: train + predict for many languages of one model, idempotent |
| `picker.py` | Pick the best model per language from dev ERR (with a 2-point min-gap tiebreak in favour of ByT5) |
| `make_submission.py` | Pull each language's predictions from its picked model, write `predictions.json` and zip flat for CodaBench |
| `tests/` | Unit tests for data prep, prediction reassembly, and picker logic |
| `colab/train_one_language.ipynb` | One-click Colab notebook (clones repo, installs deps, trains one language, scores it) |
| `utils.py` | MFR + evaluator + flat-zip helper provided by the task organisers |
| `demo.ipynb` | Original task organisers' MFR demo, kept for reference |

Trained checkpoints land under `./ckpts/<model>/<lang>/best/` (gitignored). Predictions and per-model summaries land under `./outputs/`.

---

## Quickstart on Google Colab (recommended)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ethan5767/intro-to-ai-assignment/blob/main/colab/train_one_language.ipynb)

1. Click the badge above (or open `colab/train_one_language.ipynb` in [Colab](https://colab.research.google.com/) manually).
2. Runtime → Change runtime type → **GPU**.
3. Accept the dataset terms at <https://huggingface.co/datasets/weerayut/multilexnorm2026-dev-pub> (one click, "Agree and access").
4. Run cells top to bottom — it walks you through HF login, the MFR baseline, ByT5 training on one language, evaluation, and (optionally) saving artefacts to Google Drive.

A free T4 trains ByT5-small on English in ~10–15 min with `--limit-train 5000 --epochs 3`. The full split takes longer; budget Colab Pro if you want to train all 17 languages.

---

## Quickstart locally (Linux / macOS / Windows + CUDA GPU)

```bash
# 1. Clone + create a virtual env
git clone https://github.com/Ethan5767/intro-to-ai-assignment.git
cd intro-to-ai-assignment
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1

# 2. Install
pip install -r requirements.txt

# 3. Hugging Face auth (datasets are gated)
#    Get a read token at https://huggingface.co/settings/tokens
huggingface-cli login

# 4. Sanity tests (no GPU/data needed)
python -m pytest tests/ -q

# 5. MFR baseline on all 17 languages
python run_mfr.py
```

Then to train one language:

```bash
python train_seq2seq.py --model byt5-small --lang en --epochs 3 --limit-train 5000
python predict_seq2seq.py \
    --ckpt ckpts/byt5-small/en/best \
    --lang en --split validation \
    --out-json outputs/byt5/en/predictions_dev.json
```

Or train + predict + score for many languages in one go:

```bash
python run_pipeline.py --model byt5-small --langs en de nl es --epochs 3
python run_pipeline.py --model mt5-small  --langs en de nl es --epochs 3
```

---

## Producing the final submission

```bash
# 1. MFR on every language (gives outputs/summary_mfr.csv)
python run_mfr.py

# 2. ByT5 on every language (writes outputs/summary_byt5-small.csv)
python run_pipeline.py --model byt5-small

# 3. (Optional) mT5 on the languages where ByT5 underperformed MFR or for the ablation
python run_pipeline.py --model mt5-small --langs ko th ja vi en id hr sr

# 4. Pick the dev-best model per language
python picker.py
# prints a table; writes outputs/picker.json

# 5. Assemble outputs/final/submission.zip for CodaBench
python make_submission.py
```

Upload `outputs/final/submission.zip` to <https://www.codabench.org/competitions/14162/> with a `g.skku.edu` account.

---

## Configuration knobs

| Flag (default) | Meaning |
| --- | --- |
| `--model {byt5-small,mt5-small}` | Which preset to fine-tune |
| `--lang <code>` | One of `en de nl es it hr sr sl da tr id iden trde ja ko th vi` |
| `--ckpt-root` (`./ckpts`, env `$MLN_CKPT_ROOT`) | Where checkpoints land |
| `--epochs` (`3`) | Training epochs |
| `--limit-train` (`-1`) | Cap training examples — useful on free-tier GPUs |
| `--batch-size` (`2`) / `--grad-accum` (`8`) | Effective batch is `batch_size * grad_accum` |
| `--max-in` (`256`) / `--max-out` (`64`) | Token caps; raise for long inputs (CJK / Thai may need 384/96) |
| `--num-beams` (`4`) | Beam width at inference; `1` is much faster |

GPU memory budget: ByT5/mT5 small fit in 6 GB at `bs=2 grad_accum=8 bf16`. On a T4 you can usually go `bs=4 grad_accum=4`; on an A100 `bs=16 grad_accum=1`.

---

## Reproducing the per-language picker logic

The picker selects the best dev ERR per language with one twist: ByT5 is the default unless another model wins by ≥ 2 absolute ERR points. The threshold prevents picker-overfitting on small noisy dev splits. Tunable with `--min-gap`.

The `priority` order on exact ties is `byt5 > mt5 > mfr`. See `picker.py` and `tests/test_picker.py` for the exact contract.

---

## Hardware + environment notes

- **Python:** tested on 3.11 and 3.12. Avoid 3.13 — older `pyarrow` wheels are not yet built for it.
- **PyTorch:** 2.1+ with CUDA. CPU-only works for MFR and tests but will not fine-tune ByT5/mT5 in any reasonable time.
- **VRAM:** 6 GB minimum for `--batch-size 2 --grad-accum 8 --bf16`. ByT5 on Thai/CJK can need `--max-in 384` because UTF-8 multibyte characters lengthen the byte sequence; lower `--batch-size` accordingly.
- **Disk:** each checkpoint is ~1.2 GB (ByT5-small) / ~1.1 GB (mT5-small). Training all 17 languages × 2 models needs ~40 GB.

---

## Background and design choices

- **Why ByT5 + mT5 + MFR?** No single model wins everywhere. ÚFAL's 2021 ByT5 system is the established workhorse, but MultiLexNorm++ (Buaphet et al., Jan 2026) shows MFR beats every neural model on Thai/Korean and ByT5 fragments multi-byte UTF-8 scripts. Picking per language is mathematically guaranteed to be ≥ best of any single model on dev.
- **Why token-to-token?** ÚFAL's 2021 paper: training sentence-to-sentence underperforms because it forces the model to learn alignment in addition to normalisation. Per-word inputs with a 3-token context window let the encoder focus.
- **Why the min-gap?** Dev sets for some languages are tiny (~500 sentences). A 1-point ERR difference can flip across seeds. 2 absolute points is a conservative threshold that has empirically held up.

For the full literature review, design rationale, and per-language predictions, see [`CLAUDE.md`](../CLAUDE.md) and the plan document [`plans/2026-05-05-byt5-mt5-training.md`](../plans/2026-05-05-byt5-mt5-training.md).

---

## Testing

```bash
python -m pytest tests/ -q
```

The tests cover:
- `to_token_to_token` formatting + context-window truncation + multi-word golds
- `reassemble_predictions` regrouping + ordering invariants
- `pick_best` priority + min-gap behaviour

They do **not** require a GPU, the dataset, or a HF token, so CI can run them.

---

## Citing

```bibtex
@inproceedings{ufal-multilexnorm-2021,
    title     = {{ÚFAL} at {MultiLexNorm 2021}: Improving Multilingual Lexical Normalization by Fine-tuning {ByT5}},
    author    = {Samuel, David and Straka, Milan},
    booktitle = {Proceedings of the Seventh Workshop on Noisy User-generated Text (W-NUT 2021)},
    year      = {2021},
}

@misc{multilexnormpp-2026,
    title  = {MultiLexNorm++: Benchmarking large language models for multilingual lexical normalization},
    author = {Buaphet, Weerayut and others},
    year   = {2026},
    eprint = {2601.16623},
    archivePrefix = {arXiv},
}
```

---

## License

MIT — see [`LICENSE`](LICENSE).
