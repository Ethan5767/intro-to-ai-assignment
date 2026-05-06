# MultiLexNorm 2026 — 성균관대 "인공지능개론" 제출 프로젝트

> 🇰🇷 한국어 (현재 파일) · [🇺🇸 English README](README.md)

**W-NUT 2026 MultiLexNorm 2** 공동 과제를 위한 재현 가능 파이프라인입니다. 17개 언어의 소셜 미디어 노이즈 텍스트에 대한 어휘 정규화(lexical normalization)를 수행합니다. 언어별로 바이트 단위(ByT5-small)와 서브워드 단위(mT5-small) 시퀀스-투-시퀀스 모델을 2021년 우승자(ÚFAL)의 토큰-투-토큰 형식으로 학습시키고, MFR 베이스라인과 비교한 뒤, 개발 셋(dev set)에서 가장 좋은 모델을 언어마다 선택해 최종 제출본을 만듭니다.

> 과제 페이지: <https://noisy-text.github.io/2026/multi-lexnorm.html>
> 데이터셋(접근 승인 필요): <https://huggingface.co/datasets/weerayut/multilexnorm2026-dev-pub>
> CodaBench: <https://www.codabench.org/competitions/14162/>

---

## 저장소 구성

| 경로 | 역할 |
| --- | --- |
| `data_prep.py` | 원문장을 ÚFAL 단어 단위 형식 (`<L> ... </L> <T> 단어 </T> <R> ... </R>`)으로 변환 |
| `train_seq2seq.py` | ByT5-small **또는** mT5-small을 한 언어에 대해 파인튜닝 (공유 스크립트) |
| `predict_seq2seq.py` | 학습된 체크포인트로 `validation` 또는 `test` 셋에 대해 추론하고 문장 단위로 재조립 |
| `run_mfr.py` | MFR 베이스라인 전 과정 실행: 카운트 → 테스트 예측 → `outputs/summary_mfr.csv` 작성 |
| `run_pipeline.py` | 한 모델에 대해 여러 언어의 학습+예측을 순차 실행하는 드라이버 (재실행 안전) |
| `picker.py` | 개발 셋 ERR을 보고 언어별로 최적 모델 선택 (ByT5에 우선권을 둔 2점 마진 타이브레이크 적용) |
| `make_submission.py` | 언어별 선택 모델의 예측을 모아서 `predictions.json` 생성 후 평탄(flat) 압축 |
| `tests/` | data prep, 예측 재조립, picker 로직에 대한 단위 테스트 |
| `colab/train_one_language.ipynb` | 클릭 한 번으로 시작하는 Colab 노트북 (저장소 클론 → 의존성 설치 → 한 언어 학습 → 평가) |
| `utils.py` | 과제 주최측이 제공한 MFR · 평가 함수 · 평탄 zip 도우미 |
| `demo.ipynb` | 주최측의 원본 MFR 데모 (참고용) |

학습된 체크포인트는 `./ckpts/<model>/<lang>/best/` 아래에 저장됩니다 (gitignore됨). 예측 파일과 모델별 요약은 `./outputs/` 아래에 들어갑니다.

---

## Google Colab 빠른 시작 (권장)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ethan5767/intro-to-ai-assignment/blob/main/colab/train_one_language.ipynb)

1. 위 배지를 클릭하거나, [Colab](https://colab.research.google.com/)에서 `colab/train_one_language.ipynb`을 직접 엽니다.
2. 런타임 → 런타임 유형 변경 → **GPU** 선택 (무료 T4면 충분; Pro면 A100).
3. 데이터셋 페이지 <https://huggingface.co/datasets/weerayut/multilexnorm2026-dev-pub>에 가서 *Agree and access*를 클릭해 접근 승인을 받습니다.
4. 셀을 위에서 아래로 차례로 실행합니다 — Hugging Face 로그인, MFR 베이스라인, 한 언어에 대한 ByT5 학습/평가, 그리고 (선택) 결과를 Google Drive에 백업하는 단계까지 안내합니다.

무료 T4에서 `--limit-train 5000 --epochs 3` 옵션으로 영어(en) ByT5-small을 학습하면 약 10–15분 정도 걸립니다. 전체 학습 셋을 쓰면 더 오래 걸리니, 17개 언어 모두를 학습하려면 Colab Pro가 권장됩니다.

---

## 로컬 환경 빠른 시작 (Linux / macOS / Windows + CUDA GPU)

```bash
# 1. 클론 후 가상환경 생성
git clone https://github.com/Ethan5767/intro-to-ai-assignment.git
cd intro-to-ai-assignment
python -m venv .venv
source .venv/bin/activate            # Windows의 경우: .venv\Scripts\Activate.ps1

# 2. 의존성 설치
pip install -r requirements.txt

# 3. Hugging Face 로그인 (데이터셋이 게이팅돼 있음)
#    https://huggingface.co/settings/tokens 에서 read 토큰 발급
huggingface-cli login

# 4. 단위 테스트 실행 (GPU/데이터/토큰 불필요)
python -m pytest tests/ -q

# 5. 17개 언어 전체 MFR 베이스라인
python run_mfr.py
```

한 언어를 학습하려면:

```bash
python train_seq2seq.py --model byt5-small --lang en --epochs 3 --limit-train 5000
python predict_seq2seq.py \
    --ckpt ckpts/byt5-small/en/best \
    --lang en --split validation \
    --out-json outputs/byt5/en/predictions_dev.json
```

여러 언어를 한 번에 학습+예측+채점:

```bash
python run_pipeline.py --model byt5-small --langs en de nl es --epochs 3
python run_pipeline.py --model mt5-small  --langs en de nl es --epochs 3
```

---

## 최종 제출본 만들기

```bash
# 1. 모든 언어에 대한 MFR (outputs/summary_mfr.csv 생성)
python run_mfr.py

# 2. 모든 언어에 대한 ByT5 (outputs/summary_byt5-small.csv 생성)
python run_pipeline.py --model byt5-small

# 3. (선택) ByT5가 MFR보다 약했던 언어 또는 ablation을 위해 mT5 학습
python run_pipeline.py --model mt5-small --langs ko th ja vi en id hr sr

# 4. 언어별로 dev 최적 모델 선택
python picker.py
# 표를 출력하고 outputs/picker.json 작성

# 5. CodaBench용 outputs/final/submission.zip 생성
python make_submission.py
```

`outputs/final/submission.zip`을 `g.skku.edu` 계정으로 <https://www.codabench.org/competitions/14162/>에 업로드하세요.

---

## 주요 옵션

| 플래그 (기본값) | 의미 |
| --- | --- |
| `--model {byt5-small,mt5-small}` | 어떤 사전학습 모델을 파인튜닝할지 |
| `--lang <code>` | 17개 중 하나: `en de nl es it hr sr sl da tr id iden trde ja ko th vi` |
| `--ckpt-root` (`./ckpts`, 환경변수 `$MLN_CKPT_ROOT`) | 체크포인트 저장 경로 |
| `--epochs` (`3`) | 학습 에폭 수 |
| `--limit-train` (`-1`) | 학습 예제 수 제한 — 무료 GPU에서 빠른 실험에 유용 |
| `--batch-size` (`2`) / `--grad-accum` (`8`) | 실효 배치 크기 = `batch_size * grad_accum` |
| `--max-in` (`256`) / `--max-out` (`64`) | 입력/출력 토큰 길이 상한; CJK · 태국어는 384/96으로 올리는 게 안전 |
| `--num-beams` (`4`) | 추론 시 빔 너비; `1`(그리디)이면 훨씬 빠름 |

GPU 메모리: `bs=2 grad_accum=8 bf16` 설정에서 ByT5/mT5 small이 6 GB에 들어갑니다. T4(16 GB)면 `bs=4 grad_accum=4`, A100(40 GB)이면 `bs=16 grad_accum=1`까지 무리 없이 돌아갑니다.

---

## Picker 동작 원리

언어별로 dev ERR이 가장 높은 모델을 고릅니다. 단, 작은 dev 셋의 노이즈에 과적합되지 않도록 ByT5를 기본값으로 두고, 다른 모델이 절대 ERR로 2점 이상 더 좋을 때만 교체합니다 (`--min-gap`로 조정 가능).

정확히 동점일 때는 `byt5 > mt5 > mfr` 순서로 결정합니다. 정확한 동작은 `picker.py`와 `tests/test_picker.py`에 정의돼 있습니다.

---

## 하드웨어 · 환경 메모

- **Python:** 3.11과 3.12에서 검증됨. 3.13은 일부 의존성(특히 구버전 `pyarrow`)이 아직 휠을 제공하지 않아 권장하지 않습니다.
- **PyTorch:** CUDA 지원 2.1 이상. CPU만으로도 MFR과 테스트는 돌아가지만 ByT5/mT5 파인튜닝은 사실상 불가능합니다.
- **VRAM:** `--batch-size 2 --grad-accum 8 --bf16` 기준 최소 6 GB. 태국어/CJK는 UTF-8 멀티바이트 때문에 시퀀스가 길어지므로 `--max-in 384`로 올리고 배치 크기를 줄여야 합니다.
- **디스크:** 체크포인트 하나당 약 1.2 GB(ByT5-small) / 1.1 GB(mT5-small). 17개 언어 × 2개 모델 모두 학습하면 약 40 GB가 필요합니다.

---

## 배경과 설계 근거

- **왜 ByT5 + mT5 + MFR을 다 쓰는가?** 한 모델이 모든 언어에서 이기지 못합니다. ÚFAL 2021년 우승 시스템은 ByT5 기반이지만, MultiLexNorm++ (Buaphet 외, 2026년 1월) 논문은 태국어와 한국어에서 MFR이 모든 신경망 모델을 이긴다는 결과를 보였습니다. ByT5는 멀티바이트 UTF-8 문자(태국어, CJK)에서 약합니다. 언어별로 픽업하면 어떤 단일 모델보다도 dev에서 최소한 같거나 높은 성능이 보장됩니다.
- **왜 토큰-투-토큰 형식인가?** ÚFAL 논문에 따르면 문장-투-문장 학습은 모델이 정규화 외에 토큰 정렬까지 동시에 학습해야 해 성능이 떨어집니다. 단어별 입력 + 좌우 3토큰 컨텍스트 윈도우가 인코더가 정규화에만 집중할 수 있게 해 줍니다.
- **왜 2점 마진 임계값인가?** 일부 언어의 dev 셋은 매우 작습니다 (예: ~500문장). 1점 ERR 차이는 시드만 바꿔도 뒤집힐 수 있습니다. 2점은 보수적으로 잡은 임계값입니다.

전체 문헌 리뷰와 언어별 예측 결과는 [`CLAUDE.md`](../CLAUDE.md)와 계획 문서 [`plans/2026-05-05-byt5-mt5-training.md`](../plans/2026-05-05-byt5-mt5-training.md)를 참고하세요.

---

## 테스트

```bash
python -m pytest tests/ -q
```

테스트가 다루는 것:
- `to_token_to_token`: 형식 · 컨텍스트 윈도우 자르기 · 다단어 정규화
- `reassemble_predictions`: 문장 단위 재조립 · 순서 불변
- `pick_best`: 우선순위 · 마진 임계값 동작

GPU, 데이터셋, HF 토큰 모두 필요 없으니 CI에서 그대로 돌릴 수 있습니다.

---

## 인용

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

## 라이선스

MIT — [`LICENSE`](LICENSE) 참고.
