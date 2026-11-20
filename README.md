# LifeMate — Zalo AI Challenge 2026

Giọng nói tiếng Việt → ASR → agent gọi tool qua `zaic_tools` → `submission.jsonl` (`id`, `tool_calls`, `state_hash`).

```
audio (lượt cuối + history) ─► ASR (gipformer / PhoWhisper) ─► chuẩn hoá số, giờ ─┐
bối cảnh + bảng ngày + schema app đã cài + history + 2 ví dụ few-shot ──────────────┼─► prompt (một hàm cho cả SFT và suy luận)
                                                                                    ▼
             ┌──────────── kết quả tool ◄──── ToolEnv.call ◄──── guard ◄──── LLM 4B + LoRA (1 hành động JSON / bước)
             └────────────────────────────────────────────────────────────────────►  dừng: finish | ask | hết giờ
```

Pipeline huấn luyện: transcript ASR chưa fine-tune + transcript chuẩn → `build-sft` (đổi tên app/tool, gỡ app → ca
ngoài phạm vi, app gây nhiễu) → LoRA → RFT (lấy mẫu, giữ quỹ đạo đạt 100) → hiệu chỉnh bias → train trên toàn bộ train
→ merge → Docker. Mọi lựa chọn chấm trên **leave-apps-out** (giữ lại toàn bộ mẫu của ~10 app), vì public/private có app
chưa từng xuất hiện trong train.

Chi tiết model, dữ liệu huấn luyện và cấu hình suy luận: `docker/TRAINING.md`.

## Ràng buộc của cuộc thi

- Mỗi mẫu đúng **một** `ToolEnv` khi tạo bài nộp; mọi call đã chạy đều vào file nộp (`env.get_tool_calls()`).
- Không đọc `<split>_environment/` (agent chỉ xem dữ liệu qua tool). Không gán nhãn public/private dưới mọi hình thức.
- Model gốc phải ra trước 26/10/2026; không dùng checkpoint các đội khác công bố.
- Chạy offline, giải mã greedy, seed cố định, từng mẫu một cho bản cuối.
- Bản nộp Docker: `/code/predict.sh`, `start_jupyter.sh`, `predict_notebook.ipynb`, `training_code/`; trường `time`;
  notebook phải ra cùng kết quả với `predict.sh`.

## Cấu trúc

```
lifemate/
  data.py        split, mẫu, nạp đúng zaic_tools của từng split, transcript
  catalog.py     schema tool, render gọn cho prompt, x-ref / fuzzy / x-one-of, định dạng ask
  textnorm.py    chuẩn hoá số tiền, giờ, %, dãy số điện thoại tiếng Việt
  timeutil.py    bảng ngày, nhận dạng + ép định dạng ngày giờ theo nhãn train
  prompt.py      dựng prompt (dùng chung SFT/suy luận)        fewshot.py  TF-IDF ký tự
  actions.py     định dạng hành động JSON                     guard.py    lan can trước env.call
  agent.py       vòng lặp theo lô, chấm log-prob loại hành động + bias
  llm.py         backend oracle / scripted / HF / vLLM        runner.py   chạy split → submission + trace
  scorer.py      công thức chấm + phân loại lỗi               splits.py   leave-apps-out, ngẫu nhiên
  diag.py        đếm số liệu cho các luật tuỳ chọn của guard       diffruns.py so điểm từng mẫu giữa hai lần chấm
  experiments.py thống kê + 4 thí nghiệm                      sft_data.py, augment.py  dữ liệu SFT
  train_lora.py  LoRA SFT                                     rft.py, calibrate.py, merge_lora.py, soup_lora.py, count_params.py
  asr/           audio, engines (gipformer, PhoWhisper), transcribe, metrics, finetune_whisper, merge_whisper,
                 export_manifest
  predict.py     suy luận trong image (predict.sh + notebook)      compare.py  so hai file nộp + time
configs/default.json  tham số prompt / guard / agent / sft (gốc, dùng khi train)
configs/docker.json   cấu hình của bản nộp (predict.sh)
docker/               Dockerfile, predict.sh, predict.py, preprocessing.py, start_jupyter.sh,
                      predict_notebook.ipynb, TRAINING.md (→ /code/training_code/README.md)
scripts/pipeline.sh   các bước chính      scripts/download_models.py
tests/                dữ liệu giả lập + mock zaic_tools, unit test, smoke test model tí hon
```
