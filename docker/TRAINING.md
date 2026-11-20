# LifeMate — mã nguồn huấn luyện và mô tả giải pháp

Thư mục này nằm ở `/code/training_code/` trong image. Code suy luận dùng chung package `lifemate/` (bản trong
`/code/lifemate/` giống hệt). Cấu hình của bản nộp ghi ở cuối mục 5.

## 1. Ý tưởng

Yêu cầu giọng nói (lượt cuối + các lượt user trong `history`) → ASR → chuẩn hoá số/giờ/tiền (ITN) → một LLM 4B
fine-tune LoRA, chạy vòng lặp agent: mỗi bước sinh **một** hành động JSON, hành động đi qua một lớp kiểm tra (guard)
rồi mới gọi `ToolEnv.call`, kết quả tool đưa lại vào hội thoại, lặp tới khi `finish` hoặc hỏi lại thông tin thiếu.

- **Suy luận:** `predict.sh` → `predict.py` (`lifemate/predict.py`). Nạp LLM (vLLM) và ASR một lần, rồi chạy từng mẫu:
  tạo `ToolEnv` → đọc audio + ASR → agent → `get_tool_calls()` + `get_state_hash()`. Giải mã greedy, seed cố định.
- **ASR có ngữ cảnh (tuỳ chọn, `lifemate/asr/context.py`):** khi nghe lượt cuối, PhoWhisper nhận làm prompt câu trả
  lời gần nhất của trợ lý và các tên trong `tool_calls`/`tool_results` của `history` (đều là input của mẫu, không đọc
  gì từ môi trường), để nghe đúng tên người, tên hoá đơn, câu trả lời ngắn ("Zalo nhé em"). Cách giải mã (prompt có
  hay không, số beam) lưu trong `saved_models/asr/lifemate_asr.json`; không có file thì greedy, không prompt.
- **Ghép 2 bản nghe (tuỳ chọn, `lifemate/asr/fuse.py`):** lượt cuối được nghe thêm bằng gipformer (68M, CPU). Ở
  những đoạn hai bản khác nhau, giữ chữ của PhoWhisper trừ khi chính LLM của hệ thống (log-xác suất của cả câu, có câu
  trả lời gần nhất của trợ lý làm ngữ cảnh) chấm câu dùng chữ của gipformer cao hơn một ngưỡng `delta` (+`beta`/token).
  `delta`, `beta` chọn theo WER lượt cuối trên tập dev, lưu ở `assets/fuse.json`; không có file thì không ghép.
- **Huấn luyện:** SFT trên quỹ đạo gold của tập train (có tăng cường dữ liệu); ASR fine-tune LoRA trên audio train.
  Chọn cấu hình bằng tập dev **leave-apps-out** (giữ lại toàn bộ mẫu của 10 app) vì test có app mới. LLM của bản nộp
  là trung bình trọng số của 2 LoRA cùng model gốc (mục 5): công thức v5 (không có câu lý do, tăng cường mạnh hơn) và
  công thức v3 (mỗi bước kèm một câu lý do ngắn do một LLM lớn hơn viết sẵn, chỉ lúc train).

## 2. Mô hình

| Thành phần | Model gốc | Tham số | Context | Trong image |
|---|---|---|---|---|
| ASR | [vinai/PhoWhisper-large](https://huggingface.co/vinai/PhoWhisper-large) + LoRA (đã merge) | 1,55B | 30 s audio, 448 token đầu ra | `saved_models/asr` |
| LLM | [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) + LoRA (đã merge; biến thể ở mục 5) | 4,02B (biến thể v5+v3: 4,15B) | gốc 262.144; chạy với 16.384 | `saved_models/llm` |
| ASR phụ | [g-group-ai-lab/gipformer1.5-68M-rnnt](https://huggingface.co/g-group-ai-lab/gipformer1.5-68M-rnnt) (không fine-tune) | 68M | — | `saved_models/asr2` khi bật ghép 2 bản nghe; luôn dùng lúc train (transcript nhiễu cho SFT) |
| LLM viết lý do (chỉ lúc train) | [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) (dự phòng [Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)) | 30,5B | 16.384 | không |
| LLM sinh và giải yêu cầu tổng hợp (chỉ lúc train) | [Qwen/Qwen3.5-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8) (2/2026, Apache-2.0, tắt chế độ suy nghĩ) | 35B (3B hoạt động) | 16.384 | không |

Tổng tham số lúc suy luận ≈ 5,64B (LLM 4,02B + PhoWhisper 1,55B + gipformer 68M); biến thể v5+v3 ≈ 5,77B (model gốc
4,02B + 2 adapter LoRA × 66M, dùng chung model gốc). Kiểm tra: `python -m lifemate count-params --llm saved_models/llm
--asr saved_models/asr --asr saved_models/asr2` (đếm mọi file trọng số trong các thư mục, kể cả adapter).
Image chứa trọng số đã fine-tune và merge; riêng biến thể v5+v3 chứa model gốc Qwen3-4B-Instruct-2507 (chưa merge) cùng
2 adapter ở `saved_models/llm/adapters/{main,consult}`. Model gốc tải lại từ các link trên khi train.

## 3. Vòng lặp agent (tool calling)

Prompt (một hàm `lifemate/prompt.py` dùng cho cả dữ liệu SFT và lúc suy luận):
- `context`: thời gian hiện tại, múi giờ, vị trí, hồ sơ người dùng; kèm bảng ngày (hôm nay, mai, thứ trong tuần...).
- Schema rút gọn của các app đã cài trong mẫu (`installed_apps` ∩ catalog của split) và `sys.ask_missing_info`.
- `history` (các lượt trước, lượt user lấy từ ASR) và transcript lượt cuối, kèm dạng đã chuẩn hoá ITN.
- 2 ví dụ few-shot lấy từ train bằng TF-IDF n-gram ký tự (`assets/fewshot_pool.jsonl`).

Mỗi bước model sinh `{"thought": ..., "type": "read"|"write"|"ask", "name": ..., "arguments": {...}}` hoặc
`{"thought": ..., "type": "finish"}`: `thought` là một câu ngắn nêu dữ kiện, phép tính hay điều kiện dẫn tới hành động
(ví dụ cộng các hoá đơn rồi so với số dư), viết trước khi chọn tool. Prompt chỉ đòi `thought` khi model được train
với nó: `train` ghi `lifemate_prompt.json` cạnh adapter, `merge-lora` chép sang model đã merge, lúc suy luận đọc file
này (không có file thì theo `configs/default.json`).
Guard (`lifemate/guard.py`) trước khi gọi env: sửa tên tool gần đúng, bỏ tham số lạ, ép kiểu, ép định dạng ngày giờ
theo định dạng học từ nhãn train (`assets/formats.json`), kiểm tra ID tham chiếu (`x-ref`) phải xuất hiện trong kết quả
trước, trả kết quả cache cho lệnh đọc lặp (kể cả lệnh đọc đã có trong lịch sử), chặn lệnh ghi lặp, kiểm tra
`sys.ask_missing_info` (tool đích + tham số thiếu; câu hỏi chỉ gồm tham số không bắt buộc bị trả lại). Luật thêm, bật
trong `configs/default.json` nếu chấm dev tốt hơn: chặn hỏi ID khi chưa tra app chứa nó (`ask_ids_after_read`), ID
phải lấy từ dữ liệu của đúng app hoặc đã thấy dưới đúng tên khoá (`xref_typed`), đưa tên về dạng lưu trong danh bạ
thấy ở nhãn train (`fix_names`), bỏ tham số lọc tuỳ chọn mà nhãn train không bao giờ truyền cho tool đó (tool có từ 20
lần gọi trong train, `drop_unused_params`). Các luật học từ nhãn train (`assets/formats.json`) cho phép những gì gold
làm: lệnh ghi gold lặp lại thì không chặn, câu hỏi về tham số tuỳ chọn mà gold từng hỏi thì không trả lại. Bản nộp bật
tất cả (dev leave-apps-out, transcript chuẩn, cùng adapter: 49,21 → 50,87 với ba luật đầu). Lệnh bị chặn không gọi env, model nhận phản hồi và sinh lại (tối đa 2
lần/bước). Dừng khi `finish`, sau `ask`, sau 10 bước, hoặc khi mẫu chạy quá 25 s (tính cả ASR); bước cuối được giới
hạn số token để kết thúc trước 28 s kể cả trên GPU chỉ sinh 40 token/s (`hard_limit_s`, `min_tok_per_s`). Mỗi mẫu đúng một
`ToolEnv`; không đọc `<split>_environment/`; mọi call đã chạy đều có trong file nộp.

Cấu hình của bản nộp là `configs/docker.json` (`configs/default.json` là cấu hình gốc, dùng khi train). So với cấu
hình gốc nó bật: bảng ngày cho các mốc tương đối trong yêu cầu ("thứ bảy tuần sau = 2026-10-03", `prompt.date_hints`);
guard chặn lặp lại một lệnh vừa không có tác dụng với cùng tham số (`block_failed_repeats`), bỏ tham số văn bản tuỳ chọn
của app chưa thấy trong nhãn train (`drop_optional_text_unseen`), sửa tên tham số trong câu hỏi lại (`fix_ask_params`);
agent: sau lệnh tra không ra hoặc lệnh ghi không có tác dụng thì kèm ghi chú cách sửa và chưa cho dừng trước khi thử lại
(`recover`), tự tra lại khi model định dừng ngay sau lần tra rỗng — bỏ từ xưng hô trước tên, rồi xem cả danh sách
(`recover_auto`), báo khi lệnh đọc bị lặp (`notice_repeat`); ở 3 bước đầu sinh 5 phương án (1 greedy + 4 lấy mẫu ở nhiệt
độ 0,7, seed cố định), chỉ thay phương án greedy khi quá nửa số phương án chọn lệnh khác và không bao giờ đổi một lệnh
thành dừng (`vote_safe`), không sinh thêm phương án khi mẫu đã chạy quá 12 s. `consult` + `consult_self` (bản nộp):
ở lần dừng đầu tiên (finish hoặc hỏi lại) mà chưa có lệnh ghi nào, chính model đó được hỏi lại một lần trên cùng lịch
sử nhưng với prompt không có câu lý do (`consult_thoughts: off`); nếu nó đưa ra lệnh đọc hoặc ghi mà guard cho chạy thì
nó làm nốt mẫu với prompt đó (`handover`), không thì lệnh dừng chạy như cũ; không hỏi khi mẫu đã chạy quá 15 s (ở biến
thể v5+v3 người được hỏi là adapter v3). Guard `prefer_ref_id`: tool cho chọn ID (`x-ref`) hoặc giá trị thô
(`x-one-of`) mà model điền giá trị thô (số điện thoại) có trong một bản ghi đã tra thì dùng ID của bản ghi đó (nhãn
train: đã tra ra liên hệ thì 122/125 lệnh dùng ID). Agent `force_lookup`: guard trả lại một lệnh vì ID chưa tra và chỉ
ra tool tra (hoặc vì chỉ hỏi tham số không bắt buộc), mà model cứ hỏi lại tới giới hạn hoặc dừng luôn sau đó, thì
agent viết sẵn phần đầu lệnh gọi đúng tool đó, model điền tham số (5 phương án, lấy lệnh hợp lệ nhiều phiếu nhất), mỗi
mẫu một lần; không ra lệnh hợp lệ thì dừng như cũ. Dev leave-apps-out, cùng bản trộn dev: 44,62 → 46,38 (tự hỏi lại)
→ 48,05 (thêm hai luật). Mọi lệnh vẫn qua đúng một `ToolEnv`.

Tuỳ chọn: chọn loại hành động bằng log-prob + độ lệch (bias) hiệu chỉnh trên dev (`assets/bias.json`).
Tuỳ chọn `retry_idle_finish`: nếu ngay bước đầu model trả `finish` mà câu `thought` lại nói có việc cần làm ("đã đủ
thông tin...", "cần gửi..."), và không nói lý do dừng (ngoài phạm vi, không cần làm...), model được nhắc một lần là
chưa gọi tool nào rồi sinh lại.

## 4. Dữ liệu và dữ liệu tổng hợp

Chỉ dùng tập **train** của BTC (tải trên trang cuộc thi; không phát tán lại theo thể lệ). Không dùng dữ liệu ngoài,
không gán nhãn public/private test. `public_test` chỉ dùng để chạy thử định dạng.

- **Input text cho SFT:** trộn transcript gold (`audio_text`) và transcript của gipformer **chưa fine-tune** trên audio
  train (`p_asr = 0.5`), để model quen lỗi ASR thật. Ở bản cuối, các mẫu của tập dev dùng transcript của
  PhoWhisper bản dev (model này không nghe audio của chúng lúc train) thay cho gipformer, để LLM gặp đúng kiểu lỗi
  của ASR dùng lúc suy luận.
- **Nhầm từ của ASR:** với xác suất 0,15 đổi "nhắc" ↔ "nhắn" trong câu đầu vào (lỗi PhoWhisper hay gặp trên dev),
  nhãn giữ nguyên, để model dựa vào ngữ cảnh thay vì một từ.
- **Quỹ đạo gold:** dựng lại từ `label.tool_calls` + `label.tool_results` theo đúng định dạng hành động ở mục 3.
- **Tăng cường (`lifemate/augment.py`, `configs/default.json` → `sft`, `n_aug 1` bản/mẫu):** với xác suất 0,2 đổi
  tên app thành ID ngẫu nhiên và mỗi tool với xác suất 0,5, đồng bộ trong schema, `installed_apps` và nhãn, để model
  đọc schema thay vì học thuộc tên; thêm app gây nhiễu (`p_distractor 0.1`); gỡ app cần dùng khỏi `installed_apps` →
  nhãn rỗng, tức ngoài phạm vi (`p_remove_oos 0.08`). Công thức v5 dùng `p_rename 0.6` và `p_distractor 0.3`: trên dev
  leave-apps-out, cùng cấu hình agent của bản nộp, model train theo cách này đạt 43,33 so với 41,52 của công thức v3
  (`p_rename 0.2`, `p_distractor 0.1`, có câu lý do), tức tổng quát tốt hơn với app mới.
- **Lý do từng bước (`lifemate/rationale.py`, dữ liệu tổng hợp):** Qwen3-30B-A3B-Instruct-2507 (vLLM, greedy) đọc
  đúng những gì agent thấy ở bước đó (bối cảnh, lịch sử, yêu cầu, các bước trước kèm kết quả) cùng bước gold tiếp theo,
  rồi viết một câu lý do ≤ 30 từ, có con số khi bước đó cần tính hay so sánh. Câu này thành trường `thought` của hành
  động gold trong dữ liệu SFT. Mẫu có bước thiếu lý do (prompt quá dài) giữ nhãn không có `thought`; mẫu ngoài phạm vi
  tạo bằng cách gỡ app dùng một câu cố định. Model này chỉ dùng lúc train, không có trong image; các câu lý do nằm
  ở `data/thoughts_train.jsonl` cạnh file này (`{id, step, thought}`).
- **Mẫu tổng hợp (`lifemate/synth.py`, dữ liệu tổng hợp):** với mỗi mẫu train, Qwen3.5-35B-A3B-FP8 (tắt chế độ suy nghĩ) đọc bối
  cảnh, app đã cài, dữ liệu app của chính mẫu đó (`ToolEnv.get_state()`, chỉ trên train như thể lệ cho phép), hội thoại
  trước, yêu cầu thật và đáp án, rồi viết 5 yêu cầu mới cùng kiểu nhưng khác chi tiết. Teacher giải mỗi yêu cầu hai lần
  như agent (qua guard, trên `ToolEnv` mới của train): một lần có ví dụ few-shot, một lần không. Chỉ giữ yêu cầu khi hai
  lần cho cùng chuỗi call và cùng state, không lỗi tool, không bị guard chặn, bước nào cũng có lý do; nhãn là chuỗi
  call đó, kết quả lấy bằng cách chạy lại trên một `ToolEnv` mới. Trước khi sinh, cùng bộ lọc chạy trên val (nhãn
  thật): điểm trung bình của các mẫu được giữ phải ≥ 85, không thì dừng. Mỗi mẫu tổng hợp vào SFT đúng một dòng và không
  lấy mẫu gốc làm ví dụ few-shot. Dữ liệu ở `data/synth_train_q35.jsonl`, `data/synth_thoughts_q35.jsonl`,
  `data/synth_check_q35.json` (khi bản nộp dùng mẫu tổng hợp).
- **RFT (`lifemate/rft.py`, tuỳ chọn):** với model SFT (phần train, bỏ dev), lấy 4 quỹ đạo/mẫu ở nhiệt độ 0,8, chạy
  trên `ToolEnv` của train qua guard như lúc suy luận, giữ quỹ đạo đạt điểm 100 (tối đa 2/mẫu). Quỹ đạo có lệnh bị
  guard chặn thì bỏ các lần thử bị chặn, nên model học thẳng hành động đã sửa. Bỏ mẫu mà cả 4 quỹ đạo đều đúng ngay
  (không có gì để học). Train lại trên SFT + RFT.

## 5. Tham số huấn luyện

| | LLM (`lifemate/train_lora.py`) | ASR (`lifemate/asr/finetune_whisper.py`) |
|---|---|---|
| LoRA | r 32, alpha 64, dropout 0,05; q/k/v/o/gate/up/down_proj | r 32, alpha 64, dropout 0,05; q/k/v/out_proj, fc1, fc2 |
| Tối ưu | lr 1e-4 (v5) / 2e-4 (v3), 2 epoch, batch 1 × grad-accum 16, warmup 3 %, cosine, bf16 | lr 1e-4, 3 epoch (trên dev 2 và 3 epoch ngang nhau), batch hiệu dụng 16; bản có ngữ cảnh: như trên, 60 % lượt train kèm prompt ngữ cảnh (`--p-prompt 0.6`); loss dev thấp nhất sau epoch 2, tăng nhẹ ở epoch 3 |
| Độ dài | 12.288 token, loss chỉ trên lượt assistant | yêu cầu cuối + các lượt history, transcript gold; prompt ≤ 190 token, không tính loss |
| Seed | 0 | 0 |

Seed khác: chia dev `make-split --seed 0`, sinh SFT `seed 0`, RFT seed = chỉ số quỹ đạo, suy luận `seed_everything(42)`
và vLLM `seed=42`, `PYTHONHASHSEED=0`. Kernel GPU có thể lệch rất nhỏ giữa các lần chạy.

**Cấu hình bản nộp — biến thể LLM: `soup` (λ = 0,5, prompt có câu lý do).** Ba biến thể dùng chung ASR, `assets/` và
`configs/docker.json`:
- `v5` (`FINAL_VARIANT=v5 bash scripts/pipeline.sh final`): LoRA công thức v5 trên toàn bộ train, merge → một model 4,02B.
- `v5+v3` (`FINAL_VARIANT=v5+v3`): model gốc + adapter v5 (`adapters/main`) + adapter v3 (`adapters/consult`, công thức v3
  trên toàn bộ train), vLLM nạp một lần cả 2 adapter; LLM 4,15B. Dev leave-apps-out: 43,33 (v5) → 44,88.
- `soup` (`FINAL_VARIANT=soup SOUP_LAM=0.5 SOUP_THOUGHTS=on`, mặc định của `final`): `soup-lora` cộng 2 adapter theo
  trọng số λ·v5 + (1−λ)·v3 (đúng tuyệt đối dưới dạng một adapter rank 64), rồi merge → một model 4,02B. Dev
  leave-apps-out (cặp model dev): 44,62, so với 43,33 của v5 và 41,52 của v3; public 57,20. Với cấu hình agent của
  bản nộp (tự hỏi lại + hai luật ở mục 3): dev 48,05.

Không RFT, không mẫu tổng hợp, không `bias.json`; `TRAIN_ARGS="--max-len 12288"` (`final` tự đặt lr). ASR = PhoWhisper-large LoRA train có prompt ngữ cảnh
(`--p-prompt 0.6`, 3 epoch), suy luận beam 4 không prompt (`lifemate_asr.json`: `{"prompt_final": false, "beams": 4}`);
lượt cuối ghép với gipformer (`assets/fuse.json`: `delta` 6, `beta` 0,5). Agent không bật `retry_idle_finish`. GPU train:
Google Colab G4 (RTX PRO 6000 96 GB) cho LLM, A100 80 GB cho ASR.

## 6. Train lại từ đầu

Cần Linux + GPU (đã chạy trên Google Colab G4 96 GB và A100 80 GB), Python ≥ 3.10, có internet.

```bash
pip install -r requirements.txt -r requirements-gpu.txt
python scripts/download_models.py llm gipformer phowhisper      # -> models/base-llm, models/gipformer, models/phowhisper-large
export DATA=/path/to/train+public_test                          # chứa train/ (và public_test/) của BTC

bash scripts/pipeline.sh day1          # thống kê, chia dev leave-apps-out, assets/ (few-shot pool, định dạng giá trị)
bash scripts/pipeline.sh asr-untuned   # gipformer gốc trên audio train -> outputs/asr_train_untuned.jsonl (input SFT)
bash scripts/pipeline.sh asr-ft-dev    # LoRA PhoWhisper không có audio dev -> outputs/asr_val_ft.jsonl (input SFT bản cuối)
ASR_ARGS="--p-prompt 0.6" ASR_DECODE="--beams 4 --prompt-final off" \
    bash scripts/pipeline.sh asr-ft-full   # LoRA PhoWhisper (prompt ngữ cảnh lúc train) trên toàn bộ audio train -> models/asr
bash scripts/pipeline.sh thoughts      # công thức v3 (soup, v5+v3): Qwen3-30B-A3B viết lý do -> outputs/thoughts_train.jsonl
# bản nộp không dùng mẫu tổng hợp (synth), RFT, bias.json
TRAIN_ARGS="--max-len 12288" bash scripts/pipeline.sh final   # bản nộp: soup, lam 0.5, có lý do (mục 5)
TAG=v1 bash scripts/pipeline.sh sft    # chỉ cho ghép 2 bản nghe bên dưới: checkpoints/sft-v1 chấm câu
# ghép 2 bản nghe -> assets/fuse.json: PhoWhisper dev có prompt ngữ cảnh (không nghe audio dev) nghe lượt cuối của
# dev bằng beam 4, rồi chọn delta/beta theo WER lượt cuối; LLM chấm câu là checkpoints/sft-v1 ở trên
python -m lifemate finetune-whisper --data "$DATA" --model models/phowhisper-large \
    --val-ids outputs/split_apps.val_ids.txt --out checkpoints/phowhisper-lora-dev-v2 --p-prompt 0.6
python -m lifemate transcribe --data "$DATA" --split train --ids-file outputs/split_apps.val_ids.txt --asr whisper \
    --model models/phowhisper-large --lora checkpoints/phowhisper-lora-dev-v2 --num-beams 4 \
    --history-from outputs/asr_val_ft.jsonl --out outputs/asr_val_v2_b4.jsonl
python -m lifemate fuse --data "$DATA" --split train --ids-file outputs/split_apps.val_ids.txt \
    --primary outputs/asr_val_v2_b4.jsonl --secondary outputs/asr_train_untuned.jsonl --model models/base-llm \
    --lora checkpoints/sft-v1 --tune --params-out assets/fuse.json --out outputs/asr_val_v2_b4_fused.jsonl
cp -r models/gipformer models/asr2     # ASR phụ trong image (saved_models/asr2)
```

`final` dựng dữ liệu SFT và train LoRA trên toàn bộ train theo biến thể (mục 5), đưa trọng số vào `models/llm`
(merge, hoặc model gốc + `adapters/`), dựng lại `assets/`.
`models/llm`, `models/asr` và `assets/` chính là `saved_models/llm`, `saved_models/asr` và `assets/` trong image
(`docker/Dockerfile`).

## 7. Cấu trúc `/code`

```
/code/
├── predict.sh, predict.py, preprocessing.py   chạy suy luận: /data/<split>/ -> /result/submission.jsonl
├── start_jupyter.sh, predict_notebook.ipynb   đo thời gian -> /result/jupyter_submission.jsonl
├── requirements.txt                           cài thêm trên vllm/vllm-openai:v0.19.0
├── lifemate/, configs/, assets/               code, cấu hình (bản nộp: configs/docker.json), few-shot, fuse.json
├── saved_models/llm                           LLM đã merge (biến thể v5+v3: model gốc + adapters/{main,consult})
├── saved_models/asr, saved_models/asr2        PhoWhisper đã merge (+ lifemate_asr.json), gipformer
└── training_code/                             thư mục này; data/ = dữ liệu tổng hợp (lý do từng bước)
```
