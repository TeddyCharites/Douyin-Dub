# Douyin Dub Local — Hướng dẫn cho Claude Code

## Mục tiêu
Tool chạy local, dùng cá nhân để vọc: đưa vào **file video ngắn nói tiếng Trung** (clip Douyin, 15 giây – 3 phút) → nhận về video đã **lồng tiếng Việt**, **giữ nhạc nền gốc**, có **phụ đề tiếng Việt** (tuỳ chọn song ngữ), kèm file `.srt`.
Chạy hoàn toàn offline, không dùng API trả phí. Đầu vào là file video có sẵn trên máy (tool không có chức năng tải video từ nền tảng).

## Máy chạy
- Windows, i5-12400F, RTX 5060 8GB VRAM, RAM 32GB.
- RTX 50-series: driver NVIDIA mới nhất, PyTorch bản CUDA 12.8+.
- Đường dẫn dùng `pathlib`, chạy đúng trên Windows.
- Có thể tái sử dụng kinh nghiệm và code gọi model từ dự án Video-AI (https://github.com/TeddyCharites/Video-AI): hàng đợi GPU/CPU, mỗi tác vụ GPU chạy trong tiến trình con, ghi JSON an toàn.

## Pipeline
| Bước | Việc | Công cụ gợi ý | Chạy trên |
|---|---|---|---|
| 1 | Tách âm thanh từ video | ffmpeg | CPU |
| 2 | Tách giọng nói và nhạc nền | Demucs (hoặc model tách giọng tốt hơn nếu có) | GPU |
| 3 | Nhận dạng tiếng Trung + mốc thời gian từng câu | Whisper large-v3 qua faster-whisper; **so sánh với 1–2 model ASR chuyên tiếng Trung mới nhất** rồi chọn cái chính xác hơn | GPU |
| 4 | Phân biệt người nói (tuỳ chọn, khi có ≥2 người) | model diarization chạy local | GPU/CPU |
| 5 | Dịch Trung → Việt | Qwen3.6-35B-A3B qua Ollama (giống dự án Video-AI) | GPU + RAM |
| 6 | Lồng tiếng Việt | VieNeu-TTS v3 Turbo, ONNX trên CPU | CPU |
| 7 | Khớp thời gian | Python (xem mục dưới) | CPU |
| 8 | Trộn âm thanh + gắn phụ đề + xuất video | ffmpeg | CPU |

Trước khi chọn model cho bước 2, 3, 4: tìm phiên bản mới nhất, kiểm tra yêu cầu VRAM và chạy thử trên 1 clip mẫu. Báo mình kết quả so sánh trước khi chốt.

## Quy tắc VRAM
- Không giữ nhiều model trong VRAM cùng lúc. Mỗi bước GPU chạy trong tiến trình con riêng, xong là trả sạch VRAM.
- Ollama gọi với `keep_alive: 0`.
- Chỉ một tác vụ GPU tại một thời điểm.

## Dịch (bước 5) — quan trọng
- Dịch theo **cả đoạn** (gửi toàn bộ transcript kèm số thứ tự câu) để giữ ngữ cảnh, nhưng trả về **đúng từng câu theo số thứ tự** dưới dạng JSON.
- Mỗi câu kèm thời lượng gốc; yêu cầu bản dịch **tự nhiên, kiểu nói chuyện, đủ ngắn để đọc vừa thời lượng đó**.
- Giữ nguyên tên riêng; xưng hô phù hợp ngữ cảnh (anh/em, mình/bạn...).
- Kiểm tra JSON trả về, sai thì thử lại tối đa 2 lần.
- Tắt chế độ "thinking" của Qwen để nhanh hơn.

## Khớp thời gian (bước 7)
Với mỗi câu:
1. Tạo audio tiếng Việt, đo thời lượng.
2. Nếu dài hơn khung gốc: tăng tốc đọc tối đa ~1.25x (giữ cao độ, dùng `atempo` của ffmpeg). Vẫn dài hơn → nhờ LLM rút gọn câu và đọc lại. Cho phép lấn sang khoảng lặng của câu kế tiếp nếu có.
3. Nếu ngắn hơn: giữ nguyên, căn đầu câu theo mốc gốc.
4. Ghi log những câu phải rút gọn hoặc vẫn bị lệch để mình kiểm tra.

## Phụ đề
- Xuất `.srt` tiếng Việt (và tiếng Trung, song ngữ nếu chọn).
- Clip Douyin thường có **chữ Trung in sẵn** ở nửa dưới màn hình. Giai đoạn đầu: đặt phụ đề Việt trong **khung nền mờ/đặc** đủ lớn để che vùng chữ cũ; vị trí và chiều cao khung chỉnh được trong `config.yaml`.
- Giai đoạn sau (tuỳ chọn): tự phát hiện vùng chữ Trung bằng OCR để đặt khung che đúng chỗ. Chưa làm xoá chữ bằng inpainting.
- Video dọc 9:16: phụ đề cỡ lớn, tránh mép dưới và mép phải.

## Trộn âm thanh (bước 8)
- Giọng Việt + nhạc nền (đã tách ở bước 2), nhạc nền tự giảm âm lượng khi có giọng đọc (ducking).
- Tuỳ chọn giữ lại một chút giọng gốc tiếng Trung ở âm lượng rất nhỏ.
- Giữ nguyên hình ảnh, độ phân giải, tốc độ khung hình của video gốc.

## Giao diện
Web local (FastAPI + HTML/JS, hoặc Gradio cho bản đầu), các bước:
1. **Tải video lên** → xem trước.
2. **Bản chép lời & bản dịch** → bảng từng câu: thời gian, tiếng Trung, tiếng Việt (sửa được), người nói. Sửa câu nào thì chỉ câu đó cần lồng tiếng lại.
3. **Lồng tiếng** → chọn giọng (mỗi người nói một giọng), nghe thử từng câu, tạo lại từng câu.
4. **Xuất video** → chọn kiểu phụ đề (Việt / song ngữ / không), âm lượng nhạc nền → xem và tải `.mp4`, `.srt`.
Có chế độ **"Làm hết tự động"**: thả video vào, chạy thẳng tới bước 4.

## Lưu trữ
```
outputs/<project_id>/
├── source.mp4
├── project.json
├── segments.json      # từng câu: start, end, speaker, zh, vi, audio_status, ...
├── audio/ vocals.wav, background.wav
├── tts/ 001.wav, 002.wav, ...
├── final.mp4
└── final.vi.srt
```
Trạng thái mỗi câu: `missing | ready | stale | error`. Sửa bản dịch → audio câu đó `stale`.

## Thứ tự làm
1. `check_env.py`: Python, PyTorch nhận GPU, ffmpeg, Ollama.
2. Chạy thử từng model riêng lẻ trên 1 clip mẫu (script trong `tests/`, in thời gian và VRAM).
3. Pipeline dòng lệnh: `python run.py clip.mp4` → `final.mp4`.
4. Giao diện web.
5. Hoàn thiện: phân biệt người nói, OCR vùng chữ, song ngữ.

## Cách làm việc với mình
- Mình đang học để thoát khỏi "vibe coding". Sau mỗi giai đoạn, tóm tắt bằng tiếng Việt: đã làm gì, file nào làm gì, khái niệm mới cần biết.
- Nói kế hoạch trước khi sửa nhiều file. Gợi ý commit sau mỗi phần chạy được.
- Gặp lỗi: đọc log, giải thích nguyên nhân rồi mới sửa.
- Báo dung lượng trước khi tải model lớn.
- Cuối mỗi buổi: cập nhật mục "Tiến độ" ở cuối file này.

## Tiến độ
### 2026-10-21 — Bước 1 xong
- Dự án ở `D:\douyin-dub` (chuyển ra khỏi OneDrive để không đồng bộ venv/model; tránh đường dẫn có dấu tiếng Việt).
- Môi trường ảo `.venv` (Python 3.12.9). Chạy bằng `.venv\Scripts\python.exe`. Trên máy chỉ có `py`, không có lệnh `python` trong PATH.
- Đã cài: `torch 2.11.0+cu128`, `torchaudio 2.11.0+cu128`, `numpy`.
- **Lưu ý cài PyTorch:** chỉ dùng `--index-url https://download.pytorch.org/whl/cu128`, KHÔNG thêm `--extra-index-url` PyPI (pip sẽ chọn bản torch CPU mới hơn trên PyPI). Cài gói khác thì gọi pip riêng, không kèm torch.
- `check_env.py`: tất cả OK — RTX 5060 (sm_120, 8 GB), driver 610.88, ffmpeg 9.0.2, Ollama 0.40.6 có `hf.co/unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M`.
- Chưa có git repo.

**Tiếp theo:** Bước 2 — chạy thử từng model trên 1 clip mẫu (cần người dùng đưa clip). Báo dung lượng trước khi tải model.

### 2026-10-21 — Bước 2a: so sánh ASR (chưa chốt)
- Đã cài: `faster-whisper 1.2.1`, `qwen-asr 0.0.6` (ép `transformers==4.57.6`), các gói phụ của FireRedASR2S (KHÔNG cài theo `requirements.txt` của họ vì nó ép torch 2.1 cu118). Torch vẫn 2.11 cu128.
- Code FireRedASR2S clone ở `third_party/FireRedASR2S` (commit 4e7d9aa). Model ở `models/` (~14.3 GB).
- `tests/test_asr.py`: mỗi model chạy trong tiến trình con, đo VRAM bằng nvidia-smi. Kết quả ở `tests/out/asr_*.json`.
- Clip mẫu `tests/sample.mp4`: 180.8s, 1024x576 (ngang), thuyết minh anime (Kimetsu), có nhạc nền.
- Kết quả (audio gốc chưa tách nhạc): FireRedASR2-AED đúng chữ thường nhất, nhanh nhất (32.8s, ~5.7 GB) nhưng tên riêng không thống nhất. Qwen3-ASR-1.7B tên riêng thống nhất, sai vài cụm từ, VRAM chạm trần (torch báo 8.7 GB → tràn RAM chia sẻ). Whisper large-v3 tệ nhất: mốc thời gian vỡ thành khối 30s, sót 1 câu.
- Lỗi đã gặp: PyAV 19 bỏ tham số `metadata_errors` mà faster-whisper dùng → tự đọc wav bằng soundfile. CTranslate2 không thấy `cublas64_12.dll` → thêm `torch/lib` vào PATH (không phải `os.add_dll_directory`).

### 2026-10-21 — Bước 2b: so sánh tách giọng (chờ chốt)
- Đã cài `audio-separator 0.47.1` (+ `torchvision 0.26.0+cu128` cài TRƯỚC từ index cu128, vì torchvision trên PyPI kéo torch 2.14 CPU; + `audioread` vì librosa 1.0 bỏ nó). Model ở `models/separator/` (~1.9 GB).
- `tests/test_separation.py` → `tests/out/sep/<model>/vocals.wav, background.wav`. `test_asr.py` có thêm `--tag`.
- Tách 180s: Demucs htdemucs_ft 36.7s/1.6 GB; Mel-Band RoFormer (Kim) 16.3s/3.8 GB; BS-RoFormer 1297 129s/3.1 GB.
- ASR trên vocals đã tách: Qwen3 đỡ hẳn (2/10 → 6/10 cụm khó), FireRed gần như không đổi (9–10/10). Nhạc nền của cả 3 model: FireRed chỉ nghe ra "啦啦/哈哈" → không lọt lời thoại.
- Đề xuất: Mel-Band RoFormer + FireRedASR2-AED trên vocals. Chờ người dùng nghe background.wav để chốt.

### 2026-10-21 — Bước 2c: thử dịch Qwen3.6 qua Ollama
- Chốt (tạm): tách giọng Mel-Band RoFormer, ASR FireRedASR2-AED trên vocals.
- `tests/test_translate.py`: gộp vế FireRed thành câu (≤8s, khoảng lặng ≤0.3s) → 103 vế thành 33 câu; JSON schema của Ollama (`format`), `think: false`, `keep_alive: 0`; kiểm tra thiếu/trống câu, thử lại tối đa 2 lần; câu sót chữ Hán thì dịch lại riêng; `--glossary` bảng tên riêng.
- 180s clip: ~150s dịch, ~35 token/s, model 22 GB (5.2 GB trên VRAM, còn lại RAM). JSON đúng ngay lần 1.
- Không có bảng tên: tên riêng sai (Hán Việt "Hắc Tử Mô", nhầm sang "Genya", "Kibutsuji"). Có bảng tên (`tests/glossary_kimetsu.json`): đúng hết, nhưng vẫn nhầm vai ở câu mà ASR viết tên thành chữ đồng âm lạ (颜生/言生 = 严胜).
- ~10/33 câu vượt số âm tiết (5 âm tiết/giây, ước lượng) → xử lý ở bước 7. Cần hiệu chỉnh tốc độ đọc sau khi đo TTS.
- Còn mở: bảng tên riêng tự động (LLM trích tên ở lượt 1 → người dùng sửa → dịch).

### 2026-10-21 — Bước 2d: thử TTS VieNeu v3 Turbo → BƯỚC 2 XONG
- Đã cài `vieneu 3.8.3` (bản tối thiểu, KHÔNG cài extra `[cuda]` vì nó ép torch 2.8). Model tải vào kho chung `HF_HOME=D:\ai-models\huggingface` (biến môi trường có sẵn trên máy, ~0.74 GB).
- Phải ép `Vieneu(backend="onnx", device="cpu")`: mặc định SDK thấy torch+GPU sẽ tự chạy PyTorch trên GPU.
- `tests/test_tts.py`: đọc 33 câu của `translate.json`. fp32: RTF 0.40 (85s cho 213s audio); int8: RTF 0.29 (61s), nạp 7.5s. Tốc độ đọc thực tế ~3.9 âm tiết/s (giọng Thiện Minh), không phải 5.
- Hệ quả: audio Việt dài hơn khung gốc 17% tổng thể; 28/33 câu dài hơn khung, 11–12 câu dài quá x1.25 → phải rút gọn. Đã đổi `VI_SYLLABLES_PER_SEC` trong test_translate.py thành 4.3 (= 3.9 × 1.1).
- Model đã chọn cho pipeline: Mel-Band RoFormer → FireRedASR2-AED → Qwen3.6 (Ollama, có glossary) → VieNeu v3 Turbo ONNX CPU (int8 nếu nghe không méo).

**Tiếp theo:** Bước 3 — pipeline dòng lệnh `python run.py clip.mp4` → `final.mp4`.

### 2026-10-21 — Dịch lại với 4.3 âm tiết/s + bảng gộp
- `test_tts.py` in thêm câu gốc + bản dịch. Bản dịch cũ (5.0 âm tiết/s) lưu ở `tests/out/translate_5.0.json`.
- Ollama lỗi HTTP 500 hai lần: không cấp phát được 17.9 GB RAM (RAM trống ~18 GB, sát nút; llama.cpp còn bật prompt cache tới 8 GB). Sửa: `use_mmap: true` + lỗi HTTP thì đợi 10s rồi thử lại.
- Độ dài tốt lên rõ: audio 214s → 186s (clip 181s); câu phải rút gọn 12 → 3.
- NHƯNG chất lượng tụt: 27/33 câu sót chữ Hán phải dịch lại riêng; tên riêng quay về Hán Việt (Nghiêm Thắng, Nguyên Nhất, Hắc Tử Mưu) dù có glossary; văn phong Hán Việt ("hậu đại", "đỉnh phong"); #2/#3 lại chuyển ý. Giả thuyết (chưa kiểm chứng): lượt đầu để nguyên tên bằng chữ Hán, lượt sửa chữ Hán dịch riêng từng câu không đủ ngữ cảnh → ra Hán Việt. Cần lưu output thô lượt đầu để kiểm tra.

### 2026-10-21 — Sửa lỗi chữ Hán / tên riêng trong bước dịch
- `test_translate.py`: lưu bản thô lượt đầu (`vi_raw`), `--runs N` để đo dao động, lưu `translate_run<N>.json` (`translate.json` = lần cuối).
- Giả thuyết "chữ Hán lọt là tên riêng" SAI: lọt là từ thường (粗糙, 想着, 鬼王, 烂), có lúc chép nguyên vế tiếng Trung. Lần 27 câu trước đó là lần chạy xui; bình thường 2–5/33 câu.
- `fix_cjk` mới: (1) thay "chữ Hán (nghĩa Việt)" bằng nghĩa Việt bằng regex; (2) dịch lại kèm ngữ cảnh cả bài nhưng XOÁ bản dịch lỗi khỏi ngữ cảnh (để nguyên thì model chép lại y hệt); tối đa 2 lần.
- Kết quả 3 lần chạy: tên riêng đúng 100% (0 câu Hán Việt), #2/#3 đúng ý, còn chữ Hán sau sửa 1/0/0 câu (riêng "鬼王" rất lì). Mặt trái: câu sửa lại hay bị cắt bớt ý.
- Câu vẫn còn chữ Hán sau 2 lần sửa → pipeline phải đánh dấu `error` để người dùng sửa tay trong giao diện.
- TTS trên bản dịch đã sửa (run3): 167s audio cho clip 181s, chỉ 10/33 câu dài hơn khung, 2 câu cần rút gọn (#19, #22, đều ~x1.25). Còn lỗi nghĩa không bắt tự động được: #32/#33 "mặt trăng" thành "mặt trời", #10 "kẻ điên", #28 bị cắt quá ngắn → cần sửa tay ở giao diện.

### 2026-10-21 — Bước 3 + 4: pipeline dòng lệnh và giao diện web
- `run.py clip.mp4` → `outputs/<id>/final.mp4 + final.vi.srt + final.zh.srt + fit_log.txt`; `--resume <id>` chạy tiếp. Clip mẫu 181s: chạy hết 391s (tách 25s, ASR 50s, dịch 144s, TTS+khớp 154s, xuất 17s).
- Package `dub/`: config, project (JSON ghi tạm rồi os.replace), media (ffmpeg), gpu_worker (tiến trình con: tách giọng + ASR), translate, tts, timing, render, pipeline (GPU_LOCK). `config.yaml` chứa mọi thông số.
- Khớp thời gian: khung = tới khi câu sau bắt đầu; atempo ≤1.25; còn dài → gom các câu, rút gọn 1 lượt LLM, đọc lại (≤2 vòng); câu người dùng sửa tay không tự rút gọn.
- Sửa sau lần chạy đầu: bước dịch kiểm tra thêm câu quá ngắn (<40% → dịch thiếu ý, bắt được câu #9) và quá dài (>120%); rút gọn cho model khoảng [85%, 100%] (trước đó cắt quá tay, 14 câu → còn 3); khung che mặc định đặc (opacity 1.0) vì 0.85 vẫn lộ chữ Trung.
- Kết quả cuối trên clip mẫu: 0 câu lệch, 3 câu rút gọn, 13 câu tăng tốc ≥x1.05 (tối đa x1.24); giọng Việt 175s / clip 181s.
- Web `app.py` (FastAPI + web/): 4 bước + "Làm hết tự động", một hàng đợi tác vụ (mỗi lúc một tác vụ), bảng sửa câu → câu đó stale → chỉ đọc lại câu đó; đổi giọng một người nói → các câu của người đó stale; xem trước khung che trên video; mở dự án qua `#<id>/<bước>`. Đã thử qua API + chụp màn hình Edge headless.
- Chưa làm: phân biệt người nói, OCR vùng chữ, tự trích bảng tên riêng; tests/ còn bản sao logic dịch cũ (test_translate.py).

### 2026-10-21 — Bước 5: hoàn thiện (người nói, OCR, bảng tên tự động, khớp theo chuỗi) → DỰ ÁN XONG
- Cài thêm (~150 MB): `sherpa-onnx 1.13.8`, `rapidocr 3.10.0` (+opencv-headless), `pypinyin`; model CAM++ tiếng Trung `models/speaker/campplus_zh_16k.onnx` (28 MB, tải từ GitHub k2-fsa, không cần token HF).
- `dub/diarize.py`: vector giọng CAM++ mỗi câu + gom nhóm agglomerative (cosine, ngưỡng 0.45). `tests/test_diarize.py`: clip mẫu ra 1 người; đoạn ghép 3 giọng VieNeu tách đúng 100% với ngưỡng 0.3–0.5 (0.6 gộp nhầm 2 giọng nam). Người nói mới được gán sẵn giọng khác nhau (`tts.speaker_voices`). Chưa thử trên clip Douyin thật nhiều người nói.
- `dub/ocr.py`: tìm dải phụ đề Trung = vùng dọc có nhiều dòng chữ Hán nằm giữa nhất (percentile bị chữ trong cảnh kéo giãn) → tự đặt khung che. Đọc chữ trên màn hình cho từng câu (`zh_ocr`) → gửi kèm LLM: phụ đề in sẵn đúng chính tả hơn ASR (黑死牟 vs ASR 黑死谋). Lọc dòng của câu bên cạnh (trùng chữ với ASR <50%). Tăng tốc: ffmpeg giải mã 1 lượt thay cho seek OpenCV; RapidOCR `Det.limit_type=max` (mặc định phóng dải 1024x106 lên ~7000x736, chậm 10x). OCR cả clip: ~42s.
- Bảng tên tự động: LLM xác định tác phẩm rồi liệt kê tên + kiểu viết sai. Model hay sai kiến thức → 2 lớp kiểm: kiểu viết sai phải đọc giống tên gốc (pypinyin, >½ âm tiết); hai tên khác nhau cùng một cách dịch (vd. 继国严胜 & 继国缘一 đều "Yoriichi") → bỏ cả nhóm. Vẫn còn sai không bắt được (时透无一郎 → "Giyu Tomioka") → người dùng duyệt bảng tên trong giao diện.
- `dub/timing.py` xếp theo chuỗi: câu sau được bắt đầu trễ ≤1s nếu nó còn dư khung, câu trước mượn chỗ. Lỗi đã gặp: câu overflow đẩy câu sau → dây chuyền overflow (18 câu bị rút gọn) → sửa: không đẩy quá hạn chót.
- Bước dịch: câu ≤10s (8s hay cắt giữa ý); ví dụ mẫu phải NGẮN hơn bản cứng (ví dụ cũ dạy model nói dài: tỉ lệ âm tiết/giới hạn 0.97 → 1.10); `syllables_per_sec` 4.0; kiểm câu quá dài >115% (1 vòng), quá ngắn <40%, sót chữ Hán (2 vòng, cuối cùng xoá + cảnh báo); rút gọn cắt <60% mục tiêu thì bỏ, giữ bản cũ.
- LLM dao động lớn giữa các lần chạy (tỉ lệ độ dài 0.92–1.14 cùng cấu hình): luôn đo nhiều lần trước khi kết luận.
- Web: bước Người nói / OCR, giọng theo từng người nói, chữ OCR dưới câu Trung, nút "Tự phát hiện vùng chữ", cảnh báo "xem lại"; server khởi động xoá `job` kẹt; `run_all` xoá `job` cả khi lỗi.
- Kết quả cuối (clip mẫu 181s): chạy hết ~9 phút; tên Kokushibo/Yoriichi nhất quán, hai anh em không bị gộp; 0–3 câu "lệch" ≤1.2s; ~10 câu rút gọn vừa phải.
- `README.md` hướng dẫn cài và dùng. `tests/test_translate.py` giờ gọi thẳng `dub/translate.py` (bỏ bản sao logic cũ).
- Còn mở: thử clip nhiều người nói thật; người dùng nghe chọn giọng / int8 vs fp32 / độ ducking; xoá chữ bằng inpainting (CLAUDE.md: chưa làm).

### 2026-10-21 — Đổi kiểu phụ đề
- `subtitle.style` trong config.yaml (và chọn được ở bước 4 web): `outline` (mặc định) | `box` | `blur`. Chữ trắng viền đen 2px + bóng 1px (ASS BorderStyle=1, Outline=2, Shadow=1); cỡ chữ 0.055 → 0.04 (−27%).
- outline đè lên chữ Trung vàng cỡ lớn thì cả hai khó đọc → mặc định đặt NGAY TRÊN dải chữ Trung (`outline_position: above`, đổi `over` để đè). blur: ffmpeg crop dải → boxblur → overlay, rồi vẽ phụ đề; bán kính ≤ 1/4 chiều cao dải (giới hạn của mặt phẳng màu yuv420).
- Câu dài tách thành nhiều đoạn phụ đề (tối đa 2 dòng × 40 ký tự): chia đều, ưu tiên ngắt sau dấu câu, mỗi đoạn ≤ 90% của 80 ký tự (đoạn đúng 80 ký tự không chia nổi 2 dòng ≤40), thời gian chia theo số ký tự; song ngữ cắt câu Trung theo tỉ lệ, dịch về dấu câu gần nhất. Áp dụng cả cho .srt. Clip mẫu: 30 câu → 60 đoạn, dòng dài nhất 40, 52/60 đoạn kết thúc ở dấu câu.
- So sánh: `outputs/20261021-190624_sample/final_outline.mp4`, `final_blur.mp4`, `final_box.mp4`; ảnh ghép `tests/out/compare_3.0.png`, `compare_21.0.png`.
- Kiểu phụ đề mặc định chuyển sang blur (người dùng chọn sau khi so sánh 3 kiểu).
