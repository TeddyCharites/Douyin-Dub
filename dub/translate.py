"""Dịch Trung -> Việt bằng Qwen3.6 qua Ollama (bước 5) và rút gọn câu (bước 7).

Quy tắc (CLAUDE.md): dịch cả bài một lần để giữ ngữ cảnh, trả JSON đúng từng câu
theo id, mỗi câu kèm thời lượng / số âm tiết tối đa; kiểm tra JSON, thử lại tối đa
2 lần; tắt thinking; keep_alive 0. Các quyết định về prompt xem tests/test_translate.py.
"""
import json
import re
import time
import urllib.error
import urllib.request

from .config import CONFIG

CJK = re.compile(r"[一-鿿]")
# Qwen hay viết "粗糙 (thô ráp)": chữ Hán kèm nghĩa Việt trong ngoặc -> giữ phần nghĩa Việt
CJK_GLOSS = re.compile(r"[一-鿿]+\s*[(（]([^)）]+)[)）]")
SENT_END = set("。！？!?；;…")
# Ngưỡng coi bản dịch là lỗi, so với max_syllables của câu
# Dài hơn 115%: max_syllables đã chừa ~10% cho tăng tốc, dài thêm 15% nữa là cần ~1.27x,
# quá max_tempo. Sửa ở đây (cả bài làm ngữ cảnh, không phải đọc lại TTS) rẻ hơn rút gọn sau.
TOO_LONG = 1.15
TOO_SHORT = 0.4   # ngắn hơn 40%: nhiều khả năng dịch thiếu ý
SHORTEN_MIN = 0.85  # rút gọn: không ngắn hơn 85% mức tối đa
OVERCUT = 0.6       # bản rút gọn ngắn hơn 60% mức tối đa = cắt quá tay, bỏ

SYSTEM_PROMPT = """Bạn là người dịch lời thuyết minh video ngắn từ tiếng Trung sang tiếng Việt để lồng tiếng.

Quy tắc:
- Đọc cả bài để hiểu ngữ cảnh, nhưng dịch TỪNG câu theo đúng "id", không gộp, không tách, không bỏ câu nào.
- Nội dung bản dịch của mỗi id phải tương ứng đúng câu gốc của id đó: KHÔNG chuyển ý sang câu trước/sau, kể cả khi câu gốc bị cắt giữa chừng.
- Chỉ dùng tiếng Việt: tuyệt đối không để sót chữ Hán trong bản dịch.
- Văn nói tự nhiên, kiểu người Việt kể chuyện trên mạng; không dịch word-by-word.
- Mỗi câu có "max_syllables": bản dịch KHÔNG được dài hơn số âm tiết (số tiếng) đó, vì phải đọc vừa thời lượng gốc. Câu dài thì lược ý phụ, giữ ý chính.
- Bản chép lời do máy nhận dạng nên có thể sai chữ, nhất là tên riêng (cùng một người có thể bị viết thành nhiều kiểu). Hãy suy ra tên đúng từ ngữ cảnh và dùng thống nhất cả bài.
- Giữ tên riêng; với nhân vật anime/phim Nhật, dùng tên quen thuộc với khán giả Việt (thường là tên tiếng Nhật viết La-tinh).
- Xưng hô hợp ngữ cảnh (anh/em, mình/các bạn, hắn, gã...). Câu có trường "speaker" (A, B...): mỗi chữ cái là một người nói khác nhau, chọn xưng hô theo quan hệ giữa họ.
- Câu có trường "zh_tren_man_hinh": đó là phụ đề in sẵn trên video, thường ĐÚNG CHÍNH TẢ hơn "zh" (do máy nghe). Khi hai bên khác nhau (nhất là tên riêng), tin "zh_tren_man_hinh". Nhưng nó có thể lẫn vài chữ của câu kế bên ở đầu/cuối: chỉ dịch phần nội dung ứng với "zh".
- Dùng từ thuần Việt, dễ nghe, NGẮN GỌN; tránh từ Hán Việt khó hiểu, không thêm từ đệm. Ví dụ (cứng -> tự nhiên, ngắn hơn):
  "Hắn quả thực là bậc kỳ tài hiếm có trong thiên hạ." -> "Hắn đúng là thiên tài hiếm có."
  "Phải biết rằng chuyện này hoàn toàn không hề đơn giản." -> "Chuyện này đâu có đơn giản."
  "Muốn dựa vào bản thân để đạt tới đỉnh phong võ đạo." -> "Muốn tự leo lên đỉnh võ thuật."
- Chỉ trả về JSON theo schema, không giải thích."""

GLOSSARY_PROMPT = """

Bảng tên riêng (BẮT BUỘC dùng đúng cách dịch này). Bản chép lời có thể viết sai các tên này
thành chữ đồng âm/gần âm, hãy nhận ra và vẫn dùng đúng tên:
"""

NAMES_PROMPT = """Bạn chuẩn bị bảng tên riêng để dịch một video thuyết minh tiếng Trung sang tiếng Việt.
Bước 1 - "work": video nói về tác phẩm / chủ đề gì (tên tác phẩm và nước gốc), nếu nhận ra.
Bước 2 - "names": liệt kê tên riêng (nhân vật, tổ chức, địa danh, chiêu thức, danh hiệu...). Mỗi người / vật là MỘT mục riêng: tuyệt đối không gộp hai nhân vật khác nhau (vd. hai anh em cùng họ) vào một mục.
- "zh": cách viết ĐÚNG của tên. Ưu tiên chữ trong "zh_tren_man_hinh" (phụ đề in sẵn, đúng chính tả hơn).
- "variants": các kiểu viết sai của CHÍNH tên đó trong bản "zh" (máy nghe nhầm thành chữ đồng âm), có thể rỗng.
- "vi": cách khán giả Việt quen gọi, bằng tiếng Việt hoặc tên gốc La-tinh, KHÔNG dùng tiếng Anh.
  * Tác phẩm Nhật (anime, manga, game): tên người PHẢI là tên gốc tiếng Nhật viết La-tinh như bản chính thức, TUYỆT ĐỐI KHÔNG dùng âm Hán Việt. Ví dụ: 鸣人 -> Naruto, 佐助 -> Sasuke, 路飞 -> Luffy, 索隆 -> Zoro. Tổ chức / địa danh thì dịch nghĩa như bản Việt hoá: 木叶村 -> làng Lá, 晓组织 -> Akatsuki.
  * Tác phẩm Trung (phim, tiểu thuyết): dùng âm Hán Việt quen thuộc (vd. 孙悟空 -> Tôn Ngộ Không).
- Không liệt kê từ thông thường hay cụm mô tả (vd. 天选之子). Tối đa 30 dòng. Chỉ trả về JSON theo schema."""

NAMES_SCHEMA = {
    "type": "object",
    "properties": {
        "work": {"type": "string"},
        "names": {"type": "array", "items": {
            "type": "object",
            "properties": {"zh": {"type": "string"},
                           "variants": {"type": "array", "items": {"type": "string"}},
                           "vi": {"type": "string"}},
            "required": ["zh", "variants", "vi"]}}},
    "required": ["work", "names"],
}

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"translations": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "integer"}, "vi": {"type": "string"}},
        "required": ["id", "vi"]}}},
    "required": ["translations"],
}


# Tên dịch ra tiếng Anh (model hay làm dù đã dặn) -> bỏ, để bước dịch tự xử lý
ENGLISH_WORDS = {"the", "of", "corps", "castle", "family", "clan", "city", "village", "house", "demon", "slayer"}


def sounds_like(variant, name):
    """Kiểu viết sai do ASR nghe nhầm phải ĐỌC gần giống tên gốc: hơn nửa số âm tiết
    (pinyin, bỏ thanh điệu) trùng với một đoạn cùng độ dài trong tên gốc.
    Vd. 黑死猫~黑死牟 (2/3), 原意~继国缘一 (yuan yi); 严胜≠继国缘一 -> người khác."""
    from pypinyin import lazy_pinyin
    a, b = lazy_pinyin(variant), lazy_pinyin(name)
    if len(a) > len(b):
        a, b = b, a
    best = max(sum(x == y for x, y in zip(a, b[i:i + len(a)])) for i in range(len(b) - len(a) + 1))
    return best > len(a) / 2


def drop_conflicts(glossary, log=print):
    """Hai tên Trung KHÁC NHAU (đọc không giống nhau) mà cùng một cách dịch -> model đã
    nhầm người (vd. 继国严胜 và 继国缘一 đều thành "Yoriichi": hai anh em thành một).
    Không biết bên nào đúng nên bỏ cả nhóm, để bước dịch tự xử lý."""
    by_vi = {}
    for zh, vi in glossary.items():
        by_vi.setdefault(vi, []).append(zh)
    bad = set()
    for vi, names in by_vi.items():
        if any(not sounds_like(a, b) and not sounds_like(b, a) and a not in b and b not in a
               for i, a in enumerate(names) for b in names[i + 1:]):
            bad.add(vi)
    if bad:
        log(f"Bỏ cách dịch bị nhiều tên khác nhau dùng chung (model nhầm người): "
            f"{ {vi: by_vi[vi] for vi in bad} }")
    return {zh: vi for zh, vi in glossary.items() if vi not in bad}


def count_syllables(vi):
    return len(vi.split())


def merge_clauses(clauses, max_dur, max_gap):
    """Gộp các vế FireRed (1-3 giây) thành câu để lồng tiếng: chưa gặp dấu kết thúc
    câu, khoảng lặng giữa hai vế <= max_gap, câu gộp không dài quá max_dur."""
    out = []
    for c in clauses:
        prev = out[-1] if out else None
        if (prev and prev["text"][-1:] not in SENT_END
                and c["start"] - prev["end"] <= max_gap
                and c["end"] - prev["start"] <= max_dur):
            prev["end"] = c["end"]
            prev["text"] += c["text"]
        else:
            out.append(dict(c))
    return out


def max_syllables(duration):
    return max(3, int(duration * CONFIG["translate"]["syllables_per_sec"]))


class Translator:
    def __init__(self, glossary=None, log=print):
        self.cfg = CONFIG["translate"]
        self.log = log
        self.system = SYSTEM_PROMPT
        if glossary:
            self.system += GLOSSARY_PROMPT + "\n".join(f"- {zh}: {vi}" for zh, vi in glossary.items())

    # -- gọi Ollama
    def _chat(self, user, schema=RESPONSE_SCHEMA, system=None, temperature=None):
        payload = {
            "model": self.cfg["model"],
            "messages": [{"role": "system", "content": system or self.system},
                         {"role": "user", "content": user}],
            "format": schema,
            "think": False,      # tắt chế độ suy nghĩ của Qwen cho nhanh
            "keep_alive": 0,     # xong là trả VRAM
            "stream": False,
            # use_mmap: phần model ở RAM đọc từ file theo nhu cầu, tránh lỗi 500 khi RAM trống sát ~18 GB
            "options": {"temperature": self.cfg["temperature"] if temperature is None else temperature,
                        "num_ctx": self.cfg["num_ctx"],
                        "use_mmap": True},
        }
        req = urllib.request.Request(f"{self.cfg['ollama_url']}/api/chat",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.load(r)

    @staticmethod
    def _parse(resp, expected):
        """{id: vi}; raise ValueError nếu JSON sai / thiếu câu / câu trống."""
        try:
            data = json.loads(resp["message"]["content"])
            vi = {t["id"]: t["vi"].strip() for t in data["translations"]}
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise ValueError(f"JSON hỏng: {e}") from e
        missing = expected - vi.keys()
        empty = [i for i in expected & vi.keys() if not vi[i]]
        if missing or empty:
            raise ValueError(f"thiếu câu {sorted(missing)}, câu trống {empty}")
        return {i: vi[i] for i in expected}

    def _ask(self, user, expected):
        """Gọi Ollama, kiểm tra JSON; sai thì thử lại tối đa max_retries lần."""
        last = None
        for attempt in range(1, self.cfg["max_retries"] + 2):
            try:
                return self._parse(self._chat(user), expected)
            except urllib.error.HTTPError as e:
                # Thường là Ollama nạp model thất bại vì thiếu RAM: đợi rồi thử lại
                last = f"Ollama lỗi HTTP {e.code} (xem server.log của Ollama)"
                time.sleep(10)
            except ValueError as e:
                last = str(e)
            self.log(f"Dịch lần {attempt} lỗi: {last}")
        raise RuntimeError(f"Dịch thất bại: {last}")

    # -- dịch cả bài
    def translate(self, segments):
        """segments: [{id, zh, start, end}] -> {id: vi}. Câu lỗi thì sửa (fix_bad)."""
        items = self.items(segments)
        vi = self._ask("Dịch các câu sau:\n" + json.dumps(items, ensure_ascii=False, indent=1),
                       {it["id"] for it in items})
        self.fix_bad(items, vi)
        return vi

    @staticmethod
    def items(segments):
        """Dữ liệu từng câu gửi cho LLM. Chỉ kèm "speaker" khi có từ 2 người nói."""
        multi = len({s.get("speaker") or "A" for s in segments}) > 1
        out = []
        for s in segments:
            it = {"id": s["id"], "zh": s["zh"]}
            if s.get("zh_ocr"):
                it["zh_tren_man_hinh"] = s["zh_ocr"]
            if multi:
                it["speaker"] = s.get("speaker") or "A"
            it.update(duration_s=round(s["end"] - s["start"], 1),
                      max_syllables=max_syllables(s["end"] - s["start"]))
            out.append(it)
        return out

    def extract_names(self, segments):
        """Một lượt LLM liệt kê tên riêng -> {tên Trung: cách dịch} làm bảng tên tự động."""
        data = [{k: v for k, v in it.items() if k in ("zh", "zh_tren_man_hinh")}
                for it in self.items(segments)]
        resp = self._chat("Bài thuyết minh:\n" + json.dumps(data, ensure_ascii=False, indent=1),
                          schema=NAMES_SCHEMA, system=NAMES_PROMPT, temperature=0.1)
        try:
            data = json.loads(resp["message"]["content"])
            names = data["names"]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.log(f"Không trích được tên riêng: {e}")
            return {}
        self.log(f"Tác phẩm nhận ra: {data.get('work')}")
        glossary, rejected = {}, []
        for n in names[:30]:
            vi = (n.get("vi") or "").strip()
            if not vi or CJK.search(vi) or ENGLISH_WORDS & set(vi.lower().split()):
                continue  # để bước dịch tự xử lý thay vì ép một cách dịch tiếng Anh
            # phòng khi model vẫn nhét nhiều kiểu viết vào một chuỗi
            split = lambda s: [x.strip() for x in re.split(r"[\n,，、/;；]+", s or "") if x.strip()]
            canon = split(n.get("zh"))
            if not canon:
                continue
            for zh in canon:
                glossary.setdefault(zh, vi)
            for var in (v for raw in n.get("variants") or [] for v in split(raw)):
                if sounds_like(var, canon[0]):
                    glossary.setdefault(var, vi)
                else:
                    rejected.append(f"{var}≠{canon[0]}")
        if rejected:
            self.log(f"Bỏ các 'kiểu viết sai' đọc không giống tên gốc (nhiều khả năng là người khác): {rejected}")
        return drop_conflicts(glossary, self.log)

    @staticmethod
    def problems(it, text, check_length=True):
        """Lý do một bản dịch cần làm lại, hoặc None nếu ổn."""
        if CJK.search(text):
            return "sót chữ Hán"
        if check_length:
            n, mx = count_syllables(text), it["max_syllables"]
            if n > mx * TOO_LONG and n - mx >= 2:  # câu rất ngắn: dư 1 âm tiết không đáng dịch lại
                return f"quá dài ({n}/{mx} âm tiết)"
            if n < mx * TOO_SHORT and it.get("duration_s", 0) >= 2.5:
                return f"quá ngắn ({n}/{mx} âm tiết), có thể dịch thiếu ý"
        return None

    def fix_bad(self, items, vi, check_length=True):
        """Sửa câu lỗi: sót chữ Hán, quá dài, quá ngắn (dịch thiếu ý).
        1. Thay "chữ Hán (nghĩa Việt)" bằng nghĩa Việt (không tốn lượt gọi LLM).
        2. Câu còn lỗi: dịch lại trong MỘT lượt, kèm CẢ BÀI làm ngữ cảnh (giữ tên riêng
           đúng) nhưng XOÁ bản dịch lỗi khỏi ngữ cảnh (để nguyên thì model chép lại y hệt)."""
        for i in vi:
            if CJK.search(vi[i]):
                vi[i] = re.sub(r" {2,}", " ", CJK_GLOSS.sub(r" \1", vi[i])).strip()
        for attempt in range(1, self.cfg["max_retries"] + 1):
            # Lỗi độ dài chỉ sửa 1 vòng: model thường không rút được thêm ở vòng 2, tốn ~1 phút vô ích
            length = check_length and attempt == 1
            bad = {it["id"]: why for it in items
                   if (why := self.problems(it, vi[it["id"]], length))}
            if not bad:
                return
            self.log(f"Câu cần dịch lại (lần {attempt}): {bad}")
            context = [{**it, "vi_hien_tai": "" if it["id"] in bad else vi[it["id"]]} for it in items]
            user = (f"Đây là cả bài cùng bản dịch hiện tại. Các câu id {sorted(bad)} chưa có bản dịch. "
                    "Chỉ dịch các câu đó, hoàn toàn bằng tiếng Việt (không một chữ Hán nào): dịch ĐỦ ý "
                    "của câu gốc, không vượt max_syllables, văn nói tự nhiên (không dùng từ Hán Việt "
                    "khó hiểu), tên riêng theo đúng bảng tên và thống nhất với các câu khác. "
                    "Chỉ trả về các id đó.\n" + json.dumps(context, ensure_ascii=False, indent=1))
            try:
                vi.update(self._ask(user, set(bad)))
            except RuntimeError as e:
                self.log(f"Dịch lại câu lỗi thất bại: {e}")

    # -- rút gọn (bước 7)
    def shorten(self, segments, targets):
        """Rút gọn nhiều câu trong MỘT lần gọi (mỗi lần gọi phải nạp lại model 22 GB).
        targets: {id: số âm tiết tối đa}. Cho model một KHOẢNG (min..max): chỉ cho mức
        tối đa thì model hay cắt quá tay, mất ý và để lại khoảng lặng dài. Trả về {id: vi mới}."""
        ranges = {i: [max(2, int(t * SHORTEN_MIN)), t] for i, t in targets.items()}
        context = [{"id": s["id"], "zh": s["zh"], "vi_hien_tai": s["vi"],
                    **({"so_am_tiet_moi": ranges[s["id"]]} if s["id"] in ranges else {})}
                   for s in segments]
        user = (f"Các câu id {sorted(targets)} đọc lên hơi dài so với thời lượng video. "
                "Viết lại ngắn hơn MỘT CHÚT, độ dài nằm trong khoảng 'so_am_tiet_moi' [ít nhất, nhiều nhất] "
                "âm tiết: giữ đủ ý chính, chỉ bỏ từ đệm / từ lặp / ý phụ, giọng văn nói tự nhiên, "
                "giữ tên riêng. Các câu khác chỉ để tham khảo ngữ cảnh. Chỉ trả về các id cần rút gọn.\n"
                + json.dumps(context, ensure_ascii=False, indent=1))
        vi = self._ask(user, set(targets))
        items = [{"id": i, "zh": s["zh"], "max_syllables": targets[i]}
                 for s in segments for i in [s["id"]] if i in targets]
        self.fix_bad(items, vi, check_length=False)
        # Model đôi khi cắt quá tay (vd. 41 -> 15 âm tiết dù được dặn tối thiểu 85%): mất nửa
        # ý còn tệ hơn đọc nhanh một chút -> bỏ bản đó, giữ bản cũ (pipeline sẽ ghi "lệch").
        too_short = {i: count_syllables(v) for i, v in vi.items() if count_syllables(v) < targets[i] * OVERCUT}
        if too_short:
            self.log(f"Bỏ bản rút gọn quá tay (số âm tiết còn lại): {too_short}")
        return {i: v for i, v in vi.items() if i not in too_short}
