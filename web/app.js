// Giao diện Douyin Dub Local: 4 bước, hỏi trạng thái dự án định kỳ khi đang chạy tác vụ.
const $ = (sel) => document.querySelector(sel);
const state = { config: null, id: null, data: null, pollTimer: null, segsKey: "", goToExport: false };

const STEP_NAMES = { prepare: "Chuẩn bị", separate: "Tách giọng", asr: "Nhận dạng", diarize: "Người nói", ocr: "Đọc chữ (OCR)", translate: "Dịch", tts: "Lồng tiếng", render: "Xuất video" };
const STATUS_VI = { missing: "chưa có", ready: "sẵn sàng", stale: "cũ", error: "lỗi", pending: "chưa chạy", running: "đang chạy", done: "xong" };
const FIT_VI = { ok: "vừa khung", borrow: "lấn sang câu sau", shifted: "đọc trễ", tempo: "tăng tốc", overflow: "LỆCH" };

async function api(method, url, body) {
  const opts = { method, headers: {} };
  if (body instanceof FormData) opts.body = body;
  else if (body !== undefined) { opts.body = JSON.stringify(body); opts.headers["Content-Type"] = "application/json"; }
  const res = await fetch(url, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Lỗi ${res.status}`);
  return data;
}

function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) e.setAttribute(k, v);
  }
  for (const c of children.flat()) if (c !== null && c !== undefined) e.append(c instanceof Node ? c : String(c));
  return e;
}
const badge = (cls, text) => el("span", { class: `badge b-${cls}` }, text);
const fmtTime = (t) => `${String(Math.floor(t / 60)).padStart(2, "0")}:${(t % 60).toFixed(1).padStart(4, "0")}`;
const fileUrl = (path, v = "") => `/files/${state.id}/${path}?v=${encodeURIComponent(v)}`;

function showError(msg) { const e = $("#error"); e.textContent = msg || ""; e.hidden = !msg; }

// ---------------------------------------------------------------- điều hướng 4 bước
function goStep(n) {
  if (state.id) history.replaceState(null, "", `#${state.id}/${n}`);  // tải lại trang vẫn giữ dự án + bước
  document.querySelectorAll("nav button").forEach((b) => b.classList.toggle("active", b.dataset.step == n));
  document.querySelectorAll("main section").forEach((s) => (s.hidden = s.dataset.step != n));
  if (n == 4) placeOverlay();
}
document.querySelectorAll("nav button").forEach((b) => b.addEventListener("click", () => goStep(b.dataset.step)));

// ---------------------------------------------------------------- bước 1: tải lên
const fileInput = $("#file");
function pickFile(files) {
  if (!files || !files.length) return;
  fileInput.files = files;
  $("#file-name").textContent = files[0].name;
  $("#upload").disabled = false;
}
fileInput.addEventListener("change", () => pickFile(fileInput.files));
const drop = $("#drop");
["dragover", "dragenter"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove("over")));
drop.addEventListener("drop", (e) => { e.preventDefault(); pickFile(e.dataTransfer.files); });

$("#upload").addEventListener("click", async () => {
  const fd = new FormData();
  fd.append("file", fileInput.files[0]);
  fd.append("auto", $("#auto").checked);
  $("#upload").disabled = true;
  $("#upload").textContent = "Đang tải lên...";
  try {
    const { id } = await api("POST", "/api/projects", fd);
    state.goToExport = $("#auto").checked;
    await loadProjects();
    await openProject(id);
  } catch (e) { showError(e.message); }
  $("#upload").textContent = "Tải lên";
  $("#upload").disabled = false;
});

async function loadProjects() {
  const list = await api("GET", "/api/projects");
  const ul = $("#projects");
  ul.replaceChildren(...list.slice(0, 15).map((p) => {
    const done = p.steps.render.status === "done";
    return el("li", { class: p.id === state.id ? "current" : "", onclick: () => openProject(p.id) },
      el("span", {}, p.source_name, el("br"), el("span", { class: "muted" }, new Date(p.created * 1000).toLocaleString("vi-VN"))),
      badge(done ? "done" : p.job ? "running" : "pending", done ? "đã xuất" : p.job ? "đang chạy" : "chưa xong"));
  }));
}

async function openProject(id) {
  state.id = id;
  state.segsKey = "";
  await refresh();
  const v = $("#source-video"); v.src = fileUrl("source.mp4"); v.hidden = false;
  $("#box-video").src = fileUrl("source.mp4");
  // ảnh bìa: hiện hình ngay, không phải đợi video tải
  for (const sel of ["#source-video", "#box-video", "#final-video"]) $(sel).poster = fileUrl("thumb.jpg");
  fillSettings();
  loadProjects();
}

// ---------------------------------------------------------------- hỏi trạng thái
async function refresh() {
  if (!state.id) return;
  try { state.data = await api("GET", `/api/projects/${state.id}`); }
  catch (e) { showError(e.message); return; }
  render();
  const running = !!state.data.job;
  if (running && !state.pollTimer) state.pollTimer = setInterval(refresh, 1500);
  if (!running && state.pollTimer) {
    clearInterval(state.pollTimer); state.pollTimer = null;
    loadProjects();
    fillSettings();  // tác vụ vừa xong có thể đã đổi khung che (OCR) / bảng tên (tự trích)
    if (state.goToExport && state.data.files["final.mp4"]) { state.goToExport = false; goStep(4); }
  }
}

function render() {
  const d = state.data;
  $("#project-title").textContent = `${d.source_name}` + (d.video ? ` — ${d.video.width}×${d.video.height}, ${d.video.duration.toFixed(1)}s` : "");
  $("#project-steps").replaceChildren(...Object.entries(d.steps).map(([k, s]) =>
    badge(s.status, `${STEP_NAMES[k]}: ${STATUS_VI[s.status]}${s.seconds ? ` (${s.seconds}s)` : ""}`)));

  const st = $("#status");
  st.hidden = !d.job;
  if (d.job) {
    $("#status-text").textContent = `${d.job.message || d.job.name}` + (d.queue_size ? ` (còn ${d.queue_size} tác vụ chờ)` : "");
    const pr = $("#status-progress");
    pr.hidden = !d.job.total;
    if (d.job.total) { pr.max = d.job.total; pr.value = d.job.done || 0; }
  }
  showError(d.last_error);
  document.querySelectorAll("button.primary:not(#upload), #run-translate, #save-glossary, #detect-box").forEach((b) => (b.disabled = !!d.job));

  // Chỉ vẽ lại bảng khi dữ liệu câu đổi và người dùng không đang gõ trong bảng
  const key = JSON.stringify(d.segments) + JSON.stringify(d.settings.voices);
  const typing = document.activeElement && document.activeElement.closest("table.segs");
  if (key !== state.segsKey && !typing) { state.segsKey = key; renderSegs(); renderTts(); }
  renderReport();
  renderDownloads();
}

// ---------------------------------------------------------------- bước 2: bảng chép lời & dịch
function renderSegs() {
  const segs = state.data.segments;
  if (!segs.length) { $("#seg-body").replaceChildren(el("tr", {}, el("td", { colspan: 6, class: "muted" }, "Chưa có bản chép lời. Bấm \"Nhận dạng & dịch\"."))); return; }
  $("#seg-body").replaceChildren(...segs.map((s) => el("tr", {},
    el("td", {}, s.id),
    el("td", { class: "time" }, `${fmtTime(s.start)} → ${fmtTime(s.end)}`, el("br"), el("span", { class: "muted" }, `${(s.end - s.start).toFixed(1)}s`)),
    el("td", { class: "zh" }, s.zh,
      s.zh_ocr && s.zh_ocr !== s.zh ? el("div", { class: "muted", title: "Chữ in sẵn trên video (OCR), dùng làm gợi ý khi dịch" }, `Trên màn hình: ${s.zh_ocr}`) : null),
    el("td", { class: "vi" }, el("textarea", { rows: 2, "data-id": s.id, onchange: (e) => saveSeg(s.id, { vi: e.target.value }) }, s.vi),
      s.vi_source === "user" ? el("span", { class: "muted" }, "đã sửa tay") : s.vi_source === "shortened" ? el("span", { class: "muted" }, "đã rút gọn tự động") : null,
      s.warning ? el("div", { class: "badge b-error" }, `xem lại: ${s.warning}`) : null),
    el("td", {}, el("input", { type: "text", class: "speaker", value: s.speaker || "A", onchange: (e) => saveSeg(s.id, { speaker: e.target.value }) })),
    el("td", {}, badge(s.audio_status, STATUS_VI[s.audio_status])),
  )));
}

async function saveSeg(id, body) {
  try { await api("PATCH", `/api/projects/${state.id}/segments/${id}`, body); await refresh(); }
  catch (e) { showError(e.message); }
}

async function runJob(job) {
  try { await api("POST", `/api/projects/${state.id}/run/${job}`); showError(""); await refresh(); }
  catch (e) { showError(e.message); }
}
$("#run-transcribe").addEventListener("click", () => runJob("transcribe"));
$("#run-translate").addEventListener("click", () => runJob("translate"));

$("#save-glossary").addEventListener("click", async () => {
  const glossary = {};
  for (const line of $("#glossary").value.split("\n")) {
    const [zh, ...rest] = line.split("=");
    if (zh.trim() && rest.join("=").trim()) glossary[zh.trim()] = rest.join("=").trim();
  }
  await saveSettings({ glossary });
});

async function saveSettings(body) {
  try { await api("PUT", `/api/projects/${state.id}/settings`, body); await refresh(); }
  catch (e) { showError(e.message); }
}

// ---------------------------------------------------------------- bước 3: lồng tiếng
function renderTts() {
  const d = state.data;
  const speakers = [...new Set(d.segments.map((s) => s.speaker || "A"))];
  const voices = (state.config && state.config.voices) || [];
  $("#voices").replaceChildren(...speakers.map((spk) => el("label", {}, `Người nói ${spk}: `,
    el("select", { onchange: (e) => saveSettings({ voices: { ...d.settings.voices, [spk]: e.target.value } }) },
      voices.map((v) => { const o = el("option", { value: v.id }, v.label); if ((d.settings.voices[spk] || d.settings.voices.A) === v.id) o.selected = true; return o; })))));

  if (!d.segments.length) { $("#tts-body").replaceChildren(); return; }
  $("#tts-body").replaceChildren(...d.segments.map((s) => {
    const slot = s.end - s.start;
    const fit = s.audio_status === "ready" && s.fit ? el("div", { class: "fit" },
      `gốc ${slot.toFixed(1)}s · Việt ${s.tts_duration.toFixed(1)}s `,
      badge(s.fit, FIT_VI[s.fit] + (s.fit === "tempo" || s.fit === "overflow" ? ` x${s.tempo.toFixed(2)}` : "")),
      s.vi_source === "shortened" ? el("span", { class: "old", title: "Bản trước khi rút gọn" }, s.vi_long) : null) : badge(s.audio_status, STATUS_VI[s.audio_status]);
    const audio = s.audio_status === "ready" || s.audio_status === "stale"
      ? el("audio", { controls: "", preload: "none", src: fileUrl(`tts/${String(s.id).padStart(3, "0")}.wav`, `${s.tts_duration}-${s.vi}`) }) : null;
    return el("tr", {},
      el("td", {}, s.id),
      el("td", { class: "vi" }, s.vi, s.error ? el("div", { class: "muted" }, s.error) : null),
      el("td", {}, fit),
      el("td", {}, audio),
      el("td", {}, el("button", { class: "small", onclick: () => api("POST", `/api/projects/${state.id}/segments/${s.id}/tts`).then(refresh).catch((e) => showError(e.message)) }, "Tạo lại")));
  }));
}
$("#run-tts").addEventListener("click", () => runJob("tts"));

function renderReport() {
  const r = state.data.report || [];
  $("#report").replaceChildren(r.length
    ? el("details", {}, el("summary", {}, `${r.length} câu cần chú ý (tăng tốc / rút gọn / lệch)`),
        el("ul", {}, r.map((x) => el("li", {}, `#${x.id} ${x.kind}: ${x.detail}`))))
    : "");
}

// ---------------------------------------------------------------- bước 4: xuất video
const SLIDERS = [["#box-top", "box_top"], ["#box-height", "box_height"], ["#box-opacity", "opacity"],
  ["#bg-volume", "bg_volume"], ["#duck-db", "duck_db"], ["#keep-vocals", "keep_original_vocals"]];

function fillSettings() {
  const d = state.data, s = d.settings, box = d.box_effective || { box_top: 0.8, box_height: 0.17, opacity: 0.85 };
  const values = { ...box, bg_volume: s.bg_volume, duck_db: s.duck_db, keep_original_vocals: s.keep_original_vocals };
  for (const [sel, key] of SLIDERS) { $(sel).value = values[key]; $(sel).nextElementSibling.value = values[key]; }
  document.querySelectorAll("input[name=subs]").forEach((r) => (r.checked = r.value === s.subtitle_mode));
  const style = s.subtitle_style || (state.config && state.config.subtitle.style) || "outline";
  document.querySelectorAll("input[name=style]").forEach((r) => (r.checked = r.value === style));
  $("#glossary").value = Object.entries(s.glossary || {}).map(([k, v]) => `${k} = ${v}`).join("\n");
  placeOverlay();
}
for (const [sel] of SLIDERS) $(sel).addEventListener("input", (e) => { e.target.nextElementSibling.value = e.target.value; placeOverlay(); });

// Vẽ khung che lên video xem trước, tính theo vùng hình thật (video có thể bị viền đen)
function placeOverlay() {
  const v = $("#box-video"), o = $("#box-overlay");
  // chưa đọc xong metadata video thì dùng kích thước đã lưu trong dự án
  const info = (state.data && state.data.video) || {};
  const vw = v.videoWidth || info.width, vh = v.videoHeight || info.height;
  if (!vw || !v.clientWidth) { o.style.display = "none"; return; }
  const scale = Math.min(v.clientWidth / vw, v.clientHeight / vh);
  const w = vw * scale, h = vh * scale;
  const left = (v.clientWidth - w) / 2, top = (v.clientHeight - h) / 2;
  const style = (document.querySelector("input[name=style]:checked") || {}).value || "outline";
  Object.assign(o.style, { display: "flex", left: `${left}px`, width: `${w}px`, top: `${top + h * $("#box-top").value}px`,
    height: `${h * $("#box-height").value}px`, right: "auto",
    background: style === "box" ? `rgba(0,0,0,${$("#box-opacity").value})` : "transparent",
    backdropFilter: style === "blur" ? "blur(8px)" : "none",
    outline: style === "outline" ? "1px dashed rgba(255,255,255,.6)" : "none",
    fontSize: `${Math.round(Math.min(w, h) * 0.04)}px` });
  o.textContent = "Phụ đề tiếng Việt";
}
document.querySelectorAll("input[name=style]").forEach((r) => r.addEventListener("change", placeOverlay));
$("#box-video").addEventListener("loadedmetadata", placeOverlay);
window.addEventListener("resize", placeOverlay);

$("#detect-box").addEventListener("click", () => runJob("detect_box"));

$("#run-render").addEventListener("click", async () => {
  await saveSettings({
    subtitle_mode: document.querySelector("input[name=subs]:checked").value,
    subtitle_style: document.querySelector("input[name=style]:checked").value,
    bg_volume: +$("#bg-volume").value, duck_db: +$("#duck-db").value, keep_original_vocals: +$("#keep-vocals").value,
    box: { box_top: +$("#box-top").value, box_height: +$("#box-height").value, opacity: +$("#box-opacity").value },
  });
  await runJob("render");
  state.goToExport = true;
});

function renderDownloads() {
  const f = state.data.files, t = state.data.steps.render.time || "";
  const v = $("#final-video");
  if (f["final.mp4"]) { const src = fileUrl("final.mp4", t); if (!v.src.endsWith(encodeURIComponent(t))) v.src = src; v.hidden = false; }
  else v.hidden = true;
  const links = [["final.mp4", "Tải video .mp4"], ["final.vi.srt", "Phụ đề Việt .srt"], ["final.zh.srt", "Phụ đề Trung .srt"], ["final.bilingual.srt", "Song ngữ .srt"]];
  $("#downloads").replaceChildren(...links.filter(([n]) => f[n]).map(([n, label]) => el("a", { href: fileUrl(n, t), download: n }, label)));
}

// ---------------------------------------------------------------- khởi động
(async () => {
  try { state.config = await api("GET", "/api/config"); } catch (e) { showError("Không tải được cấu hình: " + e.message); }
  await loadProjects();
  const [id, step] = location.hash.slice(1).split("/");
  if (id) { await openProject(decodeURIComponent(id)); goStep(step || 1); }
})();
