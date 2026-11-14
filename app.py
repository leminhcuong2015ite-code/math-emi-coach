import re
import io
import time
import random
import difflib
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from gtts import gTTS
from pydantic import BaseModel, Field
from youtube_transcript_api import YouTubeTranscriptApi
from google import genai
from google.genai import types
from supabase import create_client, Client

st.set_page_config(page_title="Math EMI Coach Cloud", page_icon="📐", layout="centered")

FALLBACK_MODELS = ["gemini-3.8-flash", "gemini-3.8-pro", "gemini-2.5-pro"]

# --- 1. KẾT NỐI SUPABASE & GEMINI API KEY ---
RAW_SUPABASE_URL = st.secrets.get("SUPABASE_URL", "https://snrwrbqhmsjjhibuwcel.supabase.co")
SUPABASE_URL = re.sub(r'/rest/v1/?$', '', RAW_SUPABASE_URL.strip()).rstrip('/')
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")
DEFAULT_GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", "")

@st.cache_resource
def init_supabase() -> Client | None:
    if not SUPABASE_URL or not SUPABASE_KEY or "DÁN_" in SUPABASE_KEY:
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY.strip())
    except Exception as e:
        st.error(f"Lỗi kết nối Supabase: {e}")
        return None

db = init_supabase()
query_url = st.query_params.get("url", "")

# --- 2. CẤU TRÚC DỮ LIỆU AI ---
class MathVocab(BaseModel):
    term: str = Field(description="Thuật ngữ toán học tiếng Anh (chỉ 1-3 từ)")
    phonetic: str = Field(description="Phiên âm IPA")
    meaning: str = Field(description="Nghĩa tiếng Việt chuyên ngành")
    context_sentence: str = Field(description="Câu nguyên văn trong bài giảng có chứa đúng thuật ngữ này")
    timestamp: int = Field(description="Mốc thời gian bắt đầu câu nói (giây)")
    duration: int = Field(default=0, description="Thời lượng giáo viên nói hết câu context_sentence này (tính bằng giây, thường từ 4 đến 12 giây)")

class ClassroomPhrase(BaseModel):
    phrase: str = Field(description="Câu giao tiếp sư phạm nguyên văn trong bài (dài 6-15 từ)")
    function: str = Field(description="Mục đích sử dụng")
    vietnamese_equivalent: str = Field(description="Cách diễn đạt tương đương của giáo viên VN")
    timestamp: int = Field(description="Mốc thời gian bắt đầu câu nói (giây)")
    duration: int = Field(default=0, description="Thời lượng giáo viên nói hết câu phrase này (tính bằng giây, thường từ 4 đến 12 giây)")

class GeminiOutput(BaseModel):
    suggested_title: str = Field(description="Tiêu đề ngắn gọn cho bài giảng (dưới 8 từ, tiếng Việt, ví dụ: Giải phương trình bậc nhất)")
    suggested_category: str = Field(description="Tên mục lục/chuyên đề Toán (ví dụ: Đại số, Hình học, Giải tích, Số học)")
    lesson_summary: str
    vocabularies: list[MathVocab]
    classroom_phrases: list[ClassroomPhrase]
    practice_prompt: str

# --- 3. HÀM GỌI GEMINI CHỐNG QUÁ TẢI 503 ---
def generate_with_retry(client: genai.Client, contents, config=None, max_retries: int = 4):
    last_err = None
    for attempt in range(max_retries):
        for model_name in FALLBACK_MODELS:
            try:
                return client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
            except Exception as e:
                err_str = str(e)
                last_err = e
                if "404" in err_str or "NOT_FOUND" in err_str:
                    continue
                if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str:
                    wait_sec = (attempt + 1) * 4
                    st.toast(f"⏳ Máy chủ AI đang đông ({model_name}), tự động thử lại sau {wait_sec}s (Lần {attempt+1}/{max_retries})...")
                    time.sleep(wait_sec)
                    break
                raise e
    raise last_err

# --- 4. HÀM QUẢN LÝ MỤC LỤC, ĐỔI TÊN & ĐÁNH DẤU HOÀN THÀNH TRÊN SUPABASE ---
def update_lesson_metadata(video_id: str, new_title: str, new_folder: str):
    if not db:
        return
    res = db.table("lesson_history").select("analysis_json").eq("video_id", video_id).execute()
    if res.data:
        analysis_data = res.data[0]["analysis_json"]
        clean_title = new_title.strip() or "Bài giảng chưa đặt tên"
        clean_folder = new_folder.strip() or "Chưa phân loại"
        analysis_data["custom_title"] = clean_title
        analysis_data["folder_name"] = clean_folder
        
        formatted_summary = f"[{clean_folder}] {clean_title}"
        db.table("lesson_history").update({
            "lesson_summary": formatted_summary,
            "analysis_json": analysis_data
        }).eq("video_id", video_id).execute()

        if "analysis" in st.session_state and st.session_state["analysis"].get("video_id") == video_id:
            st.session_state["analysis"]["custom_title"] = clean_title
            st.session_state["analysis"]["folder_name"] = clean_folder

def mark_lesson_completed(video_id: str, score_pct: int):
    if not db:
        return
    res = db.table("lesson_history").select("analysis_json").eq("video_id", video_id).execute()
    if res.data:
        analysis_data = res.data[0]["analysis_json"]
        analysis_data["is_completed"] = True
        prev_best = analysis_data.get("best_score", 0)
        analysis_data["best_score"] = max(prev_best, score_pct)
        db.table("lesson_history").update({
            "analysis_json": analysis_data
        }).eq("video_id", video_id).execute()

        if "analysis" in st.session_state and st.session_state["analysis"].get("video_id") == video_id:
            st.session_state["analysis"]["is_completed"] = True
            st.session_state["analysis"]["best_score"] = analysis_data["best_score"]

def parse_folder_and_title(row: dict) -> tuple[str, str]:
    a_json = row.get("analysis_json") or {}
    if isinstance(a_json, dict) and a_json.get("custom_title"):
        return a_json.get("folder_name", "Chưa phân loại"), a_json.get("custom_title")
    
    raw_summary = row.get("lesson_summary", "")
    match = re.match(r'^\[(.*?)\]\s*(.*)$', raw_summary)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "Chưa phân loại", (raw_summary[:55] + "..." if len(raw_summary) > 55 else raw_summary)

def render_folder_selector(label: str, existing_folders: list[str], current_val: str = "", key_prefix: str = "fld") -> str:
    default_list = ["Đại số", "Hình học", "Giải tích", "Số học", "Thống kê & Xác suất", "Kỹ năng đứng lớp bằng tiếng Anh"]
    merged_folders = sorted(list(set(default_list + existing_folders)))
    if current_val and current_val not in merged_folders:
        merged_folders.insert(0, current_val)
    
    options = merged_folders + ["➕ Tự nhập chuyên đề mới..."]
    default_idx = options.index(current_val) if current_val in options else 0
    
    choice = st.selectbox(label, options, index=default_idx, key=f"{key_prefix}_sel")
    if choice == "➕ Tự nhập chuyên đề mới...":
        custom_f = st.text_input("Nhập tên chuyên đề mới:", value="", placeholder="VD: Toán 7 - Chương 1...", key=f"{key_prefix}_custom")
        return custom_f.strip() if custom_f.strip() else "Chưa phân loại"
    return choice

# --- 5. HÀM TÍNH SỐ TỪ THEO PHÚT & CHUYỂN ĐỔI LATEX (HỖ TRỢ PHIÊN ÂM IPA CHUẨN) ---
def calculate_target_items(duration_seconds: int) -> tuple[int, float]:
    minutes = duration_seconds / 60.0
    if minutes < 10:
        count = 5
    else:
        count = 5 + 2 * int(minutes // 10)
    return min(count, 25), round(minutes, 1)

def get_smart_duration(item_dict: dict, sentence_text: str, extra_buffer: int = 1) -> int:
    ai_dur = item_dict.get("duration", 0)
    if ai_dur and ai_dur >= 3:
        return int(ai_dur) + extra_buffer
    word_count = len(sentence_text.split())
    estimated = max(4, min(16, round(word_count / 2.3) + 2))
    return estimated + extra_buffer

def escape_latex(text: str) -> str:
    if not text:
        return ""
    parts = re.split(r'(\$.*?\$)', text)
    escaped_parts = []
    replacements = {'&': r'\&', '%': r'\%', '#': r'\#', '_': r'\_'}
    for part in parts:
        if part.startswith('$') and part.endswith('$') and len(part) >= 2:
            escaped_parts.append(part)
        else:
            for k, v in replacements.items():
                part = part.replace(k, v)
            escaped_parts.append(part)
    return "".join(escaped_parts)

def ipa_to_latex(phonetic: str) -> str:
    if not phonetic:
        return ""
    ipa_map = {
        'ə': r'\textschwa{}',
        'ɚ': r'\textrhookschwa{}',
        'ɪ': r'\textsci{}',
        'iː': r'i\textlengthmark{}',
        'ʊ': r'\textupsilon{}',
        'uː': r'u\textlengthmark{}',
        'ʌ': r'\textturnv{}',
        'ɑ': r'\textscripta{}',
        'ɒ': r'\textturnscripta{}',
        'ɔ': r'\textopeno{}',
        'æ': r'\ae{}',
        'ɛ': r'\textepsilon{}',
        'ɜ': r'\textrevepsilon{}',
        'ɝ': r'\textrhookrevepsilon{}',
        'θ': r'\texttheta{}',
        'ð': r'\dh{}',
        'ʃ': r'\textesh{}',
        'ʒ': r'\textyogh{}',
        'ŋ': r'\ng{}',
        'ʧ': r't\textesh{}',
        'ʤ': r'd\textyogh{}',
        'ˈ': r'\textprimstress{}',
        'ˌ': r'\textsecstress{}',
        'ː': r'\textlengthmark{}',
        'ɡ': r'g',
        'ɹ': r'\textturnr{}',
        'ɾ': r'\textfishhookr{}',
        'ʔ': r'\textglotstop{}',
    }
    out = phonetic.strip().strip('/')
    for k, v in ipa_map.items():
        out = out.replace(k, v)
    out = "".join(ch if ord(ch) < 128 else "" for ch in out)
    return out

def generate_latex_document(data: dict, full_document: bool = True) -> str:
    video_id = data["video_id"]
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    custom_title = escape_latex(data.get("custom_title") or "PHIẾU HỌC TẬP TOÁN TIẾNG ANH (EMI)")
    folder_name = escape_latex(data.get("folder_name") or "Chuyên đề EMI")
    summary = escape_latex(data["lesson_summary"])
    practice_prompt = escape_latex(data["practice_prompt"])

    body_lines = [
        r"% ===================================================================",
        f"% MỤC LỤC: {folder_name} | BÀI: {custom_title} (ID: {video_id})",
        r"% ===================================================================",
        f"\\begin{{tcolorbox}}[colback=blue!5!white,colframe=blue!75!black,title=\\textbf{{{folder_name.upper()} --- {custom_title.upper()}}}]",
        f"\\textbf{{Tóm tắt nội dung:}} {summary}\\\\[4pt]",
        f"\\textbf{{Link bài giảng gốc:}} \\href{{{video_url}}}{{\\texttt{{{video_url}}}}}",
        r"\end{tcolorbox}",
        "",
        r"\section*{1. Thuật ngữ Toán học Trọng tâm (Key Mathematical Vocabulary)}",
        r"\begin{enumerate}[label=\textbf{\arabic*.}, leftmargin=*, itemsep=6pt]"
    ]

    for v in data["vocabularies"]:
        term = escape_latex(v["term"])
        phonetic_safe = ipa_to_latex(v["phonetic"])
        meaning = escape_latex(v["meaning"])
        ctx = escape_latex(v["context_sentence"])
        ts = v["timestamp"]
        t_link = f"https://www.youtube.com/watch?v={video_id}&t={ts}s"
        body_lines.append(
            f"    \\item \\textbf{{{term}}} [/\\textipa{{{phonetic_safe}}}/] --- \\textbf{{{meaning}}} "
            f"(\\href{{{t_link}}}{{\\textcolor{{blue}}{{[{ts}s]}}}})\\\\"
        )
        body_lines.append(f"    \\textit{{Ngữ cảnh:}} ``{ctx}''")

    body_lines.extend([
        r"\end{enumerate}",
        "",
        r"\section*{2. Mẫu câu Giao tiếp Sư phạm (Classroom Pedagogical Phrases)}",
        r"\begin{enumerate}[label=\textbf{\arabic*.}, leftmargin=*, itemsep=6pt]"
    ])

    for p in data["classroom_phrases"]:
        phrase = escape_latex(p["phrase"])
        func = escape_latex(p["function"])
        vn_eq = escape_latex(p["vietnamese_equivalent"])
        ts = p["timestamp"]
        t_link = f"https://www.youtube.com/watch?v={video_id}&t={ts}s"
        body_lines.append(
            f"    \\item ``\\textbf{{{phrase}}}'' "
            f"(\\href{{{t_link}}}{{\\textcolor{{blue}}{{[{ts}s]}}}})\\\\"
        )
        body_lines.append(f"    \\textit{{Mục đích sư phạm ({func}):}} {vn_eq}")

    body_lines.extend([
        r"\end{enumerate}",
        "",
        r"\section*{3. Bài tập Điền từ Khuyết (Cloze Listening Drill)}",
        r"\begin{enumerate}[label=\textbf{Câu \arabic*:}, leftmargin=*, itemsep=8pt]"
    ])

    answer_keys = []
    for idx, v in enumerate(data["vocabularies"], 1):
        masked = re.sub(
            re.escape(v["term"]),
            r"\\underline{\\hspace{3cm}}",
            v["context_sentence"],
            flags=re.IGNORECASE
        )
        masked_escaped = escape_latex(masked)
        meaning = escape_latex(v["meaning"])
        body_lines.append(f"    \\item ``{masked_escaped}'' \\hfill \\textit{{(Gợi ý: {meaning})}}")
        answer_keys.append(f"\\textbf{{Câu {idx}:}} {escape_latex(v['term'])}")

    body_lines.extend([
        r"\end{enumerate}",
        "",
        r"\section*{4. Tình huống Thực hành Giảng bài (Teach-back Challenge)}",
        r"\begin{tcolorbox}[colback=green!5!white,colframe=green!60!black,title=\textbf{THỬ THÁCH SƯ PHẠM (PEDAGOGICAL SCENARIO)}]",
        f"{practice_prompt}",
        r"\end{tcolorbox}",
        "",
        r"\vspace{6pt}",
        r"\noindent\textbf{Đáp án phần Điền từ:} " + " \\quad|\\quad ".join(answer_keys)
    ])

    body_content = "\n".join(body_lines)
    if not full_document:
        return body_content

    preamble = f"""\\documentclass[12pt,a4paper]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1,T5]{{fontenc}}
\\usepackage[vietnamese]{{babel}}
\\usepackage{{tipa}}
\\usepackage{{amsmath,amssymb,amsfonts}}
\\usepackage[margin=2cm]{{geometry}}
\\usepackage{{enumitem}}
\\usepackage{{xcolor}}
\\usepackage{{tcolorbox}}
\\usepackage[colorlinks=true,linkcolor=blue,urlcolor=blue]{{hyperref}}

\\title{{\\textbf{{{custom_title.upper()}}}}}
\\author{{Chuyên mục: {folder_name} --- Math EMI Coach}}
\\date{{\\today}}

\\begin{{document}}
\\maketitle
"""
    closing = "\n\\end{document}\n"
    return preamble + body_content + closing

# --- 6. HÀM TẠO BÀI KIỂM TRA LỘN XỘN KIỂU DUOLINGO ---
def build_duolingo_quiz(data: dict) -> list[dict]:
    vocabs = data.get("vocabularies", [])
    phrases = data.get("classroom_phrases", [])
    questions = []

    all_meanings = [v["meaning"] for v in vocabs]

    for v in vocabs:
        wrong_pool = [m for m in all_meanings if m != v["meaning"]]
        distractors = random.sample(wrong_pool, min(3, len(wrong_pool)))
        options = distractors + [v["meaning"]]
        random.shuffle(options)
        questions.append({
            "type": "mcq_meaning",
            "prompt": f"Thuật ngữ Toán học **\"{v['term']}\"** (`/{v['phonetic']}/`) có nghĩa là gì?",
            "tts_text": v["term"],
            "options": options,
            "answer": v["meaning"],
            "explanation": f"Ngữ cảnh trong bài: *\"{v['context_sentence']}\"*"
        })

    for v in vocabs:
        masked = re.sub(re.escape(v["term"]), " ______ ", v["context_sentence"], flags=re.IGNORECASE)
        questions.append({
            "type": "fill_blank",
            "prompt": f"Điền thuật ngữ tiếng Anh còn thiếu vào câu giảng sau:\n\n🗣️ *\"{masked}\"*",
            "hint": f"Nghĩa tiếng Việt: {v['meaning']} (`/{v['phonetic']}/`)",
            "tts_text": v["context_sentence"],
            "answer": v["term"],
            "explanation": f"Đáp án chuẩn: **{v['term']}**"
        })

    all_phrase_texts = [p["phrase"] for p in phrases]
    for p in phrases:
        wrong_p = [pt for pt in all_phrase_texts if pt != p["phrase"]]
        distractors = random.sample(wrong_p, min(3, len(wrong_p)))
        options = distractors + [p["phrase"]]
        random.shuffle(options)
        questions.append({
            "type": "mcq_phrase",
            "prompt": f"Khi muốn nói với học sinh: **\"{p['vietnamese_equivalent']}\"** (*{p['function']}*), bạn dùng câu nào sau đây?",
            "tts_text": p["phrase"],
            "options": options,
            "answer": p["phrase"],
            "explanation": f"Câu gốc chuẩn tại giây {p['timestamp']}s: *\"{p['phrase']}\"*"
        })

    for p in random.sample(phrases, min(2, len(phrases))):
        questions.append({
            "type": "listen_type",
            "prompt": "🔊 Bấm nghe câu phát âm mẫu bên dưới và gõ lại câu tiếng Anh bạn nghe được:",
            "hint": f"Gợi ý nghĩa: \"{p['vietnamese_equivalent']}\"",
            "tts_text": p["phrase"],
            "answer": p["phrase"],
            "explanation": f"Câu chuẩn: **{p['phrase']}**"
        })

    random.shuffle(questions)
    return questions[:min(8, len(questions))]

# --- 7. TRÌNH PHÁT YOUTUBE LẶP CÂU TỰ ĐỘNG ---
def render_youtube_player(video_id: str, start_sec: int = 0, duration: int = 6, autoplay: int = 1, loop_clip: bool = True):
    end_sec = start_sec + duration
    if not loop_clip:
        embed_url = f"https://www.youtube.com/embed/{video_id}?start={start_sec}&autoplay={autoplay}&playsinline=1&rel=0"
        iframe_html = f"""
        <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; border-radius: 10px; background: #000; margin-top: 8px;">
            <iframe src="{embed_url}" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: 0;" 
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                allowfullscreen referrerpolicy="strict-origin-when-cross-origin">
            </iframe>
        </div>
        """
        components.html(iframe_html, height=235)
        return

    loop_html = f"""
    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; border-radius: 10px; background: #000; margin-top: 6px;">
        <div id="yt-player"></div>
    </div>
    <div style="background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 8px; padding: 6px 10px; margin-top: 6px; font-family: sans-serif; font-size: 12.5px; color: #333;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <span id="loop-info">🔁 Đoạn lặp: <b>{start_sec}s ➔ {end_sec}s</b> ({duration}s)</span>
            <button id="toggle-loop-btn" onclick="toggleLoop()" style="padding: 3px 8px; border-radius: 5px; border: 1px solid #a5d6a7; background: #e8f5e9; color: #2e7d32; font-weight: bold; cursor: pointer;">
                🔁 Đang BẬT lặp
            </button>
        </div>
        <div style="display: flex; gap: 5px; justify-content: space-between;">
            <button onclick="adjustStart(-1)" style="flex: 1; padding: 4px; border-radius: 4px; border: 1px solid #ccc; background: #fff; cursor: pointer;">⏪ Đầu -1s</button>
            <button onclick="adjustStart(1)" style="flex: 1; padding: 4px; border-radius: 4px; border: 1px solid #ccc; background: #fff; cursor: pointer;">⏩ Đầu +1s</button>
            <button onclick="adjustEnd(-1)" style="flex: 1; padding: 4px; border-radius: 4px; border: 1px solid #ccc; background: #fff; cursor: pointer;">➖ Cuối -1s</button>
            <button onclick="adjustEnd(1)" style="flex: 1; padding: 4px; border-radius: 4px; border: 1px solid #ccc; background: #fff; cursor: pointer;">➕ Cuối +1s</button>
        </div>
    </div>

    <script>
      var tag = document.createElement('script');
      tag.src = "https://www.youtube.com/iframe_api";
      var firstScriptTag = document.getElementsByTagName('script')[0];
      firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);

      var player;
      var startSec = {start_sec};
      var endSec = {end_sec};
      var isLooping = true;
      var loopTimer = null;

      function updateInfoText() {{
        var dur = Math.max(1, endSec - startSec);
        document.getElementById('loop-info').innerHTML = "🔁 Đoạn lặp: <b>" + startSec + "s ➔ " + endSec + "s</b> (" + dur + "s)";
      }}

      function adjustStart(delta) {{
        startSec = Math.max(0, startSec + delta);
        if (startSec >= endSec) endSec = startSec + 2;
        updateInfoText();
        if (player && player.seekTo) player.seekTo(startSec, true);
      }}

      function adjustEnd(delta) {{
        endSec = Math.max(startSec + 2, endSec + delta);
        updateInfoText();
      }}

      function onYouTubeIframeAPIReady() {{
        player = new YT.Player('yt-player', {{
          videoId: '{video_id}',
          playerVars: {{
            'autoplay': {autoplay},
            'start': startSec,
            'playsinline': 1,
            'rel': 0,
            'modestbranding': 1
          }},
          events: {{
            'onReady': onPlayerReady,
            'onStateChange': onPlayerStateChange
          }}
        }});
      }}

      function onPlayerReady(event) {{
        var iframe = event.target.getIframe();
        iframe.style.position = "absolute";
        iframe.style.top = "0";
        iframe.style.left = "0";
        iframe.style.width = "100%";
        iframe.style.height = "100%";
        if ({autoplay} === 1) {{
          event.target.playVideo();
        }}
      }}

      function onPlayerStateChange(event) {{
        if (event.data == YT.PlayerState.PLAYING) {{
          if (loopTimer) clearInterval(loopTimer);
          loopTimer = setInterval(function() {{
            if (isLooping && player && player.getCurrentTime) {{
              var currentTime = player.getCurrentTime();
              if (currentTime >= endSec || currentTime < startSec - 1) {{
                player.seekTo(startSec, true);
                player.playVideo();
              }}
            }}
          }}, 200);
        }} else {{
          if (loopTimer) clearInterval(loopTimer);
        }}
      }}

      function toggleLoop() {{
        isLooping = !isLooping;
        var btn = document.getElementById('toggle-loop-btn');
        if (isLooping) {{
          btn.innerHTML = "🔁 Đang BẬT lặp";
          btn.style.background = "#e8f5e9";
          btn.style.color = "#2e7d32";
          if (player && player.seekTo) player.seekTo(startSec, true);
        }} else {{
          btn.innerHTML = "➡️ Đang TẮT lặp";
          btn.style.background = "#f5f5f5";
          btn.style.color = "#616161";
        }}
      }}
    </script>
    """
    components.html(loop_html, height=300)

# --- 8. HÀM TEXT-TO-SPEECH ĐỌC CHUẨN & ĐỌC CHẬM (ĐÃ SỬA LỖI DUPLICATE ELEMENT ID) ---
@st.cache_data(show_spinner=False)
def generate_tts_audio(text: str, slow: bool = False) -> bytes:
    clean_for_speech = re.sub(r'[\$\\\{\}\_\^]', ' ', text)
    tts = gTTS(text=clean_for_speech, lang='en', tld='com', slow=slow)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    return fp.getvalue()

def render_tts_buttons(text: str, key_prefix: str):
    col_normal, col_slow = st.columns(2)
    with col_normal:
        if st.button("🔊 Đọc chuẩn", key=f"{key_prefix}_norm", use_container_width=True):
            st.session_state["active_tts_key"] = key_prefix
            st.session_state["active_tts_bytes"] = generate_tts_audio(text, slow=False)
    with col_slow:
        if st.button("🐢 Đọc chậm", key=f"{key_prefix}_slow", use_container_width=True):
            st.session_state["active_tts_key"] = key_prefix
            st.session_state["active_tts_bytes"] = generate_tts_audio(text, slow=True)
            
    if st.session_state.get("active_tts_key") == key_prefix and "active_tts_bytes" in st.session_state:
        try:
            st.audio(
                st.session_state["active_tts_bytes"],
                format="audio/mp3",
                autoplay=True,
                key=f"audio_player_{key_prefix}"
            )
        except TypeError:
            st.audio(
                st.session_state["active_tts_bytes"],
                format="audio/mp3",
                autoplay=True
            )

# --- 9. HÀM XỬ LÝ PHÂN TÍCH & LƯU TRỮ SUPABASE ---
def extract_video_id(url: str) -> str | None:
    pattern = r"(?:v=|\/|youtu\.be\/|shorts\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    return match.group(1) if match else None

def trigger_analysis(url_to_analyze: str, force_refresh: bool = False, custom_title_in: str = "", folder_in: str = ""):
    video_id = extract_video_id(url_to_analyze)
    if not video_id:
        st.error("Link YouTube không hợp lệ.")
        return

    if db and not force_refresh:
        try:
            res = db.table("lesson_history").select("analysis_json").eq("video_id", video_id).execute()
            if res.data:
                cached_data = res.data[0]["analysis_json"]
                cached_data["from_cache"] = True
                st.session_state["analysis"] = cached_data
                st.session_state["active_clip"] = None
                st.session_state["last_analyzed_url"] = url_to_analyze
                return
        except Exception as e:
            st.warning(f"Không kiểm tra được cache trên Supabase: {e}")

    api_key = st.session_state.get("api_key", "").strip()
    if not api_key:
        st.error("Vui lòng nhập Gemini API Key ở thanh bên trái (hoặc cấu hình trong Secrets).")
        return

    with st.spinner("Đang đo thời lượng video & phân tích chuyên sâu bằng Gemini AI (1-3 phút)..."):
        use_direct_video = False
        transcript_text = ""
        video_duration_sec = 0

        try:
            ytt_api = YouTubeTranscriptApi()
            if hasattr(ytt_api, "fetch"):
                t_list = list(ytt_api.fetch(video_id, languages=['en', 'en-US', 'en-GB']))
                if t_list:
                    last_item = t_list[-1]
                    video_duration_sec = int(last_item.start + last_item.duration)
                transcript_text = "\n".join([f"[{int(i.start)}s -> {int(i.start + i.duration)}s] {i.text}" for i in t_list])
            else:
                t_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US', 'en-GB'])
                if t_list:
                    last_item = t_list[-1]
                    video_duration_sec = int(last_item['start'] + last_item.get('duration', 5))
                transcript_text = "\n".join([f"[{int(i['start'])}s -> {int(i['start'] + i.get('duration', 5))}s] {i['text']}" for i in t_list])
        except Exception:
            use_direct_video = True

        try:
            client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=600000))
            
            if not use_direct_video and video_duration_sec > 0:
                target_count, est_minutes = calculate_target_items(video_duration_sec)
                st.toast(f"⏱️ Video dài ~{est_minutes} phút ➔ Tự động trích xuất {target_count} từ vựng & {target_count} mẫu câu.")
                count_instruction = f"Video này dài khoảng {est_minutes} phút, hãy trích xuất ĐÚNG {target_count} thuật ngữ Toán học và {target_count} câu giao tiếp sư phạm trải đều từ đầu đến cuối bài."
            else:
                count_instruction = """
                Hãy kiểm tra tổng thời lượng của video này và trích xuất số lượng thuật ngữ Toán học cũng như số lượng câu giao tiếp sư phạm theo đúng quy tắc bậc thang sau (trải đều khắp video):
                - Nếu video dưới 10 phút: trích xuất 5 thuật ngữ và 5 mẫu câu.
                - Nếu video từ 10 đến 20 phút: trích xuất 7 thuật ngữ và 7 mẫu câu.
                - Nếu video từ 20 đến 30 phút: trích xuất 9 thuật ngữ và 9 mẫu câu.
                - Nếu video từ 30 đến 40 phút: trích xuất 11 thuật ngữ và 11 mẫu câu.
                - Cứ mỗi 10 phút tăng thêm: cộng thêm 2 thuật ngữ và 2 mẫu câu (Công thức: 5 + 2 * phần_nguyên(số_phút / 10)).
                """

            base_prompt = f"""
            Bạn là chuyên gia đào tạo giáo viên Toán dạy bằng tiếng Anh (EMI). Phân tích kỹ bài giảng Toán này:
            {count_instruction}
            1. Đặt 1 tiêu đề ngắn gọn (suggested_title, dưới 8 từ) và phân loại mục lục Toán (suggested_category, ví dụ: Đại số, Hình học, Giải tích, Số học).
            2. Trích xuất danh sách thuật ngữ Toán học quan trọng nhất kèm mốc thời gian bắt đầu (timestamp) và thời lượng (duration, tính bằng giây) để nghe trọn vẹn câu context_sentence đó.
            3. Trích xuất danh sách câu giao tiếp sư phạm tự nhiên (dẫn nhập, hỏi gợi mở, chốt ý, cảnh báo lỗi sai) dài khoảng 6-15 từ kèm mốc bắt đầu (timestamp) và thời lượng nói hết câu đó (duration, tính bằng giây).
            4. Ở trường phrase và context_sentence hãy giữ nguyên chữ viết thường để người học gõ chính tả.
            5. Đưa ra 1 tình huống thực tế yêu cầu giáo viên giảng lại một khái niệm trong bài bằng tiếng Anh.
            """
            
            if use_direct_video:
                contents_payload = [
                    types.Part.from_uri(file_uri=f"https://www.youtube.com/watch?v={video_id}", mime_type="video/mp4"),
                    base_prompt
                ]
            else:
                contents_payload = f"{base_prompt}\nTranscript:\n{transcript_text[:60000]}"

            response = generate_with_retry(
                client=client,
                contents=contents_payload,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=GeminiOutput,
                    temperature=0.3,
                ),
                max_retries=4
            )
            parsed = GeminiOutput.model_validate_json(response.text)
            final_title = custom_title_in.strip() if custom_title_in.strip() else parsed.suggested_title
            final_folder = folder_in.strip() if folder_in.strip() else parsed.suggested_category

            full_data = {
                "video_id": video_id,
                "custom_title": final_title,
                "folder_name": final_folder,
                "is_completed": False,
                "best_score": 0,
                "from_cache": False,
                **parsed.model_dump()
            }

            if db:
                db.table("lesson_history").upsert({
                    "video_id": video_id,
                    "youtube_url": f"https://www.youtube.com/watch?v={video_id}",
                    "lesson_summary": f"[{final_folder}] {final_title}",
                    "analysis_json": full_data
                }).execute()

            st.session_state["analysis"] = full_data
            st.session_state["active_clip"] = None
            st.session_state["last_analyzed_url"] = url_to_analyze
        except Exception as e:
            st.error(f"⚠️ Lỗi khi gọi Gemini AI: {e}")

def evaluate_audio_with_gemini(audio_bytes: bytes, prompt_text: str) -> str:
    api_key = st.session_state.get("api_key", "").strip()
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=300000))
    response = generate_with_retry(
        client=client,
        contents=[types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"), prompt_text],
        max_retries=3
    )
    return response.text

# --- 10. TẢI DỮ LIỆU MỤC LỤC TỪ CLOUD & THANH BÊN (CỠ CHỮ IPHONE + CẤU HÌNH) ---
history_list = []
folders_map: dict[str, list[dict]] = {}
all_folders: list[str] = []

if db:
    try:
        hist_res = db.table("lesson_history").select("video_id, lesson_summary, analysis_json, created_at").order("created_at", desc=True).execute()
        history_list = hist_res.data or []
        for h in history_list:
            f_name, t_name = parse_folder_and_title(h)
            a_json = h.get("analysis_json") or {}
            h["_folder"] = f_name
            h["_title"] = t_name
            h["_completed"] = bool(a_json.get("is_completed", False))
            h["_best_score"] = int(a_json.get("best_score", 0))
            folders_map.setdefault(f_name, []).append(h)
        all_folders = sorted(list(folders_map.keys()))
    except Exception as e:
        st.sidebar.warning(f"Lỗi tải dữ liệu Cloud: {e}")

with st.sidebar:
    st.subheader("📱 Giao diện iPhone")
    font_size = st.slider(
        "🔍 Cỡ chữ hiển thị (px):",
        min_value=14, max_value=26, value=18, step=1,
        help="Kéo sang phải để phóng to chữ trên màn hình iPhone."
    )
    
    if "analysis" in st.session_state:
        if st.button("🏠 Về Màn hình chính (Kho Chuyên Đề)", type="primary", use_container_width=True):
            del st.session_state["analysis"]
            st.rerun()

    with st.expander("⚙️ Cấu hình AI & Độ trễ lặp", expanded=False):
        api_key = st.text_input("Gemini API Key:", value=DEFAULT_GEMINI_KEY, type="password")
        st.session_state["api_key"] = api_key
        extra_buffer = st.slider(
            "⏱️ Cộng thêm giây đệm cuối câu:",
            min_value=0, max_value=6, value=1, step=1
        )
    
    st.divider()
    st.subheader("☁️ Chuyển nhanh Bài học")
    if not history_list:
        st.info("Kho bài giảng đang trống.")
    else:
        for f_name in all_folders:
            with st.expander(f"📁 {f_name} ({len(folders_map[f_name])})", expanded=False):
                for h in folders_map[f_name]:
                    vid = h["video_id"]
                    status_icon = "✅" if h["_completed"] else "📖"
                    if st.button(f"{status_icon} {h['_title']}", key=f"side_open_{vid}", use_container_width=True):
                        cached = h["analysis_json"]
                        cached["from_cache"] = True
                        cached.setdefault("custom_title", h["_title"])
                        cached.setdefault("folder_name", h["_folder"])
                        st.session_state["analysis"] = cached
                        st.session_state["active_clip"] = None
                        st.rerun()

# Áp dụng CSS phóng to cỡ chữ cho toàn bộ ứng dụng trên iPhone
st.markdown(f"""
<style>
    html, body, [class*="css"], .stMarkdown, .stText, p, li, label, .stAlert {{
        font-size: {font_size}px !important;
        line-height: 1.55 !important;
    }}
    .stButton > button {{
        font-size: {max(14, font_size - 1)}px !important;
        padding-top: 0.45rem !important;
        padding-bottom: 0.45rem !important;
    }}
    input, textarea, [data-baseweb="select"] {{
        font-size: {font_size}px !important;
    }}
</style>
""", unsafe_allow_html=True)

# --- 11. GIAO DIỆN CHÍNH ---
st.title("📐 Math EMI Coach")

if "last_analyzed_url" not in st.session_state:
    st.session_state["last_analyzed_url"] = ""
if "active_clip" not in st.session_state:
    st.session_state["active_clip"] = None

def clean_text(text: str) -> str:
    return re.sub(r'[^\w\s]', '', text.lower()).strip()

def compare_dictation(user_text: str, target_text: str) -> tuple[int, str]:
    u_words = clean_text(user_text).split()
    t_words = clean_text(target_text).split()
    ratio = int(difflib.SequenceMatcher(None, u_words, t_words).ratio() * 100)
    orig_words = target_text.split()
    highlighted = [f":green[**{w}**]" if clean_text(w) in u_words else f":red[[{w}]]" for w in orig_words]
    return ratio, " ".join(highlighted)

if query_url and query_url != st.session_state["last_analyzed_url"]:
    trigger_analysis(query_url, force_refresh=False)

# --- CÔNG CỤ ĐI KÈM: NẠP VIDEO YOUTUBE MỚI ---
with st.expander("➕ Nạp & Phân tích Video YouTube mới (Công cụ thêm dữ liệu)", expanded=not bool(history_list)):
    youtube_url = st.text_input("Dán link bài giảng YouTube:", value=query_url, placeholder="https://youtu.be/...")
    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        custom_title_input = st.text_input(
            "Tên bài học (Để trống AI tự đặt):",
            placeholder="VD: Bài 1 - Phương trình bậc nhất"
        )
    with col_meta2:
        folder_input = render_folder_selector(
            "Chọn hoặc thêm Chuyên đề:",
            existing_folders=all_folders,
            current_val=all_folders[0] if all_folders else "Đại số",
            key_prefix="new_video_fld"
        )

    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        if st.button("🚀 Phân tích & Lưu vào Kho", type="primary", use_container_width=True) and youtube_url:
            trigger_analysis(youtube_url, force_refresh=False, custom_title_in=custom_title_input, folder_in=folder_input)
            st.rerun()
    with col_btn2:
        if st.button("🔄 Phân tích lại", use_container_width=True) and youtube_url:
            trigger_analysis(youtube_url, force_refresh=True, custom_title_in=custom_title_input, folder_in=folder_input)
            st.rerun()

# ==============================================================================
# TRƯỜNG HỢP A: CHƯA MỞ BÀI HỌC CỤ THỂ -> HIỂN THỊ MÀN HÌNH CHÍNH (KHO CHUYÊN ĐỀ)
# ==============================================================================
if "analysis" not in st.session_state:
    st.subheader("📚 Kho Chuyên Đề & Bài Học")
    if not history_list:
        st.info("Chưa có bài học nào trong kho. Hãy mở mục **➕ Nạp & Phân tích Video YouTube mới** ở trên để tạo bài học đầu tiên!")
    else:
        total_lessons = len(history_list)
        completed_lessons = sum(1 for h in history_list if h["_completed"])
        
        m1, m2, m3 = st.columns(3)
        m1.metric("📁 Số Chuyên đề", len(all_folders))
        m2.metric("📖 Tổng số Bài học", total_lessons)
        m3.metric("✅ Đã học xong", f"{completed_lessons}/{total_lessons}")
        st.progress(completed_lessons / total_lessons if total_lessons > 0 else 0.0)

        col_f1, col_f2 = st.columns([1, 1])
        with col_f1:
            filter_folder = st.selectbox("📁 Lọc theo Chuyên đề:", ["Tất cả chuyên đề"] + all_folders)
        with col_f2:
            search_home = st.text_input("🔍 Tìm kiếm bài học:", placeholder="Nhập tên bài...").strip().lower()

        for f_name in all_folders:
            if filter_folder != "Tất cả chuyên đề" and f_name != filter_folder:
                continue
            
            matched_lessons = [
                item for item in folders_map[f_name]
                if not search_home or search_home in item["_title"].lower() or search_home in f_name.lower()
            ]
            if not matched_lessons:
                continue

            done_in_folder = sum(1 for item in matched_lessons if item["_completed"])
            with st.expander(f"📁 Chuyên đề: {f_name} (Hoàn thành: {done_in_folder}/{len(matched_lessons)} bài)", expanded=True):
                for h in matched_lessons:
                    vid = h["video_id"]
                    a_json = h.get("analysis_json") or {}
                    n_vocab = len(a_json.get("vocabularies", []))
                    n_phrase = len(a_json.get("classroom_phrases", []))
                    summary_short = a_json.get("lesson_summary", "")
                    
                    badge = f":green[**✅ ĐÃ HỌC XONG (Kỷ lục: {h['_best_score']}%)**]" if h["_completed"] else ":orange[**⏳ Chưa hoàn thành bài kiểm tra**]"

                    with st.container(border=True):
                        st.markdown(f"### 📌 {h['_title']}")
                        st.markdown(f"{badge} &nbsp;|&nbsp; 🔤 **{n_vocab}** từ vựng &nbsp;|&nbsp; 🗣️ **{n_phrase}** mẫu câu")
                        if summary_short:
                            st.caption(f"📝 {summary_short[:140]}...")

                        b_open, b_edit, b_del = st.columns([3, 1, 1])
                        with b_open:
                            if st.button("🚀 Vào học bài này", key=f"home_open_{vid}", type="primary", use_container_width=True):
                                cached = h["analysis_json"]
                                cached["from_cache"] = True
                                cached.setdefault("custom_title", h["_title"])
                                cached.setdefault("folder_name", h["_folder"])
                                st.session_state["analysis"] = cached
                                st.session_state["active_clip"] = None
                                st.rerun()
                        with b_edit:
                            if st.button("✏️ Sửa tên", key=f"home_edit_{vid}", use_container_width=True):
                                st.session_state[f"editing_{vid}"] = not st.session_state.get(f"editing_{vid}", False)
                        with b_del:
                            if st.button("🗑️ Xóa", key=f"home_del_{vid}", use_container_width=True):
                                db.table("lesson_history").delete().eq("video_id", vid).execute()
                                st.rerun()

                        if st.session_state.get(f"editing_{vid}", False):
                            st.divider()
                            new_t = st.text_input("Tên bài học:", value=h["_title"], key=f"h_in_t_{vid}")
                            new_f = render_folder_selector(
                                "Chuyên đề / Mục lục:",
                                existing_folders=all_folders,
                                current_val=h["_folder"],
                                key_prefix=f"h_in_f_{vid}"
                            )
                            if st.button("💾 Lưu cập nhật", key=f"h_save_{vid}", type="primary", use_container_width=True):
                                update_lesson_metadata(vid, new_t, new_f)
                                st.session_state[f"editing_{vid}"] = False
                                st.toast("Đã lưu tên mới & chuyên đề!")
                                st.rerun()

# ==============================================================================
# TRƯỜNG HỢP B: ĐANG MỞ MỘT BÀI HỌC CỤ THỂ
# ==============================================================================
else:
    data = st.session_state["analysis"]
    video_id = data["video_id"]
    cur_title = data.get("custom_title") or data.get("suggested_title") or "Bài giảng đang mở"
    cur_folder = data.get("folder_name") or data.get("suggested_category") or "Chưa phân loại"
    is_done = bool(data.get("is_completed", False))
    best_sc = int(data.get("best_score", 0))

    col_back, col_status = st.columns([2, 2])
    with col_back:
        if st.button("⬅️ Quay lại Kho Chuyên Đề", use_container_width=True):
            del st.session_state["analysis"]
            st.rerun()
    with col_status:
        if is_done:
            st.success(f"✅ **ĐÃ HỌC XONG** (Điểm: {best_sc}%)")
        else:
            st.warning("⏳ **Chưa vượt qua bài Kiểm tra**")

    st.info(f"📁 **{cur_folder}** ➔ 📌 **{cur_title}**\n\n📝 *Tóm tắt:* {data['lesson_summary']}")

    with st.expander("✏️ Đổi tên bài giảng / Chuyển sang Chuyên đề khác", expanded=False):
        edit_main_title = st.text_input("Tên bài giảng:", value=cur_title, key="main_edit_title")
        edit_main_folder = render_folder_selector(
            "Chọn hoặc tự nhập Chuyên đề:",
            existing_folders=all_folders,
            current_val=cur_folder,
            key_prefix="main_edit_fld"
        )
        if st.button("💾 Lưu thay đổi", key="btn_save_main_meta", type="primary", use_container_width=True):
            update_lesson_metadata(video_id, edit_main_title, edit_main_folder)
            st.toast("Đã lưu tên mới và chuyên đề!")
            st.rerun()

    with st.expander("📺 Xem toàn bộ Video bài giảng gốc từ đầu (Không lặp)", expanded=False):
        render_youtube_player(video_id, start_sec=0, autoplay=0, loop_clip=False)

    tab_vocab, tab_phrase, tab_drill, tab_quiz, tab_speak, tab_latex, tab_notebook = st.tabs([
        "🔤 Từ vựng", "🗣️ Mẫu câu", "🎧 Phản xạ", "🏆 Kiểm tra (Duolingo)", "🎙️ Giảng thử", "📄 Xuất LaTeX", "📓 Sổ tay"
    ])

    # --- TAB 1: TỪ VỰNG ---
    with tab_vocab:
        for idx, v in enumerate(data["vocabularies"]):
            v_dur = get_smart_duration(v, v["context_sentence"], extra_buffer)
            with st.container(border=True):
                c1, c2, c3 = st.columns([3, 1, 1])
                with c1:
                    st.markdown(f"**{idx+1}. {v['term']}** `/{v['phonetic']}/` — {v['meaning']}")
                    st.caption(f"\"{v['context_sentence']}\" *(~{v_dur}s)*")
                with c2:
                    if st.button(f"▶️ {v['timestamp']}s", key=f"vp_{idx}", use_container_width=True):
                        st.session_state["active_clip"] = f"vocab_{idx}"
                with c3:
                    if st.button("💾 Lưu", key=f"vs_{idx}", use_container_width=True):
                        if db:
                            db.table("saved_items").insert({
                                "item_type": "Vocab", "content": v["term"], "phonetic_or_func": v["phonetic"],
                                "meaning": v["meaning"], "context": v["context_sentence"],
                                "video_id": video_id, "timestamp": v["timestamp"]
                            }).execute()
                            st.toast(f"Đã lưu '{v['term']}' lên Cloud!")
                
                render_tts_buttons(v["term"], key_prefix=f"vocab_tts_{idx}")
                
                if st.session_state.get("active_clip") == f"vocab_{idx}":
                    render_youtube_player(video_id, start_sec=max(0, v["timestamp"] - 1), duration=v_dur, autoplay=1, loop_clip=True)

    # --- TAB 2: MẪU CÂU ---
    with tab_phrase:
        for idx, p in enumerate(data["classroom_phrases"]):
            p_dur = get_smart_duration(p, p["phrase"], extra_buffer)
            with st.container(border=True):
                c1, c2, c3 = st.columns([3, 1, 1])
                with c1:
                    st.markdown(f"**{idx+1}. \"{p['phrase']}\"** *(~{p_dur}s)*")
                    st.write(f"🎯 *{p['function']}* — {p['vietnamese_equivalent']}")
                with c2:
                    if st.button(f"▶️ {p['timestamp']}s", key=f"pp_{idx}", use_container_width=True):
                        st.session_state["active_clip"] = f"phrase_{idx}"
                with c3:
                    if st.button("💾 Lưu", key=f"ps_{idx}", use_container_width=True):
                        if db:
                            db.table("saved_items").insert({
                                "item_type": "Phrase", "content": p["phrase"], "phonetic_or_func": p["function"],
                                "meaning": p["vietnamese_equivalent"], "context": "",
                                "video_id": video_id, "timestamp": p["timestamp"]
                            }).execute()
                            st.toast("Đã lưu mẫu câu lên Cloud!")
                
                render_tts_buttons(p["phrase"], key_prefix=f"phrase_tts_{idx}")

                if st.session_state.get("active_clip") == f"phrase_{idx}":
                    render_youtube_player(video_id, start_sec=max(0, p["timestamp"] - 1), duration=p_dur, autoplay=1, loop_clip=True)

    # --- TAB 3: TRẠM LUYỆN PHẢN XẠ TỪNG CÂU ---
    with tab_drill:
        drill_mode = st.radio(
            "Chọn bài tập rèn luyện:",
            ["1️⃣ Nghe điền từ Toán", "2️⃣ Chép chính tả câu giảng", "3️⃣ Nhại giọng (Shadowing)"],
            horizontal=True
        )
        st.divider()

        if drill_mode == "1️⃣ Nghe điền từ Toán":
            vocab_idx = st.selectbox(
                "Chọn câu bài tập:",
                range(len(data["vocabularies"])),
                format_func=lambda i: f"Câu {i+1} (Giây {data['vocabularies'][i]['timestamp']}s) — Nghĩa: {data['vocabularies'][i]['meaning']}"
            )
            v_item = data["vocabularies"][vocab_idx]
            v_dur = get_smart_duration(v_item, v_item["context_sentence"], extra_buffer)
            masked_sentence = re.sub(re.escape(v_item["term"]), " **[ ... ]** ", v_item["context_sentence"], flags=re.IGNORECASE)
            
            st.info(f"💡 Gợi ý: {v_item['meaning']} (`/{v_item['phonetic']}/`)")
            render_youtube_player(video_id, start_sec=max(0, v_item["timestamp"] - 1), duration=v_dur, autoplay=0, loop_clip=True)
            render_tts_buttons(v_item["context_sentence"], key_prefix=f"drill_cloze_{vocab_idx}")
            
            st.markdown(f"🗣️ Câu thoại: *\"{masked_sentence}\"*")
            user_ans = st.text_input("Nhập từ còn thiếu:", key=f"cloze_{vocab_idx}")
            
            if st.button("Kiểm tra đáp án", key=f"btn_cloze_{vocab_idx}"):
                if clean_text(user_ans) == clean_text(v_item["term"]):
                    st.success(f"🎉 Chính xác! Từ cần điền là **{v_item['term']}**.")
                else:
                    st.error(f"Chưa đúng. Đáp án chuẩn là: **{v_item['term']}**")

        elif drill_mode == "2️⃣ Chép chính tả câu giảng":
            phrase_idx = st.selectbox(
                "Chọn mẫu câu luyện nghe:",
                range(len(data["classroom_phrases"])),
                format_func=lambda i: f"Mẫu câu {i+1} ({data['classroom_phrases'][i]['function']} - {data['classroom_phrases'][i]['timestamp']}s)"
            )
            p_item = data["classroom_phrases"][phrase_idx]
            p_dur = get_smart_duration(p_item, p_item["phrase"], extra_buffer)

            st.info(f"🎯 **Ngữ cảnh:** {p_item['function']} — *\"{p_item['vietnamese_equivalent']}\"*")
            render_youtube_player(video_id, start_sec=max(0, p_item["timestamp"] - 1), duration=p_dur, autoplay=0, loop_clip=True)
            render_tts_buttons(p_item["phrase"], key_prefix=f"drill_dict_{phrase_idx}")
            
            dictation_input = st.text_input("Gõ lại câu tiếng Anh bạn nghe được:", key=f"dict_{phrase_idx}")

            if st.button("Chấm điểm chép chính tả", key=f"btn_dict_{phrase_idx}"):
                score, diff_view = compare_dictation(dictation_input, p_item["phrase"])
                st.metric("Độ chính xác", f"{score}%")
                st.markdown(f"**Đối chiếu từng từ:**\n\n{diff_view}")
                st.caption(f"Câu gốc chuẩn: \"{p_item['phrase']}\"")

        elif drill_mode == "3️⃣ Nhại giọng (Shadowing)":
            shadow_options = [
                {
                    "label": f"[Mẫu câu {i+1}] {p['phrase']}",
                    "text": p["phrase"],
                    "time": p["timestamp"],
                    "dur": get_smart_duration(p, p["phrase"], extra_buffer),
                    "sub": p["vietnamese_equivalent"]
                }
                for i, p in enumerate(data["classroom_phrases"])
            ] + [
                {
                    "label": f"[Từ vựng {i+1}: {v['term']}] {v['context_sentence']}",
                    "text": v["context_sentence"],
                    "time": v["timestamp"],
                    "dur": get_smart_duration(v, v["context_sentence"], extra_buffer),
                    "sub": v["meaning"]
                }
                for i, v in enumerate(data["vocabularies"])
            ]

            s_idx = st.selectbox("Chọn câu muốn luyện nhại giọng:", range(len(shadow_options)), format_func=lambda i: shadow_options[i]["label"])
            s_item = shadow_options[s_idx]

            st.markdown(f"### 🗣️ *\"{s_item['text']}\"*")
            st.caption(f"Nghĩa/Ngữ cảnh: {s_item['sub']}")
            render_youtube_player(video_id, start_sec=max(0, s_item["time"] - 1), duration=s_item["dur"], autoplay=0, loop_clip=True)
            render_tts_buttons(s_item["text"], key_prefix=f"drill_shadow_{s_idx}")

            shadow_audio = st.audio_input("Bấm Micro để đọc nhại lại câu trên:", key=f"sh_audio_{s_idx}")
            if shadow_audio and st.button("Chấm điểm phát âm câu này", type="primary", use_container_width=True):
                with st.spinner("AI đang phân tích từng âm tiết của bạn..."):
                    prompt = f"""
                    Người học đang luyện đọc nhại lại (Shadowing) câu mẫu sau của giáo viên bản xứ:
                    Câu gốc: "{s_item['text']}"
                    Hãy nghe file thu âm và chấm điểm ngắn gọn bằng tiếng Việt:
                    1. **Bạn đã đọc là:** (Ghi lại những gì nghe được từ file âm thanh).
                    2. **Độ chính xác:** Chấm điểm trên thang 100%. Chỉ rõ từ nào phát âm sai, thiếu âm đuôi hoặc nhấn sai trọng âm.
                    3. **Mẹo sửa nhanh:** Hướng dẫn cách đặt miệng/nhấn giọng cho 1-2 từ khó nhất trong câu này.
                    """
                    feedback = evaluate_audio_with_gemini(shadow_audio.getvalue(), prompt)
                    st.markdown(feedback)

    # --- TAB 4: THỬ THÁCH KIỂM TRA LỘN XỘN KIỂU DUOLINGO ---
    with tab_quiz:
        st.subheader("🦉 Thử thách Hoàn thành Bài học (Chuẩn Duolingo)")
        st.caption("Hệ thống trộn ngẫu nhiên các câu hỏi Từ vựng, Điền từ và Nghe hiểu. Đạt **từ 80% trở lên** để nhận tích xanh **✅ ĐÃ HỌC XONG**!")

        quiz_key = f"quiz_state_{video_id}"
        if quiz_key not in st.session_state:
            st.session_state[quiz_key] = {
                "questions": build_duolingo_quiz(data),
                "current_idx": 0,
                "correct_count": 0,
                "answered": False,
                "last_is_correct": False,
                "last_feedback": ""
            }

        q_state = st.session_state[quiz_key]
        questions = q_state["questions"]
        total_q = len(questions)
        cur_idx = q_state["current_idx"]

        if st.button("🔄 Tạo đề trộn ngẫu nhiên mới", key=f"reset_quiz_{video_id}"):
            st.session_state[quiz_key] = {
                "questions": build_duolingo_quiz(data),
                "current_idx": 0,
                "correct_count": 0,
                "answered": False,
                "last_is_correct": False,
                "last_feedback": ""
            }
            st.rerun()

        if cur_idx < total_q:
            q_item = questions[cur_idx]
            st.progress(cur_idx / total_q, text=f"Tiến độ: Câu {cur_idx + 1} / {total_q}  |  Đúng: {q_state['correct_count']}")

            with st.container(border=True):
                st.markdown(f"### Câu {cur_idx + 1}:")
                st.markdown(q_item["prompt"])
                if q_item.get("hint"):
                    st.info(f"💡 {q_item['hint']}")

                render_tts_buttons(q_item["tts_text"], key_prefix=f"quiz_tts_{video_id}_{cur_idx}")

                if q_item["type"] in ("mcq_meaning", "mcq_phrase"):
                    user_choice = st.radio(
                        "Chọn đáp án đúng nhất:",
                        q_item["options"],
                        key=f"quiz_opt_{video_id}_{cur_idx}",
                        disabled=q_state["answered"]
                    )
                else:
                    user_choice = st.text_input(
                        "Nhập câu trả lời của bạn:",
                        key=f"quiz_txt_{video_id}_{cur_idx}",
                        disabled=q_state["answered"]
                    )

                if not q_state["answered"]:
                    if st.button("✔️ Kiểm tra", type="primary", use_container_width=True, key=f"chk_{video_id}_{cur_idx}"):
                        if q_item["type"] in ("mcq_meaning", "mcq_phrase"):
                            is_ok = (user_choice == q_item["answer"])
                        elif q_item["type"] == "fill_blank":
                            is_ok = (clean_text(user_choice) == clean_text(q_item["answer"]))
                        else:
                            sim = difflib.SequenceMatcher(None, clean_text(user_choice).split(), clean_text(q_item["answer"]).split()).ratio()
                            is_ok = (sim >= 0.8)

                        q_state["answered"] = True
                        q_state["last_is_correct"] = is_ok
                        if is_ok:
                            q_state["correct_count"] += 1
                        q_state["last_feedback"] = q_item["explanation"]
                        st.rerun()
                else:
                    if q_state["last_is_correct"]:
                        st.success(f"🎉 **Tuyệt vời! Chính xác!**\n\n{q_state['last_feedback']}")
                    else:
                        st.error(f"❌ **Chưa chính xác!** Đáp án đúng là: **{q_item['answer']}**\n\n{q_state['last_feedback']}")

                    if st.button("➡️ Câu tiếp theo", type="primary", use_container_width=True, key=f"nxt_{video_id}_{cur_idx}"):
                        q_state["current_idx"] += 1
                        q_state["answered"] = False
                        st.rerun()
        else:
            st.progress(1.0, text=f"Hoàn thành: {total_q} / {total_q} câu!")
            final_pct = int(round((q_state["correct_count"] / total_q) * 100)) if total_q > 0 else 0
            
            st.metric("🎯 Kết quả Bài Kiểm Tra", f"{final_pct}% ({q_state['correct_count']}/{total_q} câu đúng)")
            if final_pct >= 80:
                st.balloons()
                mark_lesson_completed(video_id, final_pct)
                st.success("🏆 **CHÚC MỪNG! BẠN ĐÃ VƯỢT QUA BÀI KIỂM TRA!**\n\nBài học này đã được gắn tích xanh **✅ ĐÃ HỌC XONG** trên Kho Chuyên Đề Cloud!")
            else:
                st.warning(f"💪 Bạn đạt **{final_pct}%** (Cần đạt tối thiểu **80%** để được gắn tích xanh ĐÃ HỌC XONG). Hãy bấm **🔄 Tạo đề trộn ngẫu nhiên mới** ở trên để thử lại nhé!")

    # --- TAB 5: TỰ GIẢNG THỬ ---
    with tab_speak:
        st.info(f"**Tình huống giảng bài:** {data['practice_prompt']}")
        audio_val = st.audio_input("Bấm Micro để thu âm bài giảng của bạn:")
        if audio_val and st.button("Chấm điểm phát âm & Sư phạm", type="primary", use_container_width=True):
            with st.spinner("AI đang chấm điểm..."):
                prompt = f"""
                Giáo viên Toán đang luyện giảng bằng tiếng Anh cho tình huống: "{data['practice_prompt']}".
                Hãy nghe đoạn thu âm và nhận xét bằng tiếng Việt:
                1. **Transcript:** Chép lại chính xác những gì giáo viên đã nói.
                2. **Nhận xét phát âm & Ngữ điệu:** Chú ý các thuật ngữ Toán, trọng âm và độ trôi chảy.
                3. **Bản nâng cấp (Native Version):** Viết lại lời giảng tự nhiên, chuẩn sư phạm bản xứ.
                """
                feedback = evaluate_audio_with_gemini(audio_val.getvalue(), prompt)
                st.markdown(feedback)

    # --- TAB 6: XUẤT CODE LATEX ---
    with tab_latex:
        st.subheader("📄 Xuất Phiếu Bài Giảng & Bài Tập ra LaTeX")
        st.success("✨ Đã tự động chuyển đổi ký tự phiên âm IPA sang lệnh `\\textipa{...}` (gói `tipa`), biên dịch không lỗi với `pdflatex` trên Overleaf & TeXstudio!")
        full_doc_mode = st.checkbox(
            "Bao gồm phần khai báo đầu file (\\documentclass...\\usepackage{tipa}...\\begin{document}) để biên dịch độc lập",
            value=True
        )
        latex_code = generate_latex_document(data, full_document=full_doc_mode)
        st.download_button(
            label="📥 Tải file .tex về máy",
            data=latex_code.encode("utf-8"),
            file_name=f"emi_lesson_{video_id}.tex",
            mime="text/plain",
            type="primary",
            use_container_width=True
        )
        st.code(latex_code, language="latex")

    # --- TAB 7: SỔ TAY CLOUD ---
    with tab_notebook:
        if db:
            try:
                res = db.table("saved_items").select("*").order("id", desc=True).execute()
                items = res.data
                if not items:
                    st.info("Sổ tay trên Cloud đang trống.")
                else:
                    df = pd.DataFrame(items)
                    st.download_button(
                        "📥 Tải file CSV (Mở bằng Excel / Google Sheets)",
                        data=df.to_csv(index=False).encode("utf-8-sig"),
                        file_name="so_tay_emi_cloud.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    for item in items:
                        with st.container(border=True):
                            ca, cb, cc = st.columns([4, 1, 1])
                            with ca:
                                st.markdown(f"**{item['content']}** ({item['phonetic_or_func']}) — {item['meaning']}")
                            with cb:
                                st.link_button("▶️", f"https://www.youtube.com/watch?v={item['video_id']}&t={item['timestamp']}s")
                            with cc:
                                if st.button("🗑️", key=f"del_{item['id']}"):
                                    db.table("saved_items").delete().eq("id", item["id"]).execute()
                                    st.rerun()
                            render_tts_buttons(item["content"], key_prefix=f"nb_tts_{item['id']}")
            except Exception as e:
                st.warning(f"Chưa tải được Sổ tay từ Cloud: {e}")

# --- KET THUC FILE APP.PY ---
