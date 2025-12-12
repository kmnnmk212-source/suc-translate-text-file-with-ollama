import gradio as gr
from transformers import pipeline
import tempfile
import os

# === نموذج الترجمة (نستخدم نموذجًا مستقرًا) ===
# نختار Helsinki-NLP لترجمة عربي ↔ إنجليزي
pipe_ar_en = pipeline("translation", model="Helsinki-NLP/opus-mt-ar-en", device=-1)
pipe_en_ar = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ar", device=-1)

def translate_chunk(text, src, tgt):
    if not text.strip():
        return ""
    try:
        if src == "Arabic" and tgt == "English":
            return pipe_ar_en(text, max_length=512)[0]['translation_text']
        elif src == "English" and tgt == "Arabic":
            return pipe_en_ar(text, max_length=512)[0]['translation_text']
        else:
            return "[غير مدعوم]"
    except Exception as e:
        return f"[خطأ: {str(e)}]"

def split_text(text, max_len=400):
    # تقسيم ذكي حسب النقاط لتجنب قطع الجمل
    sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if s.strip()]
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 2 < max_len:
            current += sent + ". "
        else:
            if current:
                chunks.append(current.strip())
            current = sent + ". "
    if current:
        chunks.append(current.strip())
    return chunks

def translate_and_save(text, src_lang, tgt_lang):
    if not text.strip():
        return None, ""

    # تقسيم وترجمة
    chunks = split_text(text)
    translated = "\n".join(translate_chunk(chunk, src_lang, tgt_lang) for chunk in chunks)

    # حفظ في ملف مؤقت
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(translated)
        temp_path = f.name

    return temp_path, translated

def translate_from_file(file, src_lang, tgt_lang):
    if file is None:
        return None, ""
    with open(file.name, "r", encoding="utf-8") as f:
        text = f.read()
    return translate_and_save(text, src_lang, tgt_lang)

# === واجهة Gradio ===
with gr.Blocks(title="مترجم ملفات نصية") as demo:
    gr.Markdown("## 📄 مترجم ملفات نصية (عربي ↔ إنجليزي)")

    with gr.Tab("نص مباشر"):
        inp = gr.Textbox(label="النص الأصلي", lines=5)
        with gr.Row():
            src = gr.Radio(["Arabic", "English"], label="من", value="Arabic")
            tgt = gr.Radio(["English", "Arabic"], label="إلى", value="English")
        btn = gr.Button("ترجم")
        out = gr.Textbox(label="الترجمة", lines=5)
        btn.click(lambda t,s,d: translate_and_save(t,s,d)[1], [inp, src, tgt], out)

    with gr.Tab("ملف نصي"):
        file_in = gr.File(label="ارفع ملف .txt", file_types=[".txt"])
        with gr.Row():
            src2 = gr.Radio(["Arabic", "English"], label="من", value="Arabic")
            tgt2 = gr.Radio(["English", "Arabic"], label="إلى", value="English")
        btn2 = gr.Button("ترجم الملف")
        preview = gr.Textbox(label="معاينة", lines=5)
        file_out = gr.File(label="تنزيل الترجمة")
        btn2.click(translate_from_file, [file_in, src2, tgt2], [file_out, preview])

demo.queue().launch()