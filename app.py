import streamlit as st
import fitz  # PyMuPDF
from huggingface_hub import InferenceClient
import openai
import time
import os
from dotenv import load_dotenv
from fpdf import FPDF
import tempfile

# --- تنظیمات اولیه ---
load_dotenv()
st.set_page_config(page_title="مترجم هوشمند", layout="wide")
st.title("📖 مترجم فایل‌های متنی و زیرنویس")

# راست‌چین کردن متن فارسی
st.markdown("""
<style>
p, div, h1, h2, h3, h4, h5, h6 {
    direction: rtl;
    text-align: right;
    font-family: 'Segoe UI', Tahoma, sans-serif;
}
</style>
""", unsafe_allow_html=True)

# --- توابع کمکی ---
def create_pdf(text, filename="ترجمه.pdf"):
    pdf = FPDF()
    pdf.add_page()
    
    # استفاده از فونت سازگار با فارسی (نیاز به فونت در پوشه fonts)
    try:
        pdf.add_font("Persian", "", "fonts/arial.ttf", uni=True)
    except:
        st.warning("فونت فارسی یافت نشد! از فونت پیش‌فرض استفاده می‌شود.")
        pdf.add_font("Arial", "", uni=True)
        pdf.set_font("Arial", size=12)
    else:
        pdf.set_font("Persian", size=12)
    
    pdf.multi_cell(0, 10, txt=text, align="R")
    
    # ذخیره موقت فایل PDF
    temp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(temp_dir, filename)
    pdf.output(pdf_path)
    return pdf_path

def extract_text_from_file(uploaded_file, file_type):
    try:
        uploaded_file.seek(0)  # مهم: reset file pointer
        if file_type == "PDF":
            try:
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                return "\n".join([page.get_text() for page in doc])
            except Exception as e:
                st.error(f"خطا در خواندن PDF: {str(e)}")
                return ""
        
        elif file_type == "زیرنویس (SRT)":
            content = uploaded_file.read().decode("utf-8")
            return "\n".join([block.split('\n')[2] for block in content.split('\n\n') if len(block.split('\n')) >= 3])
        
        else:  # متن ساده
            return uploaded_file.read().decode("utf-8")
    
    except Exception as e:
        st.error(f"خطای پردازش فایل: {str(e)}")
        return ""

# --- نوار کناری برای تنظیمات ---
with st.sidebar:
    st.header("تنظیمات پیشرفته")
    file_type = st.radio(
        "نوع فایل ورودی:",
        ["PDF", "زیرنویس (SRT)", "متن ساده"]
    )
    
    model_choice = st.radio(
        "مدل ترجمه:",
        ["DeepSeek (رایگان)", "OpenAI (نیاز به API)"]
    )

# --- آپلود فایل ---
uploaded_file = st.file_uploader(
    "فایل خود را آپلود کنید", 
    type=["pdf", "txt", "srt"],
    help="PDF, فایل متنی یا زیرنویس SRT"
)

if uploaded_file:
    st.success(f"✅ فایل {uploaded_file.name} با موفقیت آپلود شد!")
    
    # --- نمایش پیش‌نمایش فایل ---
    if st.checkbox("نمایش محتوای فایل اصلی"):
        text = extract_text_from_file(uploaded_file, file_type)
        st.text_area("محتوای استخراج‌شده", text, height=200)

    # --- ترجمه ---
    if st.button("ترجمه کن", type="primary"):
        with st.spinner("در حال پردازش و ترجمه..."):
            try:
                # استخراج متن
                text = extract_text_from_file(uploaded_file, file_type)
                
                # ترجمه
                if model_choice == "DeepSeek (رایگان)":
                    client = InferenceClient(
                        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                        token=os.getenv("HUGGINGFACE_TOKEN")
                    )
                    prompt = f"""
                    شما یک مترجم حرفه‌ای هستید. متن زیر را به فارسی روان ترجمه کنید:
                    {text}
                    """
                    translated = client.text_generation(prompt, max_new_tokens=3000)
                    time.sleep(2)  # جلوگیری از Rate Limit
                else:
                    openai.api_key = os.getenv("OPENAI_API_KEY")
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "متن را به فارسی روان و دقیق ترجمه کن."},
                            {"role": "user", "content": text}
                        ]
                    )
                    translated = response.choices[0].message.content
                
                # نمایش نتیجه
                st.subheader("ترجمه نهایی:")
                st.text_area("متن ترجمه‌شده", translated, height=300)
                
                # ایجاد فایل PDF برای دانلود
                pdf_path = create_pdf(translated)
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "دانلود ترجمه به صورت PDF",
                        f,
                        file_name=f"ترجمه_{uploaded_file.name.split('.')[0]}.pdf",
                        mime="application/pdf"
                    )
                
            except Exception as e:
                st.error(f"خطا در پردازش: {str(e)}")
                st.error("ممکن است فایل بسیار بزرگ باشد یا کلید API نامعتبر باشد.")
