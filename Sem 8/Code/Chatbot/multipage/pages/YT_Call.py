import os
import re
import streamlit as st
from datetime import date
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from fpdf import FPDF

# Custom PDF class with header (including current date) and footer.
class MyPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        today = date.today().strftime("%B %d, %Y")
        header_text = f'Earnings Call Risk Analysis Report - {today}'
        self.cell(0, 10, header_text, 0, 1, 'C')
        self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def extract_video_id(url: str) -> str:
    """Extracts the video ID from a YouTube URL (supports /watch?v=, youtu.be, and /live URLs)."""
    parsed_url = urlparse(url)
    if parsed_url.hostname in ["youtu.be"]:
        return parsed_url.path[1:]
    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        if parsed_url.path.startswith("/live/"):
            segments = parsed_url.path.split("/")
            if len(segments) >= 3:
                return segments[2]
        qs = parse_qs(parsed_url.query)
        return qs.get("v", [None])[0]
    return None

def get_transcript(video_id: str) -> str:
    """Retrieves the transcript text for the given YouTube video ID."""
    try:
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([entry["text"] for entry in transcript_data])
        return transcript
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None

def get_text_chunks(text: str, chunk_size: int = 8000, chunk_overlap: int = 1000):
    """Splits text into manageable chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def get_vector_store(text_chunks, embeddings):
    """Creates a FAISS vector store from text chunks using provided embeddings."""
    return FAISS.from_texts(text_chunks, embedding=embeddings)

def get_risk_chain(vector_store, prompt_template: str, google_api_key: str):
    """Builds a conversational retrieval chain for generating a risk analysis report."""
    model_name = "gemini-1.5-flash-latest"
    llm = ChatGoogleGenerativeAI(google_api_key=google_api_key, model=model_name, temperature=0.0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        }
    )

def clean_report_text(text: str) -> str:
    """Removes unwanted markdown symbols (# and *) from the text."""
    return re.sub(r'[#*]', '', text)

def extract_company_name(report_text: str) -> str:
    """Extracts the company name from the report text using the header 'Company:'."""
    match = re.search(r'Company:\s*(.+)', report_text)
    if match:
        company = match.group(1).strip()
        company = re.sub(r'[\.,;:]+$', '', company)
        return company
    return "ytcall_report"

def generate_pdf(report_text: str) -> bytes:
    """Generates a nicely formatted PDF from the report text and returns its bytes."""
    report_text = report_text.replace("₹", "INR ")
    pdf = MyPDF()
    pdf.add_page()
    pdf.set_font("Arial", "", 12)
    lines = report_text.splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            pdf.ln(5)
        else:
            if line.endswith(":"):
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, line, ln=1)
            else:
                pdf.set_font("Arial", "", 12)
                pdf.multi_cell(0, 10, line)
    return pdf.output(dest='S').encode('latin1')



def main():
    st.set_page_config(page_title="Earnings Call Risk Analysis Report", layout="wide")
    st.title("Youtube Video Analyzer")

    youtube_url = st.text_input("Enter YouTube Earnings Call URL:")
    if st.button("Generate PDF Report") and youtube_url:
        video_id = extract_video_id(youtube_url)
        if not video_id:
            st.error("Could not extract video ID. Please check the URL.")
            return

        transcript = get_transcript(video_id)
        if not transcript:
            st.error("Transcript could not be retrieved. (Ensure the video has captions.)")
            return

        text_chunks = get_text_chunks(transcript)
        my_api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyBSNQ6CBK3BYtsHx3TXSx4XE9yEt9K1AQU")
        embeddings = GoogleGenerativeAIEmbeddings(google_api_key=my_api_key, model="models/embedding-001")
        vector_store = get_vector_store(text_chunks, embeddings)

        # Updated prompt for risk analysis that covers liquidity, solvency, market, and other risks,
        # along with a clear investment recommendation.
        prompt_template = """
You are a seasoned financial risk analyst specializing in earnings calls. Based on the transcript provided below, generate a detailed risk analysis report that includes the following sections with clear headings and bullet points:

1. Company Overview: Provide a brief introduction with key details about the company.
2. Liquidity Risk: Analyze the company's ability to meet its short-term obligations, including cash flow and working capital concerns.
3. Solvency Risk: Evaluate the company's long-term financial stability, including debt levels and capital structure.
4. Market Risk: Discuss how market fluctuations, interest rate changes, and external economic factors affect the company.
5. Other Risks: Identify any additional risks mentioned during the call (e.g., operational, regulatory, geopolitical risks).
6. Investment Recommendation: Clearly state whether it is beneficial for investors to invest in the company, and provide a detailed explanation.

At the very top of your answer, include a header line in the format:
Company: <Company Name>

Document Text:
{context}

Question: Generate a detailed risk analysis report for the company's earnings call.
Answer:
"""
        risk_chain = get_risk_chain(vector_store, prompt_template, my_api_key)
        response = risk_chain.invoke({"question": "Generate a detailed risk analysis report for the company's earnings call."})
        risk_report = response.get("answer", "No answer was returned.")

        risk_report_clean = clean_report_text(risk_report)
        company_name = extract_company_name(risk_report_clean)
        if not company_name:
            company_name = "ytcall_report"

        pdf_bytes = generate_pdf(risk_report_clean)
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name=f"{company_name}_ytcall.pdf",
            mime="application/pdf"
        )

if __name__ == '__main__':
    main()
