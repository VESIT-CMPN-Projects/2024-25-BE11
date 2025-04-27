# --- Final Imports ---
import os
import re
import io
import streamlit as st
import pandas as pd
import pdfplumber
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain

# --- Functions ---

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages[:2]:  # First 2 pages to detect company name
            if page.extract_text():
                text += page.extract_text()
    return text

def extract_full_text_for_analysis(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

def detect_company_name(text, fallback_filename="UnknownCompany"):
    # First, try to find Company: <name> style
    match = re.search(r'Company:\s*(.+)', text, re.IGNORECASE)
    if match:
        company = match.group(1).strip()
    else:
        # Then fallback to something like <Company Name> Annual Report / Sustainability Report
        match2 = re.search(r"([A-Z][A-Za-z0-9&,\.\s\-]{3,})\s+(Annual Report|Sustainability Report|ESG Report|Company|Corporation|Inc\.|Limited|LLC|Ltd\.)", text, re.IGNORECASE)
        if match2:
            company = match2.group(1).strip()
        else:
            company = fallback_filename  # If all fails, use fallback name
    
    # Clean the company name to be safe for filenames
    company = re.sub(r'[^a-zA-Z0-9_]', '_', company)  # Only allow letters, numbers, and underscores
    company = re.sub(r'_+', '_', company)  # Merge multiple underscores
    company = company.strip('_')  # Remove leading/trailing underscores

    return company

def analyze_esg_risks(text, google_api_key):
    llm = ChatGoogleGenerativeAI(
        google_api_key=google_api_key,
        model="gemini-1.5-flash-latest",
        temperature=0.0
    )
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500)
    text_chunks = splitter.split_text(text)
    vector_store = FAISS.from_texts(
        text_chunks,
        embedding=GoogleGenerativeAIEmbeddings(
            google_api_key=google_api_key,
            model="models/embedding-001"
        )
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    prompt_template = PromptTemplate(
        input_variables=["context"],
        template = """
You are an expert ESG (Environmental, Social, Governance) analyst. Analyze the following company report text carefully.

Focus on extracting meaningful risks and opportunities under the following themes:

Environmental Risks:
- Carbon footprint
- Water usage
- Waste disposal
- Greenhouse gas emissions
- Impact on biodiversity
- Deforestation

Social Risks:
- Wage equality
- Workplace safety and conditions
- Supplier and vendor labor practices
- Human rights violations
- Diversity, equity, and inclusion (DEI)
- Data privacy and security

Governance Risks:
- Transparent communications
- ESG disclosures and reporting
- Board structure, independence, and diversity
- Corruption and fraud prevention
- Organizational integrity and ethics
- Executive compensation practices

---

Return your output STRICTLY in three clearly separated Markdown tables:

### Risk Analysis Table
| Risk Category | Summary of Risk | Potential Impact | Likelihood (Low/Medium/High) | Mitigation Strategy |
|---------------|-----------------|------------------|------------------------------|---------------------|
| ... | ... | ... | ... | ... |

### Positive Indicators Table
| Positive Factor | Current Status | Strategic Impact |
|-----------------|----------------|------------------|
| ... | ... | ... |

### Negative Indicators Table
| Negative Factor | Current Status | Strategic Impact |
|-----------------|----------------|------------------|
| ... | ... | ... |

---

**Important:**
- Only include material (significant) risks and indicators.
- Do not hallucinate any data not present in the text.
- Summarize clearly and professionally.
- Maintain consistency and structure across the tables.

---

Here is the extracted report text:
{context}
"""
    )
    qa_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt_template)
    relevant_docs = retriever.get_relevant_documents("Extract ESG risks and indicators")
    response = qa_chain.run(input_documents=relevant_docs, question="Extract ESG risks and indicators")
    return response

def split_response_to_dfs(response_text):
    def force_pipe_separation(text_block):
        fixed_lines = []
        for line in text_block.splitlines():
            if line.strip() and '|' not in line:
                line = re.sub(r' {2,}', '|', line.strip())
            fixed_lines.append(line)
        return "\n".join(fixed_lines)

    def extract_table(table_text):
        table_text = re.sub(r"\*\*.*?\*\*", "", table_text, flags=re.MULTILINE).strip()
        table_text = force_pipe_separation(table_text)
        table_lines = [line for line in table_text.splitlines() if "|" in line]
        if len(table_lines) > 1:
            cleaned_table = "\n".join(table_lines)
            df = pd.read_csv(io.StringIO(cleaned_table), sep="|", engine="python", skipinitialspace=True)
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            df.columns = df.columns.str.strip()
            df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            df = df.dropna(how="all")
            df = df[~df.isin(['---']).any(axis=1)]
            return df
        else:
            return pd.DataFrame()

    risk_start = response_text.find("Risk Analysis Table")
    positive_start = response_text.find("Positive Indicators Table")
    negative_start = response_text.find("Negative Indicators Table")

    risk_table = response_text[risk_start:positive_start].strip() if positive_start > risk_start else ""
    positive_table = response_text[positive_start:negative_start].strip() if negative_start > positive_start else ""
    negative_table = response_text[negative_start:].strip()

    risks_df = extract_table(risk_table)
    positive_df = extract_table(positive_table)
    negative_df = extract_table(negative_table)

    return risks_df, positive_df, negative_df

def calculate_esg_score(risks_df, positive_df, negative_df):
    score = 50
    explanation = []

    if not positive_df.empty:
        score += len(positive_df) * 2
        explanation.append(f"+{len(positive_df)*2} points for {len(positive_df)} positive ESG indicators.")

    if not negative_df.empty:
        score -= len(negative_df) * 5
        explanation.append(f"-{len(negative_df)*5} points penalty for {len(negative_df)} negative ESG indicators.")

    final_score = max(0, min(100, score))
    explanation.append(f"🌟 Final ESG Score: {final_score}/100.")

    return final_score, explanation

def save_as_pdf(risks_df, positive_df, negative_df, esg_score, explanation, filename: str):
    if not os.path.exists("Final_PDF"):
        os.makedirs("Final_PDF")

    pdf_path = os.path.join("Final_PDF", filename)
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    normal_style = styles['Normal']

    wrapped_style = ParagraphStyle(
        name='Wrapped',
        parent=normal_style,
        alignment=TA_LEFT,
        fontSize=8,
        leading=10
    )

    def create_wrapped_table(df, title, header_color):
        elements.append(Paragraph(title, styles['Heading2']))
        elements.append(Spacer(1, 8))
        if df.empty:
            elements.append(Paragraph("No data available.", normal_style))
            elements.append(Spacer(1, 12))
            return
        data = [list(df.columns)]
        for row in df.values:
            wrapped_row = []
            for cell in row:
                text = str(cell) if pd.notnull(cell) else ""
                wrapped_row.append(Paragraph(text, wrapped_style))
            data.append(wrapped_row)
        col_count = len(df.columns)
        table = Table(data, colWidths=[None] * col_count, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(header_color)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 20))

    elements.append(Paragraph("Extracted ESG Insights Report", styles['Title']))
    elements.append(Spacer(1, 20))
    create_wrapped_table(risks_df, "Risk Analysis Table", "#003366")
    create_wrapped_table(positive_df, "Positive Indicators Table", "#006400")
    create_wrapped_table(negative_df, "Negative Indicators Table", "#8B0000")

    elements.append(PageBreak())

    elements.append(Paragraph("🌟 Final ESG Score", styles['Heading2']))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(f"<b>{esg_score} / 100</b>", styles['Title']))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("📋 ESG Score Computation Explanation:", styles['Heading2']))
    elements.append(Spacer(1, 10))
    for exp in explanation:
        elements.append(Paragraph(f"• {exp}", normal_style))
        elements.append(Spacer(1, 8))

    doc.build(elements)

# --- Streamlit App ---

def main():
    st.set_page_config(page_title="🌿 ESG Insights Extractor", layout="wide")
    st.title("🌍 ESG Risk Analyzer")

    pdf_file = st.file_uploader("📂 Upload Annual Public Report or ESG Report:", type=["pdf"])

    if pdf_file and st.button("🔍 Analyze ESG Insights"):
        with st.spinner("Analyzing ESG Report... please wait 🚀"):
            report_text = extract_full_text_for_analysis(pdf_file)
            first_page_text = extract_text_from_pdf(pdf_file)

            company_name = detect_company_name(first_page_text, fallback_filename=os.path.splitext(pdf_file.name)[0])
            safe_company_name = company_name

            if not os.path.exists("Final_PDF"):
                os.makedirs("Final_PDF")

            cache_file_path = f"Final_PDF/cache_response_{safe_company_name}.txt"

            my_api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyBnsQBunwYh_IJsChlJP1BSmGcqat40wl8")

            if not os.path.exists(cache_file_path):
                raw_esg_risks = analyze_esg_risks(report_text, my_api_key)
                with open(cache_file_path, "w", encoding="utf-8") as f:
                    f.write(raw_esg_risks)
            else:
                with open(cache_file_path, "r", encoding="utf-8") as f:
                    raw_esg_risks = f.read()

            risks_df, positive_df, negative_df = split_response_to_dfs(raw_esg_risks)

            if not risks_df.empty:
                st.subheader("Extracted Risk Analysis Table:")
                st.dataframe(risks_df, use_container_width=True)

            if not positive_df.empty:
                st.subheader("Extracted Positive Indicators Table:")
                st.dataframe(positive_df, use_container_width=True)

            if not negative_df.empty:
                st.subheader("Extracted Negative Indicators Table:")
                st.dataframe(negative_df, use_container_width=True)

            if risks_df.empty and positive_df.empty and negative_df.empty:
                st.warning("⚠️ No ESG insights extracted.")
                return

            esg_score, explanation = calculate_esg_score(risks_df, positive_df, negative_df)

            final_pdf_filename = f"{safe_company_name}_esg_risk.pdf"

            save_as_pdf(risks_df, positive_df, negative_df, esg_score, explanation, filename=final_pdf_filename)
            st.success(f"✅ ESG Insights PDF (Score: {esg_score}/100) saved!")

            with open(os.path.join("Final_PDF", final_pdf_filename), "rb") as f:
                st.download_button(
                    label="📥 Download ESG Insights Report (PDF)",
                    data=f,
                    file_name=final_pdf_filename,
                    mime="application/pdf"
                )

if __name__ == "__main__":
    main()
