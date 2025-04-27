import numpy as np
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
import os
import requests
from urllib.parse import urlparse
import pickle
import pandas as pd
from PIL import Image
import numpy as np
import joblib
import speech_recognition as sr
import pyttsx3
import threading


# Initialize text-to-speech engine

engine = pyttsx3.init()
rate = engine.getProperty('rate')   # getting details of current speaking rate
print (rate)                        #printing current voice rate
engine.setProperty('rate',100)


# Function to convert text to speech
def text_to_speech(text):
    print(text)
    print("hereeee")
    if engine._inLoop:
        engine.endLoop()
    try:
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except RuntimeError as e:
        print("Error:", e)

def text_to_speech_thread(text):
    thread = threading.Thread(target=text_to_speech, args=(text,))
    thread.start()

vector_store = []

st.set_page_config(
    page_title="Chat with Multiple PDFs",
    page_icon=":blue_book:",
)


os.environ['GOOGLE_API_KEY'] = 'AIzaSyBnsQBunwYh_IJsChlJP1BSmGcqat40wl8'
# st.set_option('server.allow_dangerous_deserialization', True)

embeddings = GooglePalmEmbeddings()


# Function to get text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


# Function to save the vector store to a file
def save_vector_store(vector_store, filename='vector_store.pkl'):
    vector_store.save_local(filename)

# Function to load the vector store from a file
def load_vector_store(filename='vector_store.pkl'):
    if os.path.exists(filename):
        # return FAISS.load_local(filename, embeddings, allow_dangerous_deserialization=True)
        return FAISS.load_local(filename, embeddings)
    else:
        print("Failed to load vector store. File does not exist.")
        return None

# Function to create vector store from text chunks
def get_vector_store(text_chunks):

    
    # If pickle file doesn't exist, create vector store and save it
    embeddings = GooglePalmEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    # Save vector store to pickle file
    save_vector_store(vector_store)
    print("Vector store created and saved to pickle file.")
    return vector_store

# Function to perform similarity search with score
def similarity_search_with_score(vector_store, query):
    results_with_scores = vector_store.similarity_search_with_score(query)
    # results_with_scores_sorted = sorted(results_with_scores, key=lambda x: x[1])  # Sort based on score
    for idx, (doc, score) in enumerate(results_with_scores):
        print(score)
        print(f"Rank: {idx + 1}, Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")


# Function to serialize the FAISS index to bytes
def serialize_to_bytes(vector_store):
    return vector_store.serialize_to_bytes()

# Function to deserialize the FAISS index from bytes
def deserialize_from_bytes(serialized_bytes, embeddings):
    return FAISS.deserialize_from_bytes(embeddings=embeddings, serialized=serialized_bytes)

# Function to merge two FAISS vector stores
def merge_vector_stores(vector_store1, vector_store2):
    vector_store1.merge_from(vector_store2)

# Function to perform similarity search with filtering
def similarity_search_with_filter(vector_store, query, filter_metadata):
    results = vector_store.similarity_search(query, filter=filter_metadata)
    for doc in results:
        print(f"Content: {doc.page_content}, Metadata: {doc.metadata}")


# Function to create conversational chain from vector store
prompt_template = """
Make sure to provide all the details, also make sure that the formatting of the answer is nice, if the answer is not in
provided context just say, "answer is not available in the context", don't provide the wrong answer, if the answer contains numerical data, then also give the units like $ or million or billion based on what is given in the context, if the user requests the answer in tabular format, please provide the answer accordingly in a perfectly formatted table, If the question is of logical reasoning and open-ended, please give logical answers, also justify the alignment of the answers Give all the answer in proper formatting, if the user asks to answer in bullet points, then answer in bullet points ensuring each point starts from a new line.\n\n
Context:\n {context}?\n
Question: \n{question}\n

Answer:
"""


# Function to create conversational chain from vector store
def get_conversational_chain(vector_store, prompt_template):
    llm = GooglePalm()
    # Load conversational chain with prompt
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PromptTemplate(template=prompt_template, input_variables=["context", "question"])}
    )
    return conversation_chain


# Function to handle user input


def user_input(user_q):
    print("here")
    response = st.session_state.conversation({'question': user_q})

    st.session_state.chatHistory = response['chat_history']
    db = load_vector_store()
    answer = response['answer']
    docs_and_scores = db.similarity_search_with_score(answer)
    first_result = docs_and_scores[0]
    document = first_result[0]
    similarity_score = first_result[1]
    print("similarity score")
    print(similarity_score)


    if 'similarity_scores' not in st.session_state:
        st.session_state.similarity_scores = []
        
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    st.session_state.messages.append(answer)

    if(similarity_score<=0.5): st.session_state.similarity_scores.append("High Similarity")
    elif (similarity_score>0.5 and similarity_score<0.7): st.session_state.similarity_scores.append("Medium Similarity with the documents. Analysis given by the chatbot.")
    else:  st.session_state.similarity_scores.append("Low Similarity")

    page_number = document.metadata.get("page", None)
    print("Page Number:", page_number)
   


    st.rerun()


# Function to process uploaded PDFs
def process_pdf(pdf_path):
    # Clear the contents of the "User_Reports" folder
    
    main_text = ""
    with pdf_path as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            main_text += page.extract_text()
        text_chunks = get_text_chunks(main_text)
        vector_store = get_vector_store(text_chunks)
        st.session_state.conversation = get_conversational_chain(vector_store, prompt_template)
        st.session_state.faq_conversation = get_conversational_chain(vector_store, prompt_template)
        
        # Save the processed PDF in the "User_Reports" folder
        folder_path = "User_Reports"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Save the new file
        file_name = os.path.basename(pdf_path.name)
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, "wb") as file:
            pdf_path.seek(0)  # Move cursor to the beginning of the file
            file.write(pdf_path.read())
    
    return f"Processed PDF: {os.path.basename(pdf_path.name)}"


# Function to search for PDFs using Google Custom Search API
def search_for_pdfs(query, num_results=5):
    search_url = "https://www.googleapis.com/customsearch/v1"
    cse_id = '552cd3984b4504de8'
    api_key = 'AIzaSyBOGwNOY9zS2exJKPbDl0cVmjePAVkYTZQ'
    
    params = {
        'q': f"{query} annual public report filetype:pdf",
        'num': num_results,
        'cx': cse_id,
        'key': api_key,
    }
    
    response = requests.get(search_url, params=params)
    data = response.json()

    pdf_results = []
    if 'items' in data:
        for item in data['items']:
            pdf_url = item['link']
            pdf_name = os.path.basename(urlparse(pdf_url).path)
            pdf_results.append((pdf_name, pdf_url))

    return pdf_results


def save_processed_data(text_chunks):
    with open('processed_data.pkl', 'wb') as f:
        pickle.dump(text_chunks, f)


# Function to load processed text data
def load_processed_data():
    if os.path.exists('processed_data.pkl'):
        with open('processed_data.pkl', 'rb') as f:
            return pickle.load(f)
    else:
        return None
  
    
import plotly.graph_objects as go

def save_to_csv(df, filename='data.csv'):
    df.to_csv(filename, index=False)
   
    
def generate_graph_from_csv():
    # Read the CSV file
    df = pd.read_csv("data.csv")

    # Set the keywords as the index
    df.set_index(df.columns[0], inplace=True)

    # Convert string representations of monetary values to numeric values
    for col in df.columns:
        if df[col].dtype == object:  # Check if column dtype is object (string)
            # Replace dollar signs, commas, and percentage signs with empty string
            df[col] = df[col].str.replace('[\$,%,million,billion,M,B,m,b,Million,Billion]', '', regex=True)
            # Replace '---' with NaN (Not a Number)
            df[col] = df[col].replace('[---, ------, --, -, ----, -----]', np.nan)
            # Convert to float
            df[col] = df[col].astype(float)

    # Plotting the graph using Plotly
    fig = go.Figure()

    for column in df.columns:
        fig.add_trace(go.Bar(
            x=df.index,
            y=df[column],
            name=column,
            hovertemplate='<b>%{x}</b><br>%{customdata}',
            customdata=df[column].apply(lambda x: f"${x:,.2f}" if not np.isnan(x) else "Not available"),
        ))

    # Update layout to keep y-axis empty and show hover information
    fig.update_layout(
        barmode='group',
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
        xaxis_title='Fiscal Year',
        yaxis=dict(
            title='',
            showticklabels=False
        ),
        legend=dict(
            title='Metrics',
            font=dict(
                size=12
            )
        )
    )

    # Displaying the graph
    st.plotly_chart(fig)


# Function to load and resize images
def load_image(image_path, size=(50,50)):
    image = Image.open(image_path)
    image = image.resize(size)
    return image


# Load images
human_image = load_image("human_icon.png", size=(100,100))
chatgpt_image = load_image("bot.png", size=(100,100))


def clear_user_reports_folder():
    folder_path = "User_Reports"
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            os.remove(file_path)



# List of financial terms and their meanings
financial_terms = {
    "revenue": "Total income generated by the company.",
    "profit": "Income remaining after all expenses have been deducted.",
    "assets": "Resources owned by the company.",
    "liabilities": "Debts or obligations of the company.",
    "equity": "The value of the company's assets minus its liabilities.",
    "gross profit": "Revenue minus the cost of goods sold.",
    "net income": "The total amount of revenue minus expenses, interest, and taxes.",
    "depreciation": "The decrease in the value of an asset over time.",
    "amortization": "The process of spreading the cost of an intangible asset over its useful life.",
    "cash flow": "The movement of money into or out of a business.",
    "EBITDA": "Earnings before interest, taxes, depreciation, and amortization.",
    "balance sheet": "A financial statement that summarizes a company's assets, liabilities, and equity.",
    "income statement": "A financial statement that shows a company's revenues and expenses over a period of time.",
    "cash flow statement": "A financial statement that shows how changes in balance sheet accounts and income affect cash and cash equivalents.",
    "working capital": "The difference between current assets and current liabilities.",
    "cost of goods sold": "The direct costs attributable to the production of goods sold by a company.",
    "capital expenditure": "Money spent by a business or organization on acquiring or maintaining fixed assets.",
    "operating income": "The amount of profit realized from a business's operations after deducting operating expenses.",
    "interest expense": "The cost of borrowing money.",
    "dividend": "A payment made by a corporation to its shareholders, usually as a distribution of profits.",
    "retained earnings": "The portion of a company's profits that are retained and reinvested in the business.",
    "working capital ratio": "A measure of a company's liquidity and its ability to meet short-term obligations.",
    "current ratio": "A liquidity ratio that measures a company's ability to pay short-term obligations or debts.",
    "quick ratio": "A liquidity ratio that measures a company's ability to use its quick assets to pay off its current liabilities.",
    "debt-to-equity ratio": "A financial ratio that indicates the proportion of equity and debt used by a company to finance its assets.",
    "return on investment (ROI)": "A performance measure used to evaluate the efficiency or profitability of an investment.",
    "return on assets (ROA)": "A ratio that indicates how profitable a company is relative to its total assets.",
    "return on equity (ROE)": "A measure of a company's profitability that compares net income to shareholders' equity.",
    "corporate governance": "The system of rules, practices, and processes by which a company is directed and controlled.",
    "board of directors": "A group of individuals elected to represent shareholders' interests and oversee the management of a corporation.",
    "corporate social responsibility (CSR)": "The practice of operating a business in a manner that benefits society.",
    "risk management": "The process of identifying, assessing, and prioritizing risks followed by coordinated and economical application of resources to minimize, monitor, and control the probability and/or impact of unfortunate events.",
    "risk assessment": "The process of evaluating potential risks to an organization's assets, earning capacity, or overall business value.",
    "risk mitigation": "The process of implementing measures to reduce the likelihood or impact of risks.",
    "business continuity planning": "The process of creating systems of prevention and recovery to deal with potential threats to a company.",
    "compliance": "Adhering to laws, regulations, standards, and codes of conduct applicable to an organization.",
    "internal controls": "Processes, policies, and procedures implemented by an organization to ensure the integrity of financial and operational information, promote accountability, and prevent fraud.",
    "audit": "A systematic review or examination of records, accounts, and transactions to ensure accuracy and compliance with relevant laws and regulations.",
    # Add more terms as needed
}


# Function to check for financial terms in the response
def underline_financial_terms(response):
    for term, meaning in financial_terms.items():
        if term in response:
            response = response.replace(term, f'<abbr title="{meaning}">{term}</abbr>')
    return response



# Main function
def main():
    
    st.header("Chat with Annual Public Reports 💲")


    # Check if files have been deleted on app start
    if "files_deleted" not in st.session_state:
        st.session_state.files_deleted = False

    # If files haven't been deleted yet, delete them now
    if not st.session_state.files_deleted:
        clear_user_reports_folder()
        st.session_state.files_deleted = True

    # Check session states
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if 'chatHistory' not in st.session_state:
        st.session_state.chatHistory = None


    # Sidebar for settings
    st.sidebar.title("Settings")
    st.sidebar.subheader("Upload and Process PDFs")


    # Upload PDFs
    pdf_docs = st.sidebar.file_uploader("Upload your PDF Files", accept_multiple_files=True)
    faq_responses = {}
    # Process uploaded PDFs
    if pdf_docs and st.sidebar.button("Process PDFs"):
        with st.spinner("Processing PDFs"):
            for pdf in pdf_docs:
                st.write(process_pdf(pdf))
        with st.spinner("Generating FAQ's"):
            st.session_state.faq_displayed = True
            st.subheader("Here some FAQ's")
            faq_questions = {
                "Financial Performance": ["What were the total revenues and net profits for the year?", 
                                        "How did the company perform financially in the last fiscal year?",
                                        "Were there any significant changes in the company's financial metrics compared to the previous year?"],
                # "Operational Highlights": ["Can you provide key operational highlights mentioned in the report?",
                #                             "Were there any notable achievements or challenges faced by the company during the year?",
                #                             "How has the company expanded or changed its operations?"],
                # "Risk Mitigation": ["What risks and challenges does the company highlight in the report?",
                #                      "How does the company plan to mitigate potential risks or challenges?"],
                #  "Strategic Goals": ["What are the company's plans and strategic goals for the upcoming fiscal year?",
                #                      "How does the company plan to stay competitive in its industry",
                #                      ],
                #  "Corporate Governance:": ["How does the company approach corporate governance, and are there any changes or updates mentioned?",
                #                      "What information is provided about the composition of the board of directors and executive compensation?"],
                #  "Management Discussion and Analysis (MD&A)" :["What insights does the Management Discussion and Analysis (MD&A) section provide regarding the company's performance?"
                #                      "Are there any specific factors that management believes influenced financial results?"],
                #  "Sustainability Initiatives": ["Can you provide details about the company's sustainability initiatives?",
                #                                 "How does the company address sustainability and social responsibility in its operations?"]
            }
            faq_data = {}

            # Display FAQ questions and corresponding chatbot answers
            for category, questions in faq_questions.items():
                st.subheader(category)
                category_data = [] 
                for question in questions:
                    expander = st.expander(f"Q: {question}")
                    with expander:
                        response = st.session_state.faq_conversation({'question': question})
                        if response['chat_history']:
                            answer = response['chat_history'][-1].content
                            st.markdown(f'<p style="font-family: Arial, sans-serif; font-size: 16px; text-align: justify;">A: {answer}</p>',unsafe_allow_html=True)
                            category_data.append((question, answer))
                        else:
                            st.write("A: No answer found")
                faq_data[category] = category_data
                            
            st.session_state.faq_data = faq_data
            st.session_state.faq_conversation = None


    # Search bar for PDFs
    company_name = st.sidebar.text_input("Enter Company Name", key="company_name")
    year = st.sidebar.text_input("Enter Year", key="year")
    search_button_clicked = st.sidebar.button("Search")


    # Search for PDFs based on company name and year
    if search_button_clicked:
        if company_name and year:
            search_results = search_for_pdfs(company_name + " " + year)
            if search_results:
                st.sidebar.subheader("Search Results:")
                for pdf_name, pdf_url in search_results:
                    st.sidebar.markdown(
                        f'<a href="{pdf_url}" target="_blank" style="display: block; padding: 10px; margin-bottom: 10px; border-radius: 5px; border: 1px solid #ccc; text-decoration: none; color: #333; background-color: #f9f9f9;">{pdf_name}</a>',
                        unsafe_allow_html=True
                    )
            else:
                st.sidebar.write("No results found.")
        else:
            st.sidebar.write("Please enter both company name and year.")


    # if 'question_submitted' not in st.session_state:
    #     st.session_state.question_submitted = False
# Handle user input


    if st.session_state.chatHistory:
        idx = 0
        for i, message in enumerate(st.session_state.chatHistory):
            if i % 2 == 0:
                col1, col2 = st.columns([1, 8])
                with col1:
                    st.image(human_image, width=40)
                with col2:
                    st.markdown(f'<p style="font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; text-align: left;">{message.content}</p>', unsafe_allow_html=True)
            else:
                if "|" in message.content and "---" in message.content:
                    # If the response contains table formatting, format it into a proper table
                    rows = [row.split("|") for row in message.content.split("\n") if row.strip()]
                    
                    # Remove empty first and last columns
                    if rows and len(rows[0]) > 2:
                        rows = [row[1:-1] for row in rows]

                    # Remove rows filled with '--- --- ---'
                    character = '-'

                    # Filter out rows where all cells contain only the specified character
                    rows = [row for row in rows if not all(cell.strip() == character * len(cell.strip()) for cell in row)]

                    if len(rows) > 1 and all(len(row) == len(rows[0]) for row in rows):
                        # Check for empty column names and replace them with a default name
                        columns = [col.strip() if col.strip() else f"Column {i+1}" for i, col in enumerate(rows[0])]
                        df = pd.DataFrame(rows[1:], columns=columns)
                        st.write(df)
                    
                        
                        save_to_csv(df)
                        generate_graph_from_csv()
                        if len(st.session_state.similarity_scores) > idx:
                            st.write(f"<p style='color: #3366ff; font-size: 17px;'>Similarity Score: {st.session_state.similarity_scores[idx]}</p>", unsafe_allow_html=True)

                        idx= idx+1
                        st.markdown('<hr style="border-top: 1px solid #555555;">', unsafe_allow_html=True)

                        
                    else:
                        col1, col2 = st.columns([1, 8])
                        with col1:
                            st.image(chatgpt_image)
                        with col2:
                            
                            # st.write(message.content)
                            response_with_underline = underline_financial_terms(message.content)
                            st.markdown(f'<p style="font-family: Arial, sans-serif; font-size: 16px; text-align: justify;">{response_with_underline}</p>', unsafe_allow_html=True)
                        if len(st.session_state.similarity_scores) > idx:
                            st.write(f"<p style='color: #3366ff; font-size: 17px;'>Similarity Score: {st.session_state.similarity_scores[idx]}</p>", unsafe_allow_html=True)


                        idx= idx+1
                        st.markdown('<hr style="border-top: 1px solid #555555;">', unsafe_allow_html=True)

                else:
                    col1, col2 = st.columns([1, 8])
                    with col1:
                        st.image(chatgpt_image)
                    with col2:
                        
                        response_with_underline = underline_financial_terms(message.content)
                        st.markdown(f'<p style="font-family: Arial, sans-serif; font-size: 16px; text-align: justify;">{response_with_underline}</p>', unsafe_allow_html=True)
                    if len(st.session_state.similarity_scores) > idx:
                        st.write(f"<p style='color: #3366ff; font-size: 17px;'>Similarity Score: {st.session_state.similarity_scores[idx]}</p>", unsafe_allow_html=True)

                    idx=idx+1
                    if st.button(f"Text to Speech {i}"):  
                        print("CLICKED")
                        text_to_speech_thread(st.session_state.messages[i//2])
                    st.markdown('<hr style="border-top: 1px solid #555555;">', unsafe_allow_html=True)
                    
    user_question = st.text_input("Ask a Question from the PDF Files", key="pdf_question")

    if st.button("Get Response") and user_question :
        user_q =  user_question
        user_input(user_q)
        
    


if __name__ == "__main__":
    main()