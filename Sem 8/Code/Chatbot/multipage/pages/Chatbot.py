# Chatbot 📃
import numpy as np
import time
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
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
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
 
# engine = pyttsx3.init()
# rate = engine.getProperty('rate')   # getting details of current speaking rate
# print (rate)                        #printing current voice rate
# engine.setProperty('rate',100)
 
 
# # Function to convert text to speech
# def text_to_speech(text):
#     print(text)
#     print("hereeee")
#     if engine._inLoop:
#         engine.endLoop()
#     try:
#         engine.say(text)
#         engine.runAndWait()
#         engine.stop()
#     except RuntimeError as e:
#         print("Error:", e)
 
# def text_to_speech_thread(text):
#     thread = threading.Thread(target=text_to_speech, args=(text,))
#     thread.start()
 
# vector_store = []
 
# st.set_page_config(
#     page_title="Chat with Multiple PDFs",
#     page_icon=":blue_book:",
# )
 
 
os.environ['GOOGLE_API_KEY'] = 'AIzaSyBnsQBunwYh_IJsChlJP1BSmGcqat40wl8'
# st.set_option('server.allow_dangerous_deserialization', True)
 
embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
 
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=5000)
    chunks = text_splitter.split_text(text)
    return chunks
 
 
# Function to save the vector store to a file
def save_vector_store(vector_store, filename='vector_store.pkl'):
    vector_store.save_local(filename)
 
# Function to load the vector store from a file
def load_vector_store(filename='vector_store.pkl'):
    if os.path.exists(filename):
        return FAISS.load_local(filename, embeddings, allow_dangerous_deserialization=True)
        # return FAISS.load_local(filename, embeddings)
    else:
        print("Failed to load vector store. File does not exist.")
        return None
 
# Function to create vector store from text chunks
def get_vector_store(text_chunks):
 
    # If pickle file doesn't exist, create vector store and save it
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    # Save vector store to pickle file
    save_vector_store(vector_store)
    print("Vector store created and saved to pickle file.")
    return vector_store
 
# Function to perform similarity search with score
# def similarity_search_with_score(vector_store, query):
#     results_with_scores = vector_store.similarity_search_with_score(query)
#     # results_with_scores_sorted = sorted(results_with_scores, key=lambda x: x[1])  # Sort based on score
#     for idx, (doc, score) in enumerate(results_with_scores):
#         print(score)
#         print(f"Rank: {idx + 1}, Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")
 
 
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
# def similarity_search_with_filter(vector_store, query, filter_metadata):
#     results = vector_store.similarity_search(query, filter=filter_metadata)
#     for doc in results:
#         print(f"Content: {doc.page_content}, Metadata: {doc.metadata}")
 
 
# Function to create conversational chain from vector store
 
 
 
# Function to create conversational chain from vector store
def get_conversational_chain(vector_store, prompt_template):

    prompt_template = """
        You are the most accurate model in the world with extensive knowledge of legal contracts and their terms. Answer the question as detailed as possible based on the provided context, ensuring all relevant details are included.


        Instructions:
        - Provide answers only in English.
        - Ensure clear and proper formatting.
        - If the answer is not in the provided context, respond with "Answer is not available in the context."
        - Include units for numerical data (e.g., $, million, billion).
        - If requested, provide the answer in a well-formatted table.
        - For bullet points, ensure each point starts on a new line.
        - For logical reasoning or open-ended questions, provide justified and logically aligned answers.

        Context:
        {context}

        Question:
        {question}

        Answer:

    """
    model_name = "gemini-1.5-flash-latest"
    llm = ChatGoogleGenerativeAI(model=model_name,temperature=0.0)
    # Load conversational chain with prompt
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PromptTemplate(template=prompt_template, input_variables=["context", "question"])}
    )
    return conversation_chain
 
# Global variable to store FAQ data
faq_data = {}
 
# Function to process uploaded PDFs
def process_pdf(pdf_path):
    prompt_template = """
   
    """
   
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
 
 
 
# Function to generate FAQs
def generate_faqs(pdf_name):
    faq_questions = {
        "Liquidity Risk": [
            "What are the company's current liquidity ratios (e.g., current ratio, quick ratio)?",
            "Does the company have sufficient cash flow to cover its short-term obligations?",
            "How efficiently does the company manage its cash conversion cycle, inventory, and receivables?",
        ],
        "Solvency Risk": [
            "What is the company's debt-to-equity ratio, and how has it changed over time?",
            "Is the company at risk of defaulting on its long-term debts?",
            "What is the company's interest coverage ratio (ability to cover interest payments)?",
        ],
        "Market Risk": [
            "How sensitive is the company to changes in interest rates, exchange rates, or commodity prices?",
            "Does the company hedge against foreign currency or commodity price fluctuations?",
            "What external economic factors are most likely to impact the company’s financial health?",
        ],
        "Credit Risk": [
            "What is the company’s credit rating, and how has it changed over time?",
            "What percentage of the company’s receivables is overdue or at risk of default?",
            "How effectively does the company manage credit risk through policies or credit insurance?",
        ]
    }

    global faq_data
    faq_data[pdf_name] = {}

    for category, questions in faq_questions.items():
        pdf_faq = {}
        for question in questions:
            time.sleep(5)  # Simulate delay for processing
            response = st.session_state.faq_conversation.invoke({'question': question})
            if response['chat_history']:
                answer = response['chat_history'][-1].content
                pdf_faq[question] = answer
            else:
                pdf_faq[question] = "No answer found"
        
        faq_data[pdf_name][category] = pdf_faq  # Store FAQs by category

 
def user_input(user_q):
    print("here")
    response = st.session_state.conversation.invoke({'question': user_q})
 
    st.session_state.chatHistory = response['chat_history']
    db = load_vector_store()
    answer = response['answer']
    # docs_and_scores = db.similarity_search_with_score(answer)
    # first_result = docs_and_scores[0]
    # document = first_result[0]
    # similarity_score = first_result[1]
    # print("similarity score")
    # print(similarity_score)
 
 
    # if 'similarity_scores' not in st.session_state:
    #     st.session_state.similarity_scores = []
       
    if 'messages' not in st.session_state:
        st.session_state.messages = []
 
    st.session_state.messages.append(answer)
 
    # if(similarity_score<=0.5): st.session_state.similarity_scores.append("High Similarity")
    # elif (similarity_score>0.5 and similarity_score<0.7): st.session_state.similarity_scores.append("Medium Similarity with the documents. Analysis given by the chatbot.")
    # else:  st.session_state.similarity_scores.append("Low Similarity")
 
    # page_number = document.metadata.get("page", None)
    # print("Page Number:", page_number)
   
 
 
    st.rerun()
 
 
 
 
# Function to search for PDFs using Google Custom Search API
def search_for_pdfs(query, num_results=5):
    search_url = "https://www.googleapis.com/customsearch/v1"
    cse_id = 'f79b6c79519ab4aeb'
    api_key = 'AIzaSyAd70UU-1uvFQS1TPBjWnsq_sdVhSyDOGg'
   
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
 
   
   
def save_to_csv(df, filename='data.csv'):
    df.to_csv(filename, index=False)
   
   
 
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
    "consideration": "Something of value exchanged for a promise or performance in a contract.",
    "term": "The duration or length of time that a contract is valid.",
    "party": "A person or entity involved in the contract.",
    "indemnity": "A promise to compensate for loss or damage.",
    "breach": "The failure to fulfill the terms of a contract without a legal excuse.",
    "remedy": "The action or procedure that can be taken to enforce a right or redress a wrong in case of a breach.",
    "jurisdiction": "The authority or power of a court to hear and decide a case.",
    "force majeure": "Circumstances beyond the control of the parties that prevent the fulfillment of a contract.",
    "severability": "The ability to separate provisions of a contract that are invalid or unenforceable from those that are valid and enforceable.",
    "confidentiality": "The obligation to keep information shared during the contract period private.",
    "termination": "The ending of a contract before its expiration date.",
    "amendment": "A change or modification to the terms of a contract.",
    "waiver": "The voluntary relinquishment or abandonment of a right or claim under a contract.",
    "governing law": "The laws of a particular jurisdiction that will govern the interpretation and enforcement of the contract.",
    "assignment": "The transfer of rights or obligations under a contract to another party.",
    "representation and warranty": "Statements made by one party in a contract regarding certain facts or circumstances.",
    "dispute resolution": "The process for resolving conflicts or disagreements arising from the contract.",
    "notices": "Formal communications sent between parties to the contract.",
    "force majeure": "An unforeseeable circumstance that prevents someone from fulfilling a contract.",
    "entire agreement": "A clause stating that the contract represents the complete understanding between the parties and supersedes any prior agreements.",
    "arbitration clause": "A provision in a contract that requires the parties to resolve disputes through arbitration rather than litigation.",
    "choice of law clause": "A provision in a contract that specifies which jurisdiction's laws will govern the interpretation and enforcement of the contract.",
    "forum selection clause": "A provision in a contract that specifies the jurisdiction or venue where disputes arising from the contract will be litigated.",
    "non-disclosure agreement (NDA)": "A contract in which the parties agree not to disclose confidential information shared between them.",
    "non-compete clause": "A provision in a contract that restricts one party from competing with another party within a certain geographic area or industry for a specified period of time.",
    "severability clause": "A provision in a contract that states that if one part of the contract is found to be invalid or unenforceable, the remaining parts of the contract will still be valid and enforceable.",
    "merger clause": "A provision in a contract that states that the written contract represents the complete agreement between the parties and supersedes any prior agreements or understandings, whether written or oral.",
    "integration clause": "Similar to a merger clause, it states that the written contract represents the complete agreement between the parties.",
    "entire agreement clause": "Similar to a merger clause, it states that the written contract represents the complete agreement between the parties.",
    "confidentiality clause": "A provision in a contract that requires one or both parties to keep certain information confidential.",
    "termination clause": "A provision in a contract that specifies the conditions under which the contract can be terminated.",
    "savings clause": "A provision in a contract that preserves certain rights or obligations even if other parts of the contract are found to be invalid or unenforceable.",
    "survival clause": "A provision in a contract that specifies which terms and conditions will continue to apply even after the contract has been terminated or expired.",
    "indemnification clause": "A provision in a contract that requires one party to compensate the other party for certain losses or liabilities.",
    "limitation of liability clause": "A provision in a contract that limits the amount of damages that one party can be held liable for in case of breach or negligence.",
    "assignment clause": "A provision in a contract that allows one party to transfer its rights or obligations under the contract to another party.",
    "notices clause": "A provision in a contract that specifies how and when formal communications between the parties will be delivered or received.",
    "force majeure clause": "A provision in a contract that excuses one or both parties from performance if certain unforeseen events occur, such as natural disasters or acts of war."
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
   
    st.header(" Risk Analysis Chatbot 📃")

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
    if "faq_data" not in st.session_state:
        st.session_state.faq_data = {}
 
 
 
    # Sidebar for settings
    st.sidebar.title("Settings")
    st.sidebar.subheader("Upload and Process PDFs")
 
 
    # Upload PDFs
    pdf_docs = st.sidebar.file_uploader("Upload your PDF Files", accept_multiple_files=True)
    # Process uploaded PDFs
    if pdf_docs and st.sidebar.button("Process PDFs"):
        with st.spinner("Processing PDFs"):
            for pdf in pdf_docs:
                pdf_name = os.path.basename(pdf.name)
                st.write(process_pdf(pdf))
               
                generate_faqs(pdf_name)
                st.session_state.faq_data = faq_data
               
 
   # Display FAQs if available
    for pdf_name, categories in st.session_state.faq_data.items():
        st.markdown('<hr style="border-top: 1px solid #555555;">', unsafe_allow_html=True)
        st.markdown(f"<h4>{pdf_name}</h4>", unsafe_allow_html=True)  # Display PDF name
        
        for category, questions_and_answers in categories.items():
            st.markdown(f"<h3>{category}</h3>", unsafe_allow_html=True)  # Display category name
            
            for question, answer in questions_and_answers.items():
                if "Q:" in answer:
                    # Handle responses containing multiple QA pairs
                    qa_pairs = answer.split("Q:")
                    for qa_pair in qa_pairs[1:]:
                        q_a_pair = qa_pair.split("A:")
                        if len(q_a_pair) >= 2:
                            question = q_a_pair[0].strip()
                            answer = q_a_pair[1].strip()
                            st.markdown(f'<p style="font-weight: bold;">Q: {question}</p>', unsafe_allow_html=True)
                            st.markdown(f'<p>A: {answer}</p>', unsafe_allow_html=True)
                           
                else:
                    
                    if "experienced legal assistant" in question or "Tabular format" in question or "|" in answer or "---" in answer or "|---|" in answer:
                        print(answer)
                        # If the response contains table formatting, format it into a proper table

                        rows = [row.split("|") for row in answer.split("\n") if row.strip()]
                        
                        # Remove empty first and last columns
                        if rows and len(rows[0]) > 2:
                            rows = [row[1:-1] for row in rows]

                            # Remove rows filled with '--- --- ---'
                            character = '-'
                            rows = [row for row in rows if not all(cell.strip() == character * len(cell.strip()) for cell in row)]

                            if len(rows) > 1 and all(len(row) == len(rows[0]) for row in rows):
                                # Check for empty column names and replace them with a default name
                                columns = [col.strip() if col.strip() else f"Column {i+1}" for i, col in enumerate(rows[0])]
                                df = pd.DataFrame(rows[1:], columns=columns)
                                
                                # Display DataFrame with text wrap for cells
                                st.write(df.style.set_properties(**{'white-space': 'pre-wrap'}))
                                
                                save_to_csv(df)
                        else:
                            # Check if '<br>' tags are present in the answer
                            if '<br>' in answer:
                                print("HERE")
                                # If '<br>' tags are present, replace them with line breaks and display as HTML
                                formatted_answer = answer.replace('<br>', ' ')
                                st.markdown(f'<p style="font-family: Arial, sans-serif; font-size: 16px; text-align: justify;">A: {formatted_answer}</p>', unsafe_allow_html=True)
                            else:
                                # Display as plain text if no '<br>' tags are present
                                st.markdown(f'<p style="font-family: Arial, sans-serif; font-size: 16px; font-weight: bold;">Q: {question}</p>', unsafe_allow_html=True)
                                st.markdown(f'<p style="font-family: Arial, sans-serif; font-size: 16px; text-align: justify;">A: {answer}</p>', unsafe_allow_html=True)
                    else:
                        # Display non-table responses as plain text
                        st.markdown(f'<p style="font-family: Arial, sans-serif; font-size: 16px; font-weight: bold;">Q: {question}</p>', unsafe_allow_html=True)
                        st.markdown(f'<p style="font-family: Arial, sans-serif; font-size: 16px; text-align: justify;">A: {answer}</p>', unsafe_allow_html=True)
                   
   
            
               
 
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
                       
                        # if len(st.session_state.similarity_scores) > idx:
                        #     st.write(f"<p style='color: #3366ff; font-size: 17px;'>Similarity Score: {st.session_state.similarity_scores[idx]}</p>", unsafe_allow_html=True)
 
                        # idx= idx+1
                    
                        st.markdown('<hr style="border-top: 1px solid #555555;">', unsafe_allow_html=True)
 
                       
                    else:
                        col1, col2 = st.columns([1, 8])
                        with col1:
                            st.image(chatgpt_image)
                        with col2:
                           
                            # st.write(message.content)
                            response_with_underline = underline_financial_terms(message.content)
                            st.markdown(f'<p style="font-family: Arial, sans-serif; font-size: 16px; text-align: justify;">{response_with_underline}</p>', unsafe_allow_html=True)
                        # if len(st.session_state.similarity_scores) > idx:
                        #     st.write(f"<p style='color: #3366ff; font-size: 17px;'>Similarity Score: {st.session_state.similarity_scores[idx]}</p>", unsafe_allow_html=True)
 
 
                        # idx= idx+1
                        st.markdown('<hr style="border-top: 1px solid #555555;">', unsafe_allow_html=True)
 
                else:
                    col1, col2 = st.columns([1, 8])
                    with col1:
                        st.image(chatgpt_image)
                    with col2:
                       
                        response_with_underline = underline_financial_terms(message.content)
                        st.markdown(f'<p style="font-family: Arial, sans-serif; font-size: 16px; text-align: justify;">{response_with_underline}</p>', unsafe_allow_html=True)
                    # if len(st.session_state.similarity_scores) > idx:
                    #     st.write(f"<p style='color: #3366ff; font-size: 17px;'>Similarity Score: {st.session_state.similarity_scores[idx]}</p>", unsafe_allow_html=True)
 
                    # idx=idx+1
                    # if st.button(f"Text to Speech {i}"):  
                    #     print("CLICKED")
                        # text_to_speech_thread(st.session_state.messages[i//2])
                    st.markdown('<hr style="border-top: 1px solid #555555;">', unsafe_allow_html=True)
                   
    user_question = st.text_input("Ask a Question from the PDF Files", key="pdf_question")
 
    if st.button("Get Response") and user_question :
        user_q =  user_question
        user_input(user_q)
       
   
 
 
if __name__ == "__main__":
    main()