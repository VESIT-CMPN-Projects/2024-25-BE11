# Frequently Asked Questions 🤔
import streamlit as st

def display_faq_data(faq_data):
    st.title("Frequently Asked Questions")
    
    # Group questions by category
    categories = {}
    for category, questions_answers in faq_data.items():
        # Display category as header
        st.header(category)
        # Iterate through the questions and answers in each category
        for question, answer in questions_answers:
            # Display question
            st.subheader(f"Q: {question}")
            # Display answer
            st.write(f"A: {answer}")
            # Add a horizontal line between each question-answer pair
            st.markdown("---")

def main():
    # Retrieve faq_data from session_state
    faq_data = st.session_state.faq_data
    
    # Check if faq_data exists
    if faq_data:
        display_faq_data(faq_data)
    else:
        st.error("FAQ data not found. Please go back to the previous page and generate the FAQ data.")

if __name__ == "__main__":
    main()