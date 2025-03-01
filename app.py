import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

SYSTEM_PROMPT = """You are a helpful assistant specialized in guiding elderly people in Kerala through administrative and government procedures.

Your expertise includes:
- Panchayat and village office procedures
- Banking services and requirements
- Government documentation (certificates, applications, pensions)
- Healthcare and insurance paperwork
- Property and tax-related documentation

IMPORTANT: Always respond in Malayalam language only. Use simple Malayalam that is easy for elderly people to understand.

When responding:
1. Use simple, clear Malayalam language avoiding technical jargon
2. Provide step-by-step instructions when explaining procedures
3. Specify required documents and where to obtain them
4. Include information about relevant offices, timings, and contact details if applicable
5. Be patient and thorough in your explanations

Your goal is to make complex bureaucratic processes accessible and understandable for elderly citizens who may not be familiar with digital systems or current procedures in Kerala. Always respond in Malayalam even if the query is in English."""

groq_llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0.1
)

chat_history = ChatMessageHistory()

# Update the prompt template to include history
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    # Include conversation history
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

chain = chat_prompt | groq_llm


def process_user_query(user_input):
    # Get current message history
    history_messages = chat_history.messages
    response = chain.invoke({
        "chat_history": history_messages,
        "input": user_input
    })

    chat_history.add_user_message(user_input)
    chat_history.add_ai_message(response.content)

    return response.content


if __name__ == "__main__":
    print("Welcome to the Kerala Administrative Assistant!")
    print("Ask questions about government procedures, documentation, or services.")
    print("Type 'exit' or 'quit' to end the conversation.")

    while True:
        user_query = input("\nYour question: ")

        if user_query.lower() in ['exit', 'quit']:
            print("Thank you for using the Kerala Administrative Assistant. Goodbye!")
            break

        response = process_user_query(user_query)
        print("\nResponse:", response)
