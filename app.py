import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

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

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="human_input")
])

# Replace LLMChain with pipe syntax
chain = chat_prompt | groq_llm


def process_user_query(user_input):
    # Replace run with invoke
    response = chain.invoke(
        {"human_input": [HumanMessage(content=user_input)]})
    # Extract content from the response
    return response.content


if __name__ == "__main__":
    user_query = "How can I apply for a new ration card"
    response = process_user_query(user_query)
    print(response)
