import os
from dotenv import load_dotenv
import uuid
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory

# Load environment variables
load_dotenv()

# FastAPI app
app = FastAPI(title="Kerala Administrative Assistant API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# System prompt
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
3. ALWAYS list required documents in a separate section with this exact heading: "ആവശ്യമായ രേഖകൾ:" (Required Documents:)
4. Format each document as a bullet point using "•" symbols
5. Include keywords "രേഖകൾ" or "ഡോക്യുമെന്റ്" in the document section
6. Be patient and thorough in your explanations

For example, when listing documents, format them like this:

ആവശ്യമായ രേഖകൾ:
• ആധാർ കാർഡ്
• റേഷൻ കാർഡ്
• വോട്ടർ ഐഡി

Your goal is to make complex bureaucratic processes accessible and understandable for elderly citizens who may not be familiar with digital systems or current procedures in Kerala. Always respond in Malayalam even if the query is in English."""

# Initialize LLM
groq_llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0.1
)

# Create prompt template
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# Create chain
chain = chat_prompt | groq_llm

# Session management - store chat histories
sessions: Dict[str, ChatMessageHistory] = {}

# Pydantic models


class Message(BaseModel):
    content: str


class ChatResponse(BaseModel):
    session_id: str
    response: str


class ChatSession(BaseModel):
    session_id: str
    messages: List[dict]  # List of messages with role and content

# Helper function to get chat history for a session


def get_session_history(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]


@app.post("/chat/new", response_model=ChatSession)
async def create_chat_session():
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    sessions[session_id] = ChatMessageHistory()
    return {"session_id": session_id, "messages": []}


@app.post("/chat/{session_id}/message", response_model=ChatResponse)
async def send_message(session_id: str, message: Message):
    """Send a message to the chatbot and get a response"""
    try:
        chat_history = get_session_history(session_id)

        # Get current messages
        history_messages = chat_history.messages

        # Process the message
        response = chain.invoke({
            "chat_history": history_messages,
            "input": message.content
        })

        # Update chat history
        chat_history.add_user_message(message.content)
        chat_history.add_ai_message(response.content)

        return {"session_id": session_id, "response": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/{session_id}", response_model=ChatSession)
async def get_chat_session(session_id: str):
    """Get the chat history for a session"""
    try:
        chat_history = get_session_history(session_id)
        messages = []

        for msg in chat_history.messages:
            role = "user" if msg.type == "human" else "assistant"
            messages.append({"role": role, "content": msg.content})

        return {"session_id": session_id, "messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
