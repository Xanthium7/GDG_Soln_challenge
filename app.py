import os
from dotenv import load_dotenv

from groq import Groq

load_dotenv()

PROMPT = "You are an expert assistant who hels elderly ppl with documents and procedures in lesgl sectors like pachayath village offices banks etc.. This is pirmarily specific to keralal. an Elderly person will ask yyou a prompt about some clasrification he would require to help them identify their various requirements. return your rezponse ina simple straight forward way"

client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {"role": "system", "content": PROMPT},
        {
            "role": "user",
            "content": "Explain the importance of fast language models",
        }
    ],
    model="llama-3.3-70b-versatile",
)

print(chat_completion.choices[0].message.content)
