from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from os import getenv

load_dotenv()
try:
    model = ChatGoogleGenerativeAI(model=getenv('MODEL', ''))
    res = model.invoke("say hello")
    print(res.content)
except Exception as e:
    print(f"Error: {e}")
