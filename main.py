from dotenv import load_dotenv
load_dotenv()

from langchain_core import __version__ as core_version
from langgraph.version import __version__ as lg_version
from langchain_google_genai import ChatGoogleGenerativeAI

print(f"langchain_core version {core_version}")
print(f"langgraph version {lg_version}")





def main():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    response = llm.invoke("Explain RAG in simple 50 words")
    print(response.content, response)


if __name__ == "__main__":
    main()
