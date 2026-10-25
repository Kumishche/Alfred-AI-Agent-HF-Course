from langchain_community.retrievers import BM25Retriever
from langchain_core.tools import Tool, tool
from langchain_openai import ChatOpenAI

import os
import datasets
from langchain_core.documents import Document



guest_dataset = datasets.load_dataset('agents-course/unit3-invitees', split='train')

docs = [
    Document(
        page_content='\n'.join([
            f"Name: {guest['name']}",
            f"Relation: {guest['relation']}",
            f"Description: {guest['description']}",
            f"Email: {guest['email']}"
        ]),
        metadata={'name': guest['name']}
    )
    for guest in guest_dataset
]

bm25_retriever = BM25Retriever.from_documents(docs)


def extract_text(query: str) -> str:
  """Retrieves detailed information about gala guests based on their name or relation."""
  results = bm25_retriever.invoke(query)

  if results:
    return '\n\n'.join([doc.page_content for doc in results[:3]])
  else:
    return 'No matching guest information found.'

guest_info_tool = Tool(
    name='guest_info_retriever',
    description='Retrieves detailed information about gala guests based on their name or relation',
    func=extract_text
)


model = ChatOpenAI(
    model=os.getenv('LLM_MODEL'),
    base_url=os.getenv('BASE_URL'),
    api_key=os.getenv('API_KEY')
)


@tool
def draft_conversation(data: str) -> str:
  """Returns draft for starting conversation with a requested guest"""
  try:
    messages=[
        {
            'role': 'system',
            'content': 'You are an expert in networking. Your goal is to make such a strong and harmonious conversation starter based on the information given from the user.',
        },
        {
            'role': 'user',
            'content': data,
        },
      ]
    response = model.invoke(messages)
    return response.content
  except Exception as e:
    return f'Error: {e}'
