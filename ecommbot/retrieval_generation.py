# from langchain_core.documents import Documnet
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI
from ecombot.ingest import ingestdata
import os
azure_api=os.getenv("AZURE_OPENAI_KEY")
azure_endpoint=os.getenv("https://connection.openai.azure.com/")
azure_version="2024-02-01"

def generation(vstore):
    retriver=vstore.as_retriever(search_kwargs={"k":3})
    PRODUCT_BOT_TEMPLATE = """
    Your ecommercebot bot is an expert in product recommendations and customer queries.
    It analyzes product titles and reviews to provide accurate and helpful responses.
    Ensure your answers are relevant to the product context and refrain from straying off-topic.
    Your responses should be concise and informative.

    CONTEXT:
    {context}

    QUESTION: {question}

    YOUR ANSWER:
    
    """
    prompt = ChatPromptTemplate.from_template(PRODUCT_BOT_TEMPLATE)

    llm = AzureChatOpenAI(api_key=azure_api,api_version=azure_version,azure_endpoint=azure_endpoint,azure_deployment="gpt-4")

    chain = (
        {"context":retriver, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

if __name__=='__main__':
    vstore = ingestdata("done")
    chain  = generation(vstore)
    print(chain.invoke("can you tell me the best bluetooth buds?"))
