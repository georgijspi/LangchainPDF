from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI(model_name="gpt-3.5-turbo")


prompt = ChatPromptTemplate(
    input_variables=["content"],
    messages=[
        HumanMessagePromptTemplate.from_template(
            "{content}"
        ),
        # HumanMessagePromptTemplate(
        #     speaker="bot",
        #     message="You entered: {content}",
        # ),
    ],
)

chain = LLMChain(
    llm = chat,
    prompt = prompt,
    # output_key="chat",
)

# infinite chat loop
while True:
    content = input("Usage: type 'quit' to exit chat\n>> ")
    
    if content == "quit":
        break
    
    result = chain({"content": content})

    print(result["text"])

