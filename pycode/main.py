from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', default='return a list of numbers')
parser.add_argument('--language', default='python')
args = parser.parse_args()

load_dotenv()

llm = OpenAI(model_name="gpt-3.5-turbo")

code_prompt = PromptTemplate(
    input_variables = ["language", "task"],
    template = "Write a very short {language} function that will {task}. Do not include any trailing characeters, give the code and nothing else.",
)

testCode_prompt = PromptTemplate(
    input_variables = ["language", "code"],
    template = "Write a test for the following {language} code:\n{code}. Do not include any trailing characeters, give the code and nothing else.",
)

code_chain = LLMChain(
    llm = llm,
    prompt = code_prompt,
    output_key="code",
)

testCode_chain = LLMChain(
    llm = llm,
    prompt = testCode_prompt,
    output_key="testCode",
)

chain = SequentialChain(
    chains = [code_chain, testCode_chain],
    input_variables=["language", "task"],
    output_variables=["testCode", "code"],
    # output_key="seqChain",
)

result = chain({
    "language": args.language,
    "task": args.task
})

print("GENEREATED CODE:")
print(result["code"])

print("GENEREATED TEST:")
print(result["testCode"])
# if __name__ == "__main__":
#     pass