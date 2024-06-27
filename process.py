
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import warnings
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_google_genai import GoogleGenerativeAI

output_parser = CommaSeparatedListOutputParser()

warnings.filterwarnings("ignore")

load_dotenv(dotenv_path=".env" , override=True)

llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)

prompt=ChatPromptTemplate.from_messages(
    ["""
     You are an instagram user and known for writing the most engaging and entertaining captions.
     Study the following description of the image the user has submitted ,it is as follows: {caption}
     You have to create an instagram caption according to the given text prompt ,the text prompt describes what the user expects from the caption and it is as follows:{prompt}
     Do not mention any color.
     Do not mention gender.
     Do not mention numbers.
     Your answer should be concise and not long than 4 lines.
     """
     ]
)



chain =prompt| llm |output_parser
def gen_chain(caption,prompt):
    response=chain.invoke({"caption": caption,"prompt":prompt })
    # response=str(response)
    response=' '.join(response)
    return response

gen_chain('a mac with long black hair','a cool caption')