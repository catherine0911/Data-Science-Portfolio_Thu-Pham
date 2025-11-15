from langchain_openai.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

def generate_insight(summary_text, openai_api_key):
    # Build a chat prompt
    system_prompt = SystemMessagePromptTemplate.from_template(
        "You are a data analyst. Provide a clear, insightful summary of analysis results."
    )
    human_prompt = HumanMessagePromptTemplate.from_template("{summary}")

    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    llm = ChatOpenAI(
        model_name="gpt-5-mini",
        openai_api_key=openai_api_key,
        temperature=0.3
    )

    chain = LLMChain(prompt=chat_prompt, llm=llm)
    return chain.run(summary=summary_text)
