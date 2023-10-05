import config
import os
from libs.langchain.langchain.agents.agent_iterator import AgentExecutorIterator
from libs.langchain.langchain.agents.agent import AgentExecutor
from libs.experimental.langchain_experimental.plan_and_execute import agent_executor
from libs.langchain.langchain.agents.agent import AgentExecutor
from libs.langchain.langchain.agents.mrkl.base import ZeroShotAgent
from libs.langchain.langchain.chains.llm import LLMChain
from libs.langchain.langchain.memory.buffer import ConversationBufferMemory
from libs.langchain.langchain.retrievers.document_compressors.chain_extract import LLMChainExtractor
from libs.langchain.langchain.tools.base import Tool
from libs.langchain.langchain.utilities.google_search import GoogleSearchAPIWrapper

os.environ['OPENAI_API_KEY'] = config.OPENAI_API_KEY
os.environ['GOOGLE_API_KEY'] = config.GOOGLE_API_KEY
os.environ['GOOGLE_CSE_ID'] = config.GOOGLE_CSE_ID


search = GoogleSearchAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    )
]

prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)
memory = ConversationBufferMemory(memory_key="chat_history")

agent = ZeroShotAgent(llm_chain=LLMChainExtractor, tools=tools, verbose=True)
agent_chain = AgentExecutorIterator.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory
)

agent_chain.run(input="How many people live in canada?")
agent_chain.run(input="what is their national anthem called?")