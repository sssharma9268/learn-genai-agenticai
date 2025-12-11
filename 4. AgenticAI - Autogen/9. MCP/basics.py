from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
import asyncio
import time 
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

model_client = OpenAIChatCompletionClient(
    model="gemini-2.5-flash",
    api_key=api_key,
)

async def main():

    params = StdioServerParams(
        command="uvx",
        args=["mcp-server-time", "--local-timezone=America/New_York"]
    )

    async with McpWorkbench(server_params=params) as workbench:
        agent = AssistantAgent(
            name="Agent",
            system_message="You are a helpful assistant.",
            model_client=model_client,
            workbench=workbench,
        )

        task = "What time is it now in Tehran?"

        async for msg in agent.run_stream(task=task):
            print('--' * 20)
            print(msg)

if __name__ == "__main__":
    asyncio.run(main())