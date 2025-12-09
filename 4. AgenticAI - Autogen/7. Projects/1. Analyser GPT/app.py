import asyncio
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from agents import planner_agent,local_agent,language_agent,travel_summary_agent, model_client


async def main():
    try:
        termination = TextMentionTermination("TERMINATE")
        group_chat = RoundRobinGroupChat(
            [planner_agent, local_agent, language_agent, travel_summary_agent], termination_condition=termination
        )
        
        await Console(group_chat.run_stream(task="Plan a 3 day trip to Nepal."))
    finally:
        await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())