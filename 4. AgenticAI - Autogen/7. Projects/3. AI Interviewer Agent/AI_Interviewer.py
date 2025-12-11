from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('GEMINI_API_KEY')
model_client = OpenAIChatCompletionClient(
    model="gemini-2.5-flash",
    api_key=api_key,
)


# We need three agents: 1. Interviewer 2. Candidate 3. Career Coach
# 1st and 3rd can be of type AssistantAgent, 2nd will be of type UserProxyAgent who will be giving interview.

async def team_config(job_position:str):
    interviewer = AssistantAgent(
        name="Interviewer",
        model_client=model_client,
        description=f"An AI agent that conducts interviews for a {job_position} position.",
        system_message=f'''
        You are a professional interviewer for a {job_position} position.
        Ask one clear question at a time and wait for user to respond.
        Your job is to continue and ask questions, don't pay any attention to career coach response.
        Make sure to ask question based on Candidate's answer and your expertise in the field.
        Ask 3 questions in total covering technical skills and experience, problem-solving abilities, and cultural fit.
        After asking 3 questions, say 'TERMINATE' at the end of the interview.
        Make it under 50 words.
        '''
    )

    candidate = UserProxyAgent(
        name="Interviewee",
        description=f" An agent that simulates a candidate for a {job_position} position",
        input_func=input
    )

    career_coach = AssistantAgent(
        name="Career_Coach",
        model_client=model_client,
        description=f"An AI agent that provides feedback and advice to candidates for a {job_position} position.",
        system_message=f'''
        You are a career coach specializing in preparing candidates for {job_position} interviews.
        Provide constructive feedback on the candidate's responses and suggest improvements.
        After the interview, summarize the candidate's performance and provide actionable advice.
        Make it under 100 words.
        '''
    )

    team = RoundRobinGroupChat(
        participants=[interviewer,candidate,career_coach],
        termination_condition=TextMentionTermination(text="TERMINATE"),
        max_turns=20     
    )

    return team

async def interview(team):
    async for message in team.run_stream(task='Start the interview with the first question?'):
        message = message.content 
        yield message



async def main():
    job_position = "Software Engineer"
    team = await team_config(job_position)

    async for message in interview(team=team):
        print('-'*50)
        print(message)

if __name__== "__main__":
    import asyncio
    asyncio.run(main())