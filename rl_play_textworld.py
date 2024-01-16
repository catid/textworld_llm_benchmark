# Play TextWorld with LLM+RL

################################################################################
# Dependencies

import logging

logging.basicConfig(level=logging.INFO)


################################################################################
# Test Runner

from rl.agent import Agent
from rl.textworld_env import TextWorldEnv

def test_once(args):
    agent = Agent(args)
    env = TextWorldEnv(args)

    while not env.done:
        agent.act(env)


################################################################################
# Entrypoint

import os, shutil, argparse

def delete_tw_games_folder():
    folder_path = "tw_games"

    # Check if the folder exists
    if os.path.exists(folder_path):
        # Remove the folder and all its contents
        shutil.rmtree(folder_path)
        print(f"The folder '{folder_path}' has been deleted.")
    else:
        print(f"The folder '{folder_path}' does not exist.")

def main():
    parser = argparse.ArgumentParser(description="Run TextWorld tests.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument("--openai_api_key", type=str, default="14d78630027e15de243c8b3b489a91fa", help="API key for OpenAI client.")
    parser.add_argument("--openai_base_url", type=str, default="http://devnuc.lan:5000/v1/", help="Base URL for OpenAI client.")
    parser.add_argument("--openai_model", type=str, default="gpt-3.5-turbo-16k", help="Model to request")
    parser.add_argument("--seed", type=int, default=0, help="Specify a seed or 0 for random")
    parser.add_argument("--max_tokens", type=int, default=512, help="LLM tokens")
    parser.add_argument("--temperature", type=float, default=0.1, help="LLM temperature")
    parser.add_argument("--disable_rl", action="store_true", help="Disable RL: Just uses first suggested LLM action (Baseline)")
    parser.add_argument("--max_episode_steps", type=int, default=50, help="Time limit to complete the challenge (in turns)")
    parser.add_argument("--full_messages", action="store_true", help="Keep full LLM output?  Normally it just enters the agent action into the LLM message list")

    args = parser.parse_args()

    print(f"args: {args}")

    # Clean up mess from last time
    delete_tw_games_folder()

    # Let's go!
    test_once(args)

if __name__ == "__main__":
    main()
