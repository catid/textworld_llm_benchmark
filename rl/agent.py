# OpenAI API does not support Min-P but we want to use that

import openai

import logging, time


################################################################################
# Prompt Engineering

def get_first_prompt(text):
    return f"""I'm a software engineer playing a very difficult puzzle game.  Please think step-by-step and help me finish the game!  The game initially shows:
```
{text}
```

Please provide a concise bulleted list of:
* A list of important observations and facts to remember.  Include the known remaining goals.
* A list of the best commands to enter, starting with the best option.  Write the commands in 'quotes'."""

def get_step_prompt(prev_action, text):
    return f"""After entering `{prev_action}`:
```
{text}
```

Please provide a concise bulleted list of:
* A list of important observations and facts to remember.  Include the known remaining goals.
* A list of the best commands to enter, starting with the best option.  Write the commands in 'quotes'."""


################################################################################
# Output Parsing

import re

def remove_trailing_punctuation_and_whitespace(text):
    # Regular expression to remove trailing punctuation and whitespace
    return re.sub(r'[.!?,:;]*\s*$', '', text.strip())

def extract_llm_results(text):
    lines = text.split('\n')

    observations = []
    commands = []

    in_observations = False
    in_commands = False

    # Patterns to detect the start of observations and commands sections
    observations_start_pattern = re.compile(r'observations|facts|notes', re.IGNORECASE)
    commands_start_pattern = re.compile(r'commands|actions|steps', re.IGNORECASE)

    # Pattern to identify bullet points (both * and numeric)
    bullet_point_pattern = re.compile(r'^(\*\s|\d+\.\s)')

    for line in lines:
        line = line.strip()

        # Check for empty line to potentially exit the current section
        if not line:
            continue

        # If it is not bulleted
        if not bullet_point_pattern.match(line):
            # Detecting section headers for observations and commands
            if observations_start_pattern.search(line):
                in_observations = True
                in_commands = False
                continue
            elif commands_start_pattern.search(line):
                in_observations = False
                in_commands = True
                continue

        if not in_observations and not in_commands:
            continue

        debullet = re.sub(bullet_point_pattern, '', line)

        if in_observations:
            cleaned_text = remove_trailing_punctuation_and_whitespace(debullet)
            observations.append(cleaned_text)
        elif in_commands:
            # Remove quotes if present
            command_match = re.search(r'`([^`]+)`|\'([^\']+)\'|"([^"]+)"|(.+)', debullet)
            if command_match:
                command = next(filter(None, command_match.groups()), '').strip()
                command = remove_trailing_punctuation_and_whitespace(command)
                commands.append(command)

    return observations, commands


################################################################################
# LLM-Based Agent

class Agent:
    def __init__(self, args):
        self.args = args
        self.client = openai.OpenAI(
            api_key=args.openai_api_key,
            base_url=args.openai_base_url
        )
        self.reset()

    def query_llm(self, messages, max_tokens=512, temperature=0.1):
        # Retry forever
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.args.openai_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=1,
                )

                content = response.choices[0].message.content.strip()

                return content
            except Exception as e:
                logging.error(f"client.chat.completions.create error: {e}.  Retrying in 1 second...")
                time.sleep(1)

    def reset(self):
        self.messages = [
            {"role": "system", "content": "You are a game playing genius AI assistant, helping the user to solve a challenging game."},
        ]
        self.commands = []

    def act(self, env):
        if not env.prev_command:
            prompt = get_first_prompt(env.observation)
        else:
            prompt = get_step_prompt(env.prev_command, env.observation)

        self.messages.append({"role": "user", "content": prompt})

        content = self.query_llm(self.messages, max_tokens=self.args.max_tokens, temperature=self.args.temperature)

        if len(self.messages) >= 4:
            self.messages[-1]["content"] = env.observation
            self.messages[-2]["content"] = env.prev_command

        self.messages.append({"role": "assistant", "content": content})

        observations, commands = extract_llm_results(content)

        #print(f"content: {content} observations = {observations} commands = {commands}")

        if self.args.disable_rl:
            command = commands[0] if len(commands) > 0 else "look"
        else:
            command = "FIXME"

        self.commands.append(command)

        env.step(command)

        return env.done
