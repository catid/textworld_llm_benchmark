# OpenAI
from openai import OpenAI
client = OpenAI(
    api_key="14d78630027e15de243c8b3b489a91fa",
    base_url="http://devnuc.lan:5000/v1"
)

import textworld

from textworld.generator import GameOptions, make_game, compile_game

options = GameOptions()

options.nb_rooms = 20
options.nb_objects = 10
options.nb_parallel_quests = 5
options.force_recompile = True
options.path = "tw_games/custom_game.z8"
options.chaining.min_length = 1
options.chaining.max_length = 4
options.chaining.min_breadth = 1
options.chaining.max_breadth = 4
options.chaining.min_depth = 1
options.chaining.max_depth = 4
options.seeds = 12345

print(options)

game = make_game(options)
game_file = compile_game(game, options)


import textworld.gym

# Register a text-based game as a new environment.
env_id = textworld.gym.register_game(game_file,
                                     max_episode_steps=50)

env = textworld.gym.make(env_id)  # Start the environment.


def extract_first_quoted_string(input_string):
    import re
    match = re.search(r'"([^"]*)"', input_string)
    if match:
        return match.group(1)
    else:
        return None

obs, infos = env.reset()  # Start new episode.
env.render()

advice = ""
command = ""

messages = [
    {"role": "system", "content": "You are a game playing genius helping the user to solve a challenging game."},
]

messages.append({
    "role": "user", "content":
        f"""
I'm a software engineer playing a very difficult puzzle game.  Please help me finish the game!

The game initially shows:
```
{obs}
```

Think step by step and come up with the best action to take next. Write the command in \"quotes\" to select the next action."""
})

score, moves, done = 0, 0, False
while not done:
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        temperature=0.1,
        max_tokens=4096,
        n=1,
    )
    content = response.choices[0].message.content.strip()

    print(f"AI: {content}")

    # If AI provided a quoted command take that one
    command = extract_first_quoted_string(content)
    if not command:
        command = content

    # If the command is multi-line just take the first line
    command = command.split('\n', 1)[0]

    # Truncate the command at sentence
    command = command.split('.', 1)[0]

    truncated = False
    full_command = command
    if len(command) > 32:
        truncated = True
        command = command[:32]

    obs, score, done, infos = env.step(command)
    env.render()
    moves += 1

    print("moves: {}; score: {}".format(moves, score))

    messages.append({"role": "assistant", "content": content})

    reply = f"""Game now displays:
```
{obs}
```
"""
    if infos:
        reply += f"""Info:
```
{infos}
```
"""
    if truncated:
        reply += "Your previous command was cut off because it was too long.  Please modify your command to be shorter by removing unnecessary detail.\n"
    reply += "What are your long-term and next immediate goals? Avoid going in circles. Think step by step and come up with the best action to take next. Write the command in \"quotes\" to select the next action."

    messages.append({"role": "user", "content": reply})

env.close()
print("final moves: {}; score: {}".format(moves, score))
