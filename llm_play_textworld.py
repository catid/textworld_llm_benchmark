import logging
from multiprocessing import Process, Manager, Queue
from tqdm import tqdm
import os, shutil
import random
import argparse


################################################################################
# Dependencies

logging.basicConfig(level=logging.WARNING)

# OpenAI
from openai import OpenAI
client = OpenAI(
    api_key="14d78630027e15de243c8b3b489a91fa",
    base_url="http://devnuc.lan:5000/v1"
)

# TextWorld
import textworld
import textworld.gym
from textworld.generator import GameOptions, make_game, compile_game


################################################################################
# Tools

def extract_first_quoted_string(input_string):
    import re
    match = re.search(r'"([^"]*)"', input_string)
    if match:
        return match.group(1)
    else:
        return None

def extract_command(content):
    command = extract_first_quoted_string(content) or content.split('\n', 1)[0].split('.', 1)[0]
    truncated = len(command) > 32
    return command[:32], truncated

def interact_with_environment(env, command):
    try:
        return env.step(command)
    except Exception as e:
        logging.error(f"Error during environment interaction: {e}")
        return None, 0, True, {}


################################################################################
# Test Runner

def test_once(args, shared_dict, progress_queue):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)
    seed = random.randint(1, 10000000)

    options = GameOptions()
    options.seeds = seed
    options.nb_rooms = 20
    options.nb_objects = 10
    options.nb_parallel_quests = 5
    options.force_recompile = True
    options.path = f"tw_games/custom_game{seed}.z8"
    options.chaining.min_length = 1
    options.chaining.max_length = 4
    options.chaining.min_breadth = 1
    options.chaining.max_breadth = 4
    options.chaining.min_depth = 1
    options.chaining.max_depth = 4

    try:
        game = make_game(options)
        game_file = compile_game(game, options)
    except Exception as e:
        logging.error(f"Error during game generation: {e}")
        return

    try:
        env_id = textworld.gym.register_game(game_file, max_episode_steps=args.max_episode_steps)

        env = textworld.gym.make(env_id)  # Create the environment
        obs, infos = env.reset()

        if args.verbose:
            env.render()

        messages = [
            {"role": "system", "content": "You are a game playing genius AI assistant, helping the user to solve a challenging game."},
        ]

        messages.append({"role": "user", "content": f"""I'm a software engineer playing a very difficult puzzle game.  Please help me finish the game!

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
                max_tokens=512,
                n=1,
            )
            content = response.choices[0].message.content.strip()
            command, truncated = extract_command(content)
            if args.verbose:
                logging.debug(f"AI Command: {command}")

            obs, score, done, infos = interact_with_environment(env, command)
            if args.verbose:
                env.render()
                logging.info(f"Moves: {moves}; Score: {score}")

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
            reply += "What are your long-term goals? Think step by step and come up with the best action to take next. Write the command in \"quotes\" to select the next action."

            messages.append({"role": "user", "content": reply})

            progress_queue.put(1)  # Progress update
            moves += 1

        shared_dict['scores'].append(score)
        shared_dict['total_moves'] += moves

        if args.verbose:
            logging.info(f"Final moves: {moves}; Score: {score}")

    except Exception as e:
        logging.error(f"Error during test execution: {e}")


################################################################################
# Parallel Test Runner

def run_tests(args):
    manager = Manager()
    shared_dict = manager.dict()
    shared_dict['scores'] = manager.list()
    shared_dict['total_moves'] = 0
    progress_queue = Queue()

    processes = []

    print(f"Generating {args.num_tests} random maps and exploring in parallel instances...")

    # Create and start processes
    for _ in range(args.num_tests):
        p = Process(target=test_once, args=(args, shared_dict, progress_queue))
        p.start()
        processes.append(p)

    # Initialize tqdm progress bar
    with tqdm(total=args.num_tests * args.max_episode_steps) as pbar:  # Assuming 50 moves per test
        while any(p.is_alive() for p in processes):
            while not progress_queue.empty():
                progress_queue.get()
                pbar.update(1)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Calculate statistics
    scores = shared_dict['scores']
    avg_score = sum(scores) / len(scores) if scores else 0
    min_score = min(scores) if scores else 0
    max_score = max(scores) if scores else 0
    avg_moves = shared_dict['total_moves'] / args.num_tests

    # Print statistics
    print(f"Min/Avg/Max Score: {min_score}/{avg_score}/{max_score}")
    print(f"Average Number of Moves: {avg_moves}")


################################################################################
# Entrypoint

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
    parser.add_argument("--num_tests", type=int, default=1, help="Number of tests to run.")
    parser.add_argument("--max_episode_steps", type=int, default=100, help="Maximum number of steps per episode.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")

    args = parser.parse_args()

    # Clean up mess from last time
    delete_tw_games_folder()

    # Let's go!
    run_tests(args)

if __name__ == "__main__":
    main()
