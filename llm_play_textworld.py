import logging
from multiprocessing import Process, Manager, Queue
from tqdm import tqdm
import os, shutil
import random
import argparse
import time


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
        env = textworld.gym.make(env_id)
        obs, infos = env.reset()

        if args.verbose:
            env.render()

        system_message = "You are a game playing genius AI assistant."
        user_message = f"I'm playing a puzzle game. Help me find the next step. The game shows:\n```\n{obs}\n```."

        score, moves, done = 0, 0, False

        while not done:
            response = client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                n=1,
            )
            content = response.choices[0].message.content.strip()
            command, truncated = extract_command(content)

            obs, score, done, infos = interact_with_environment(env, command)

            # Update user_message with the latest command and its result
            user_message = f"Next step: '{command}' resulted in:\n```\n{obs}\n```"
            if infos:
                user_message += f"\nInfo: '{infos}'."

            if truncated:
                user_message += "\nCommand was truncated. Please be more concise."

            progress_queue.put(1)  # Progress update
            moves += 1

        shared_dict['scores'].append(score)
        shared_dict['total_moves'] += moves

        if args.verbose:
            logging.info(f"Final moves: {moves}; Score: {score}")

    except Exception as e:
        logging.error(f"Error during test execution: {e}")

    finally:
        if 'env' in locals() and env is not None:
            env.close()


################################################################################
# Parallel Test Runner

def run_tests(args):
    manager = Manager()
    shared_dict = manager.dict()
    shared_dict['scores'] = manager.list()
    shared_dict['total_moves'] = 0
    progress_queue = Queue()

    print(f"Generating {args.num_tests} random maps and exploring in parallel instances...")

    # Initialize tqdm progress bar
    with tqdm(total=args.num_tests * args.max_episode_steps) as pbar:
        processes = []
        for _ in range(min(args.num_tests, args.parallel)):
            p = Process(target=test_once, args=(args, shared_dict, progress_queue))
            p.start()
            processes.append(p)

        completed_tests = 0
        while completed_tests < args.num_tests:
            # Remove and join completed processes
            for p in list(processes):
                if not p.is_alive():
                    p.join()
                    processes.remove(p)
                    completed_tests += 1

                    # Start a new process if there are tests left
                    if completed_tests < args.num_tests:
                        print(f"Completed {completed_tests}/{args.num_tests} tests.")
                        p = Process(target=test_once, args=(args, shared_dict, progress_queue))
                        p.start()
                        processes.append(p)

            # Update progress bar
            while not progress_queue.empty():
                progress_queue.get()
                pbar.update(1)

            time.sleep(0.1)  # Prevents the loop from hogging CPU

    # Calculate and print statistics
    scores = shared_dict['scores']
    avg_score = sum(scores) / len(scores) if scores else 0
    min_score = min(scores) if scores else 0
    max_score = max(scores) if scores else 0
    avg_moves = shared_dict['total_moves'] / args.num_tests

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
    parser.add_argument("--parallel", type=int, default=16, help="Number of tests to run in parallel.")
    parser.add_argument("--num_tests", type=int, default=100, help="Total number of tests to run.")
    parser.add_argument("--max_tokens", type=int, default=256, help="Number of tests to run in parallel.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Total number of tests to run.")
    parser.add_argument("--max_episode_steps", type=int, default=50, help="Maximum number of steps per episode.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")

    args = parser.parse_args()

    print(f"args: {args}")

    # Clean up mess from last time
    delete_tw_games_folder()

    # Let's go!
    run_tests(args)

if __name__ == "__main__":
    main()
