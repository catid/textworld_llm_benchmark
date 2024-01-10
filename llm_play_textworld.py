# Play TextWorld with LLMs

from multiprocessing import Process, Manager, Queue
from tqdm import tqdm
import logging, os, shutil, random, time, struct, argparse
import numpy as np
import scipy.stats as stats


################################################################################
# Dependencies

logging.basicConfig(level=logging.WARNING)

# OpenAI
import openai

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

    # Generate a seed using os.urandom
    random_seed_bytes = os.urandom(8)  # Get 8 bytes of random data
    seed = struct.unpack("<Q", random_seed_bytes)[0] % 4294967296  # Convert bytes to a single integer

    # Initialize the OpenAI client with the provided arguments
    client = openai.OpenAI(
        api_key=args.openai_api_key,
        base_url=args.openai_base_url
    )

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
                model="gpt-4-1106-preview", # Actually using Mixtral
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

            messages.append({"role": "assistant", "content": command})
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

def print_scores(args, shared_dict, completed_tests=None):
    # Extract scores and calculate basic statistics
    scores = shared_dict['scores']
    avg_score = np.mean(scores) if scores else 0
    std_dev = np.std(scores, ddof=1) if len(scores) > 1 else 0
    min_score = min(scores) if scores else 0
    max_score = max(scores) if scores else 0

    # Print basic statistics
    if not completed_tests:
        completed_tests = args.num_tests
        print(f"Final results for {args.num_tests} tests:")
    else:
        print(f"Completed {completed_tests}/{args.num_tests} tests.")
    
    avg_moves = shared_dict['total_moves'] / completed_tests if completed_tests else 0
    print(f"Min/Avg/Max Score: {min_score}/{avg_score}/{max_score} ± stddev={std_dev}")
    print(f"Average Number of Moves: {avg_moves}")

    # Calculate and print confidence interval and standard deviation if scores are available
    if scores:
        # Choose your confidence level (e.g., 95%)
        confidence_level = 0.95
        z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        margin_error = z_score * (std_dev / np.sqrt(len(scores)))

        print(f"95% Confidence Interval for the Average Score: {avg_score} ± {margin_error:.2f}")

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
                        print_scores(args, shared_dict, completed_tests)

                        p = Process(target=test_once, args=(args, shared_dict, progress_queue))
                        p.start()
                        processes.append(p)

            # Update progress bar
            while not progress_queue.empty():
                progress_queue.get()
                pbar.update(1)

            time.sleep(0.1)  # Prevents the loop from hogging CPU

    print_scores(args, shared_dict)


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
    parser.add_argument("--parallel", type=int, default=32, help="Number of tests to run in parallel.")
    parser.add_argument("--num_tests", type=int, default=100, help="Total number of tests to run.")
    parser.add_argument("--max_tokens", type=int, default=256, help="Number of tests to run in parallel.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Total number of tests to run.")
    parser.add_argument("--max_episode_steps", type=int, default=50, help="Maximum number of steps per episode.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument("--openai_api_key", type=str, default="14d78630027e15de243c8b3b489a91fa", help="API key for OpenAI client.")
    parser.add_argument("--openai_base_url", type=str, default="http://devnuc.lan:5000/v1", help="Base URL for OpenAI client.")

    args = parser.parse_args()

    print(f"args: {args}")

    # Clean up mess from last time
    delete_tw_games_folder()

    # Let's go!
    run_tests(args)

if __name__ == "__main__":
    main()
