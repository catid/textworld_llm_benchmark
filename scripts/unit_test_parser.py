text = """
 Sure, here are the updated observations and best commands based on the new information:

Important observations and facts to remember:

* You have already opened the type 1 box and retrieved the American limited edition keycard.
* You need to take the American limited edition keycard before unlocking the American limited edition gate.
* You need to open the American limited edition gate leading east.
* You need to move east and pick up the shirt from the attic.

Best commands to enter:

* `take American limited edition keycard`
* `unlock American limited edition gate with American limited edition keycard`
* `open American limited edition gate`
* `move east`
* `pick up shirt`
* `open door` (if there is a closed door leading south)

Note: It's important to remember to take the American limited edition keycard before unlocking the gate.
"""


text2 = """
**Important observations and facts:**

* The player is in a steam room.
* There is a box in the room which contains a rectangular passkey and a towel.
* The player can go south or west from the room.
* The remaining goals are:
        1. Retrieve the towel from the box.
        2. Take the rectangular passkey from the box.
        3. Move west.
        4. Attempt to move south twice.
        5. Unlock the rectangular locker.
        6. Take the Canadian limited edition latchkey from the shelf in the basement.
        7. Unlock the Canadian limited edition safe with the Canadian limited edition latchkey.
        8. Deposit the bug on the floor of the washroom.

**Best commands to enter:**

1. "Take towel from box"
2. "Take passkey from box"
3. "Go west"
4. "Go south"
5. "Go south"
6. "Unlock rectangular locker with rectangular passkey"
7. "Go to the basement"
8. "Take Canadian limited edition latchkey from the shelf"
9. "Go west"
10. "Unlock Canadian limited edition safe with Canadian limited edition latchkey"
11. "Go east"
12. "Go north"
13. "Go west"
14. "Go south"
15. "Go west"
16. "Deposit bug on floor of washroom"
"""

text3 = """
**Important observations and facts:**

* The game involves navigating a world and solving puzzles to progress.
* The current location is a vault with a stand, a closed passageway to the south, an exit to the east, and a northern entrance that is unblocked.
* There is a key on the stand in the vault.
* The goal is to explore the world, pick up the key, move north, find a cubicle with a safe, unlock and open the safe, take a backup calendar from the toolbox, deposit the calendar on the cubicle floor, take the cuboid keycard from the safe, unlock and open the cuboid box, retrieve a folder from the box, move south, move east, recover a sponge from the board, retrieve an insect from the cabinet in the bathroom, and then stop.

**Best commands:**

1. "take key" (to pick up the key from the stand)
2. "go north" (to move to the next location)

Note: The commands should be entered in the order listed above, one at a time.
"""


text4 = """
**Important observations and facts:**

* The game is a text-based adventure game.
* The player is currently in the kitchenette.
* There are unguarded exits to the east, north, and west.
* The game has provided a sequence of actions to follow:
        1. Find the closet.
        2. Open the closet.
        3. Pick up the bug from the trunk inside the closet.
        4. Travel west.
        5. Venture south.
        6. Pick up the butterfly from the floor of the bar.
        7. Place the butterfly into the suitcase.
        8. Retrieve the latchkey from the bookshelf.
        9. Unlock the scullery's safe with the latchkey.
        10. Venture north.
        11. Venture north again.
        12. Move west.
        13. Retrieve the pair of headphones from the rack in the spare room.
        14. Venture north.
        15. Venture east.
        16. Pick up the TextWorld limited edition latchkey from the dresser in the bedchamber.
        17. Lock the TextWorld limited edition safe in the bedchamber.
* The game does not recognize the verb "find".
* The game does not recognize the object "trunk inside the closet".

**Best commands to enter:**

1. "Look around"
2. "Examine the room"
3. "Check for a closet"
4. "Open the closet"
5. "Take the bug from the trunk" (once the trunk is found inside the closet)
6. "Go west"
7. "Go south" (assuming the game allows this command instead of "venture south")
8. "Pick up the butterfly from the floor of the bar"
9. "Place the butterfly into the suitcase"
10. "Retrieve the latchkey from the bookshelf"
11. "Go west" (assuming the game allows this command instead of "venture west")
12. "Insert the latchkey into the safe in the scullery's lock to unlock it"
13. "Go north" (assuming the game allows this command
"""


# This is my least favorite aspect of Python
import os, sys
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.insert(0, project_root)

from rl.agent import extract_llm_results






print(extract_llm_results(text))

print(extract_llm_results(text2))

print(extract_llm_results(text3))

print(extract_llm_results(text4))
