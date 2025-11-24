import random

from Baseline.Util.solution import Solution
from Baseline.Util.util import insert_, get_all_position, get_feasible_insert_position
from Baseline.Util.parameter_ahasp import paramet_ahasp


def generate_solution_random(instance):
    """
    Generates an initial solution by randomly inserting tasks one by one.
    It ensures topological feasibility by checking constraints during the second chain insertion.

    Args:
        instance (list): The problem instance data.

    Returns:
        Solution: A randomly generated valid Solution object.
    """
    task_num = len(instance)
    sequence_map = {}

    # Initialize path map for all robots with 0 (indicates empty start)
    # Robot IDs range from 1 to ROBOT_NUM
    path_init_task_map = {
        r: 0 for r in range(1, paramet_ahasp.ROBOT_NUM + 1)
    }

    # Iterate through all tasks to build the solution
    for task in range(1, task_num + 1):
        # Randomly decide which chain to process first (Parent vs Child)
        chains = ['parent', 'child']
        random.shuffle(chains)
        first_chain, second_chain = chains

        # 1. Insert into the First Chain
        # "All positions" are valid because there is no coupling constraint yet for this task
        all_positions = get_all_position(sequence_map, path_init_task_map, task, first_chain)

        # Pick a random position and insert
        # Note: random.sample returns a list, [0] extracts the position tuple
        pos_1 = random.sample(list(all_positions), 1)[0]
        insert_(sequence_map, path_init_task_map, task, pos_1, first_chain)

        # 2. Insert into the Second (Coupled) Chain
        # Must use 'get_feasible_insert_position' to satisfy precedence constraints created by step 1
        feasible_positions = get_feasible_insert_position(sequence_map, path_init_task_map, task, second_chain)

        # Pick a random feasible position and insert
        pos_2 = random.sample(list(feasible_positions), 1)[0]
        insert_(sequence_map, path_init_task_map, task, pos_2, second_chain)

    return Solution(instance, sequence_map, path_init_task_map)
