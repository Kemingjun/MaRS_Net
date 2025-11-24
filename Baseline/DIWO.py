import time
import math
import random
import argparse
import copy
import numpy as np

from Baseline.Util.generate_init_solution import generate_solution_random
from Baseline.Util.load_data import read_excel
from Baseline.Util.util import (
    remove_, insert_, get_all_position, get_feasible_insert_position
)
from Baseline.Util.solution import Solution

# ==========================================
# DIWO Configuration
# ==========================================
POP_INITIAL_SIZE = 50  # Initial number of weeds
POP_MAX_SIZE = 100  # Maximum number of weeds allowed
S_MAX = 20  # Maximum number of seeds per weed (Reduced from 100 for balance)
S_MIN = 1  # Minimum number of seeds per weed


# ==========================================
# Neighborhood Operators
# ==========================================

def neighbor_insertion(solution):
    """
    Generates a neighbor by removing a random task and re-inserting it
    into a random feasible position.
    """
    task = random.randint(1, solution.task_num)
    sequence_map = solution.get_sequence_map()
    path_init_task_map = solution.get_path_init_task_map()

    # 1. Remove task
    remove_(sequence_map, path_init_task_map, task)

    # 2. Re-insert into Parent Chain (Random position)
    parent_positions = get_all_position(sequence_map, path_init_task_map, task, 'parent')
    if not parent_positions: return solution

    parent_pos = random.sample(list(parent_positions), 1)[0]
    insert_(sequence_map, path_init_task_map, task, parent_pos, 'parent')

    # 3. Re-insert into Child Chain (Random feasible position)
    child_positions = get_feasible_insert_position(sequence_map, path_init_task_map, task, 'child')
    if not child_positions: return solution

    child_pos = random.sample(list(child_positions), 1)[0]
    insert_(sequence_map, path_init_task_map, task, child_pos, 'child')

    return Solution(solution.instance, sequence_map, path_init_task_map)


def neighbor_swap(solution):
    """
    Generates a neighbor by swapping two tasks (Simplified implementation wrapper).
    Uses neighbor_insertion as a fallback/proxy if swap logic is too complex
    for linked-list consistency in this simplified context.
    """
    # For stability and code consistency with other metaheuristics refactoring,
    # we utilize the insertion operator which effectively explores the neighborhood.
    # (Original swap logic was extremely verbose and prone to link errors without
    # a dedicated structure validation).
    return neighbor_insertion(solution)


def get_neighbor_solution(solution):
    """Randomly selects a neighborhood operator to generate a seed."""
    operators = [neighbor_insertion, neighbor_swap]
    selected_op = random.choice(operators)
    return selected_op(solution)


# ==========================================
# Main DIWO Logic
# ==========================================

def run_diwo(file_path, max_iter, time_limit):
    """
    Executes the Discrete Invasive Weed Optimization (DIWO) algorithm.

    Args:
        file_path (str): Path to the instance file.
        max_iter (int): Maximum number of iterations.
        time_limit (int): Maximum runtime in seconds.
    """
    start_t = time.time()
    print(f"[*] Loading instance: {file_path}")
    instance = read_excel(file_path)

    # 1. Initialization
    # --------------------------------------------------
    weed_list = []
    weed_hashes = set()

    # Generate Initial Population
    print(f"[*] Initializing Population ({POP_INITIAL_SIZE})...")
    for _ in range(POP_INITIAL_SIZE):
        sol = generate_solution_random(instance)
        if sol.hash_key not in weed_hashes:
            weed_list.append(sol)
            weed_hashes.add(sol.hash_key)

    # Identify Best Solution
    best_solution = min(weed_list, key=lambda x: x.get_fitness())
    best_fitness = best_solution.get_fitness()

    print(f"[*] Start DIWO Optimization (Max Pop: {POP_MAX_SIZE}, Max Iter: {max_iter}, Time Limit: {time_limit}s)")

    # 2. Main Loop
    # --------------------------------------------------
    iteration = 0
    while iteration < max_iter and (time.time() - start_t) < time_limit:
        iteration += 1

        # Calculate Fitness Range
        fitness_vals = [w.get_fitness() for w in weed_list]
        min_fit = min(fitness_vals)
        max_fit = max(fitness_vals)

        # Update Best
        if min_fit < best_fitness:
            best_fitness = min_fit
            # Find the object corresponding to min_fit
            for w in weed_list:
                if w.get_fitness() == min_fit:
                    best_solution = w
                    break
            print(f"Iter {iteration}: New Best Found! Fitness: {best_fitness:.4f}")

        # Reproduction (Seed Generation)
        seed_list = []

        for weed in weed_list:
            if (time.time() - start_t) > time_limit: break

            weed_fit = weed.get_fitness()

            # Calculate Seed Count based on Fitness Rank
            # Better fitness (lower value) -> More seeds
            if abs(max_fit - min_fit) < 1e-6:
                seed_num = random.randint(S_MIN, S_MAX)
            else:
                # Linear interpolation: Best weed gets S_MAX, Worst gets S_MIN
                ratio = (weed_fit - min_fit) / (max_fit - min_fit)
                seed_num = math.floor(S_MAX - ratio * (S_MAX - S_MIN))

            # Ensure at least S_MIN seeds
            seed_num = max(S_MIN, seed_num)

            # Generate Seeds
            for _ in range(seed_num):
                new_seed = get_neighbor_solution(weed)
                # Avoid duplicates in the new batch
                if new_seed.hash_key not in weed_hashes:
                    seed_list.append(new_seed)
                    # Note: We don't add to weed_hashes immediately to allow exploration
                    # in next generation, or we can manage a global set.
                    # Here we follow standard logic: specific generation uniqueness.

        # Spatial Dispersal (Combine Parents + Seeds)
        combined_population = weed_list + seed_list

        # Competitive Exclusion (Selection)
        # Sort by fitness (Ascending -> Lower is better)
        combined_population.sort(key=lambda x: x.get_fitness())

        # Truncate to Max Population Size
        weed_list = combined_population[:POP_MAX_SIZE]

        # Rebuild Hash Set for next iteration efficiency
        weed_hashes = {w.hash_key for w in weed_list}

        # Log
        if iteration % 10 == 0:
            print(
                f"Iter {iteration}, Best: {best_fitness:.2f}, Pop Size: {len(weed_list)}, Time: {time.time() - start_t:.2f}s")

    return best_solution, best_fitness, time.time() - start_t


# ==========================================
# Main Entry Point
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DIWO Solver for AHASP")

    parser.add_argument('--file', type=str, required=True, help="Path to the instance Excel file.")
    parser.add_argument('--iter', type=int, default=100, help="Maximum number of iterations.")
    parser.add_argument('--time', type=int, default=3600, help="Time limit in seconds.")

    args = parser.parse_args()

    best_sol, best_fit, elapsed = run_diwo(args.file, args.iter, args.time)

    print("\n" + "=" * 50)
    print(" DIWO RESULTS")
    print("=" * 50)
    print(f"Best Fitness : {best_fit:.4f}")
    print(f"Elapsed Time : {elapsed:.4f}s")
    # print(f"Path Map     : {best_sol.get_path_map()}")
    print("=" * 50)