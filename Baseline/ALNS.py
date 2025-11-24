import time
import math
import argparse
import numpy as np
import random

from Baseline.Util.generate_init_solution import generate_solution_random
from Baseline.Util.load_data import read_excel
from Baseline.Util.operators import (
    destroy_couple_random,
    destroy_couple_worst_cost,
    destroy_couple_worst_distance,
    destroy_couple_worst_tardiness,
    repair_couple_greedy,
    repair_couple_greedy_urgency_priority,
    repair_couple_greedy_cost_priority
)
from Baseline.Util.util import get_T

# ==========================================
# ALNS Configuration & Operators
# ==========================================
DESTRUCT_OPERATORS = [
    destroy_couple_random,
    destroy_couple_worst_cost,
    destroy_couple_worst_distance,
    destroy_couple_worst_tardiness,
]

CONSTRUCT_OPERATORS = [
    repair_couple_greedy,
    repair_couple_greedy_urgency_priority,
    repair_couple_greedy_cost_priority
]

# Scoring Parameters
SIGMA_1 = 33  # New global best
SIGMA_2 = 13  # Better than current or New solution
SIGMA_3 = 9  # Accepted (Metropolis)
RHO = 0.1  # Reaction factor (decay rate)
L_S = 15  # Segment length (iterations to update weights)


def select_operators(w_destruct, w_construct):
    """Selects destruction and construction operators using Roulette Wheel Selection."""
    # Normalize weights to probabilities
    p_destruct = np.array(w_destruct) / sum(w_destruct)
    p_construct = np.array(w_construct) / sum(w_construct)

    d_idx = np.random.choice(np.arange(len(w_destruct)), p=p_destruct)
    c_idx = np.random.choice(np.arange(len(w_construct)), p=p_construct)

    return d_idx, c_idx


def run_alns(file_path, max_iter, time_limit):
    """
    Executes the Adaptive Large Neighborhood Search (ALNS) algorithm.

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
    d_op_num = len(DESTRUCT_OPERATORS)
    c_op_num = len(CONSTRUCT_OPERATORS)

    # Weights and Scores
    w_destruct = [1.0] * d_op_num
    w_construct = [1.0] * c_op_num
    score_destruct = [0.0] * d_op_num
    score_construct = [0.0] * c_op_num
    count_destruct = [0] * d_op_num
    count_construct = [0] * c_op_num

    solution_table = {}  # Tabu / History mechanism

    # Task parameters
    task_num = len(instance)
    d_num = math.ceil(task_num * 0.2)  # Destruction rate (20%)
    temperature = get_T(instance)

    # Generate Initial Solution
    current_sol = generate_solution_random(instance)
    solution_table[current_sol.hash_key] = current_sol

    current_fitness = current_sol.get_fitness()
    best_sol = current_sol
    best_fitness = current_fitness

    print(f"[*] Start ALNS Optimization (Max Iter: {max_iter}, Time Limit: {time_limit}s)")

    # 2. Main Loop
    # --------------------------------------------------
    iteration = 0
    while iteration < max_iter and (time.time() - start_t) < time_limit:
        iteration += 1

        # A. Operator Selection
        d_idx, c_idx = select_operators(w_destruct, w_construct)

        # B. Apply Operators (Destroy & Repair)
        # Note: destroy operators return (seq, path, list), construct takes (*args, solution)
        destroyed_data = DESTRUCT_OPERATORS[d_idx](current_sol, d_num)
        new_sol = CONSTRUCT_OPERATORS[c_idx](*destroyed_data, current_sol)

        # Check if solution is new
        is_new = new_sol.hash_key not in solution_table
        if is_new:
            solution_table[new_sol.hash_key] = new_sol

        new_fitness = new_sol.get_fitness()

        # C. Acceptance & Scoring
        current_score = 0
        accepted = False

        if new_fitness < current_fitness:
            # Improved
            accepted = True
            if new_fitness < best_fitness:
                # Global Best
                best_sol = new_sol
                best_fitness = new_fitness
                current_score = SIGMA_1
                print(f"Iter {iteration}: New Best Found! Fitness: {best_fitness:.4f}")
            else:
                # Better than current
                current_score = SIGMA_2 if is_new else 0

        elif new_fitness == current_fitness:
            # Same quality
            accepted = is_new  # Only accept if it's a new structure
            current_score = SIGMA_2 if is_new else 0

        else:
            # Worse (Simulated Annealing Criterion)
            delta_e = new_fitness - current_fitness
            prob = math.exp(-delta_e / temperature)
            if random.random() < prob:
                accepted = True
                current_score = SIGMA_3 if is_new else 0

        # Update Current Solution if accepted
        if accepted:
            current_sol = new_sol
            current_fitness = new_fitness

        # D. Update Statistics
        count_destruct[d_idx] += 1
        count_construct[c_idx] += 1
        score_destruct[d_idx] += current_score
        score_construct[c_idx] += current_score

        # E. Update Weights (Adaptive Segment)
        if iteration % L_S == 0:
            for i in range(d_op_num):
                usage = max(1, count_destruct[i])
                w_destruct[i] = w_destruct[i] * (1 - RHO) + RHO * (score_destruct[i] / usage)
                # Reset segment stats
                score_destruct[i] = 0
                count_destruct[i] = 0

            for i in range(c_op_num):
                usage = max(1, count_construct[i])
                w_construct[i] = w_construct[i] * (1 - RHO) + RHO * (score_construct[i] / usage)
                # Reset segment stats
                score_construct[i] = 0
                count_construct[i] = 0

        # Log
        if iteration % 10 == 0:
            print(
                f"Iter {iteration}, Best: {best_fitness:.2f}, Curr: {current_fitness:.2f}, Time: {time.time() - start_t:.2f}s")

    return best_sol, best_fitness, time.time() - start_t


# ==========================================
# Main Entry Point
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ALNS Solver for AHASP")

    parser.add_argument('--file', type=str, required=True, help="Path to the instance Excel file.")
    parser.add_argument('--iter', type=int, default=100, help="Maximum number of iterations.")
    parser.add_argument('--time', type=int, default=3600, help="Time limit in seconds.")

    args = parser.parse_args()

    best_sol, best_fit, elapsed = run_alns(args.file, args.iter, args.time)

    print("\n" + "=" * 50)
    print(" ALNS RESULTS")
    print("=" * 50)
    print(f"Best Fitness : {best_fit:.4f}")
    print(f"Elapsed Time : {elapsed:.4f}s")
    # print(f"Path Map     : {best_sol.get_path_map()}")
    print("=" * 50)