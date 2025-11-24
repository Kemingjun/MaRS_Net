import time
import math
import argparse
import random

from Baseline.Util.generate_init_solution import generate_solution_random
from Baseline.Util.load_data import read_excel
from Baseline.Util.operators import destroy_couple_random, repair_couple_greedy
from Baseline.Util.util import (
    copy_dict_int_dict, copy_dict_int_int,
    remove_, insert_, get_all_position, get_feasible_insert_position,
    cal_fitness, get_T
)
from Baseline.Util.solution import Solution

# ==========================================
# IGA Configuration
# ==========================================
D_NUM_COEFFICIENT = 0.3  # Destruction ratio (30% of tasks)


def perturbation(current_solution, d_num):
    """
    Applies perturbation: Random Destruction + Greedy Repair.
    Used to escape local optima.
    """
    # 1. Destruction
    destroyed_seq, destroyed_path, destroyed_tasks = destroy_couple_random(current_solution, d_num)

    # 2. Construction
    new_solution = repair_couple_greedy(destroyed_seq, destroyed_path, destroyed_tasks, current_solution)

    return new_solution


def local_search(solution, start_t, time_limit):
    """
    Performs Local Search (First Improvement Hill Climbing).
    Iteratively removes and re-inserts tasks to find better positions.
    """
    sequence_map = solution.get_sequence_map()
    path_init_task_map = solution.get_path_init_task_map()

    # Process tasks in random order
    task_list = list(range(1, solution.task_num + 1))
    random.shuffle(task_list)

    for task in task_list:
        # Strict time check inside inner loop
        if time.time() - start_t > time_limit:
            break

        # Baseline for this iteration
        current_best_seq = copy_dict_int_dict(sequence_map)
        current_best_path = copy_dict_int_int(path_init_task_map)
        fitness_min, _, _ = cal_fitness(solution.instance, current_best_seq, current_best_path)

        # 1. Remove task
        remove_(sequence_map, path_init_task_map, task)
        is_improved = False

        # 2. Search Parent positions
        parent_positions = get_all_position(sequence_map, path_init_task_map, task, 'parent')
        for parent_pos in parent_positions:
            temp_seq = copy_dict_int_dict(sequence_map)
            temp_path = copy_dict_int_int(path_init_task_map)
            insert_(temp_seq, temp_path, task, parent_pos, 'parent')

            # 3. Search Child positions
            child_positions = get_feasible_insert_position(temp_seq, temp_path, task, 'child')
            for child_pos in child_positions:
                final_seq = copy_dict_int_dict(temp_seq)
                final_path = copy_dict_int_int(temp_path)
                insert_(final_seq, final_path, task, child_pos, 'child')

                # 4. Evaluate
                fitness, _, _ = cal_fitness(solution.instance, final_seq, final_path)

                if fitness < fitness_min:
                    current_best_seq = final_seq
                    current_best_path = final_path
                    fitness_min = fitness
                    is_improved = True
                    break  # First improvement found

            if is_improved:
                break

        # Apply changes (or revert if no improvement)
        sequence_map = current_best_seq
        path_init_task_map = current_best_path

    return Solution(solution.instance, sequence_map, path_init_task_map)


def run_iga(file_path, max_iter, time_limit):
    """
    Executes the Iterated Greedy Algorithm (IGA).

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
    task_num = len(instance)
    d_num = math.ceil(task_num * D_NUM_COEFFICIENT)
    temperature = get_T(instance)

    # Generate Initial Solution
    current_sol = generate_solution_random(instance)
    current_fitness = current_sol.get_fitness()

    best_sol = current_sol
    best_fitness = current_fitness

    print(f"[*] Start IGA Optimization (Max Iter: {max_iter}, Time Limit: {time_limit}s)")

    # 2. Main Loop
    # --------------------------------------------------
    iteration = 0
    while iteration < max_iter and (time.time() - start_t) < time_limit:
        iteration += 1

        # A. Local Search (Intensification)
        ls_sol = local_search(current_sol, start_t, time_limit)

        # B. Perturbation (Diversification)
        new_sol = perturbation(ls_sol, d_num)
        new_fitness = new_sol.get_fitness()

        # C. Acceptance Criterion
        accepted = False
        if new_fitness < current_fitness:
            accepted = True
            # Update Best
            if new_fitness < best_fitness:
                best_sol = new_sol
                best_fitness = new_fitness
                print(f"Iter {iteration}: New Best Found! Fitness: {best_fitness:.4f}")

        elif new_fitness == current_fitness:
            pass  # Neutral move

        else:
            # Metropolis Criterion
            delta_e = new_fitness - current_fitness
            prob = math.exp(-delta_e / temperature)
            if random.random() < prob:
                accepted = True

        if accepted:
            current_sol = new_sol
            current_fitness = new_fitness

        # Log
        if iteration % 10 == 0:
            print(
                f"Iter {iteration}, Best: {best_fitness:.2f}, Curr: {current_fitness:.2f}, Time: {time.time() - start_t:.2f}s")

    return best_sol, best_fitness, time.time() - start_t


# ==========================================
# Main Entry Point
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IGA Solver for AHASP")

    parser.add_argument('--file', type=str, required=True, help="Path to the instance Excel file.")
    parser.add_argument('--iter', type=int, default=100, help="Maximum number of iterations.")
    parser.add_argument('--time', type=int, default=3600, help="Time limit in seconds.")

    args = parser.parse_args()

    best_sol, best_fit, elapsed = run_iga(args.file, args.iter, args.time)

    print("\n" + "=" * 50)
    print(" IGA RESULTS")
    print("=" * 50)
    print(f"Best Fitness : {best_fit:.4f}")
    print(f"Elapsed Time : {elapsed:.4f}s")
    # print(f"Path Map     : {best_sol.get_path_map()}")
    print("=" * 50)