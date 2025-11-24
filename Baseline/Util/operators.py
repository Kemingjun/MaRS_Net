import random
from Baseline.Util.util import (
    remove_, insert_, get_all_position, get_feasible_insert_position,
    cal_fitness, copy_dict_int_dict, copy_dict_int_int
)
from Baseline.Util.parameter_ahasp import paramet_ahasp
from Baseline.Util.solution import Solution


# ==========================================
# Helper Functions (Internal Logic)
# ==========================================

def _ensure_fitness_calculated(solution):
    """Ensures the solution has fitness and info_map calculated."""
    if solution.info_map is None or not solution.info_map.get(1):
        solution.get_fitness()


def _destroy_by_metric(solution, d_num, metric_key_func, reverse=True):
    """
    Generic destroy operator based on a sorting metric.

    Args:
        solution: Current solution object.
        d_num: Number of tasks to destroy.
        metric_key_func: Lambda function to extract the metric value for a task.
        reverse: True for descending sort (remove largest values), False for ascending.
    """
    _ensure_fitness_calculated(solution)

    sequence_map = solution.get_sequence_map()
    path_init_task_map = solution.get_path_init_task_map()
    task_info_map = solution.info_map

    # Calculate/Extract metric for all tasks
    task_metric_list = []
    for task in sequence_map.keys():
        val = metric_key_func(task, task_info_map)
        task_metric_list.append((task, val))

    # Sort and select top d_num tasks
    task_metric_list.sort(key=lambda x: x[1], reverse=reverse)
    destroyed_task_list = [item[0] for item in task_metric_list[:d_num]]

    # Remove selected tasks
    for task in destroyed_task_list:
        remove_(sequence_map, path_init_task_map, task)

    return sequence_map, path_init_task_map, destroyed_task_list


def _greedy_insert_single_task(instance, sequence_map, path_init_task_map, task_to_insert):
    """
    Finds the best insertion position for a single task by exhaustive search
    (Greedy strategy). Returns the updated maps with the task inserted.
    """
    best_seq_map = None
    best_path_map = None
    min_fitness = float('inf')

    # 1. Try all possible positions in Parent Chain
    parent_positions = get_all_position(sequence_map, path_init_task_map, task_to_insert, 'parent')

    for parent_pos in parent_positions:
        # Create temporary copy for parent insertion
        temp_seq = copy_dict_int_dict(sequence_map)
        temp_path = copy_dict_int_int(path_init_task_map)
        insert_(temp_seq, temp_path, task_to_insert, parent_pos, 'parent')

        # 2. Try all feasible positions in Child Chain
        child_positions = get_feasible_insert_position(temp_seq, temp_path, task_to_insert, 'child')

        for child_pos in child_positions:
            # Create temporary copy for child insertion
            final_seq = copy_dict_int_dict(temp_seq)
            final_path = copy_dict_int_int(temp_path)
            insert_(final_seq, final_path, task_to_insert, child_pos, 'child')

            # 3. Evaluate Fitness
            fitness, _, _ = cal_fitness(instance, final_seq, final_path)

            if fitness < min_fitness:
                min_fitness = fitness
                best_seq_map = final_seq
                best_path_map = final_path

    if best_seq_map is None:
        print(f"[Warning] No feasible insertion found for task {task_to_insert}.")
        # Fallback: return originals (conceptually shouldn't happen if logic is correct)
        return sequence_map, path_init_task_map

    return best_seq_map, best_path_map


def _apply_greedy_repair(current_solution, destroyed_sequence_map, path_init_task_map, destroyed_task_list):
    """
    Applies greedy insertion for the list of tasks in the provided order.
    """
    instance = current_solution.instance

    for task in destroyed_task_list:
        destroyed_sequence_map, path_init_task_map = _greedy_insert_single_task(
            instance, destroyed_sequence_map, path_init_task_map, task
        )

    return Solution(instance, destroyed_sequence_map, path_init_task_map)


# ==========================================
# Destroy Operators
# ==========================================

def destroy_couple_random(solution, d_num):
    """
    Randomly removes d_num tasks from both Carrier and Shuttle chains.
    """
    path_init_task_map = solution.get_path_init_task_map()
    destroyed_sequence_map = solution.get_sequence_map()

    all_tasks = list(range(1, solution.task_num + 1))
    destroyed_task_list = random.sample(all_tasks, d_num)

    for task in destroyed_task_list:
        remove_(destroyed_sequence_map, path_init_task_map, task)

    return destroyed_sequence_map, path_init_task_map, destroyed_task_list


def destroy_couple_worst_cost(solution, d_num):
    """
    Removes tasks with the highest weighted cost (Distance + Tardiness).
    """
    # Lambda: Calculate cost
    get_cost = lambda t, info: (info[t]['distance'] * paramet_ahasp.WEIGHT +
                                info[t]['tardiness'] * (1 - paramet_ahasp.WEIGHT))

    return _destroy_by_metric(solution, d_num, get_cost, reverse=True)


def destroy_couple_worst_distance(solution, d_num):
    """
    Removes tasks with the highest travel distance.
    """
    get_dist = lambda t, info: info[t]['distance']
    return _destroy_by_metric(solution, d_num, get_dist, reverse=True)


def destroy_couple_worst_tardiness(solution, d_num):
    """
    Removes tasks with the highest tardiness.
    """
    get_tardiness = lambda t, info: info[t]['tardiness']
    return _destroy_by_metric(solution, d_num, get_tardiness, reverse=True)


# ==========================================
# Repair Operators
# ==========================================

def repair_couple_greedy(destroyed_sequence_map, path_init_task_map, destroyed_task_list, current_solution):
    """
    Greedy Repair: Inserts tasks in random order, but each insertion
    finds the locally optimal position.
    """
    random.shuffle(destroyed_task_list)
    return _apply_greedy_repair(current_solution, destroyed_sequence_map, path_init_task_map, destroyed_task_list)


def repair_couple_greedy_cost_priority(destroyed_sequence_map, path_init_task_map, destroyed_task_list,
                                       current_solution):
    """
    Greedy Repair with Cost Priority: Inserts tasks with higher original costs first.
    """
    current_info_map = current_solution.info_map

    # Sort by original cost descending
    destroyed_task_list.sort(key=lambda t: current_info_map[t]["cost"], reverse=True)

    return _apply_greedy_repair(current_solution, destroyed_sequence_map, path_init_task_map, destroyed_task_list)


def repair_couple_greedy_urgency_priority(destroyed_sequence_map, path_init_task_map, destroyed_task_list,
                                          current_solution):
    """
    Greedy Repair with Urgency Priority: Inserts tasks with earlier due dates first.
    """
    instance = current_solution.instance

    # Sort by due date (index 5) ascending (earlier date = more urgent)
    destroyed_task_list.sort(key=lambda t: instance[t - 1][5], reverse=False)

    return _apply_greedy_repair(current_solution, destroyed_sequence_map, path_init_task_map, destroyed_task_list)


def repair_couple_random(destroyed_sequence_map, path_init_task_map, destroyed_task_list, current_solution):
    """
    Random Repair: Inserts tasks into random valid positions.
    """
    instance = current_solution.instance

    for destroyed_task in destroyed_task_list:
        # Randomly decide which chain to insert first (Parent vs Child)
        insert_chain_order = random.sample(['parent', 'child'], 2)
        first_chain = insert_chain_order[0]
        second_chain = insert_chain_order[1]

        # 1. Insert into first chain (random valid position)
        all_positions = get_all_position(destroyed_sequence_map, path_init_task_map, destroyed_task, first_chain)
        # Note: random.sample on set returns a list
        pos_first = random.sample(list(all_positions), 1)[0]
        insert_(destroyed_sequence_map, path_init_task_map, destroyed_task, pos_first, first_chain)

        # 2. Insert into second chain (random feasible position)
        feasible_positions = get_feasible_insert_position(destroyed_sequence_map, path_init_task_map, destroyed_task,
                                                          second_chain)

        if not feasible_positions:
            print(f"[Warning] No feasible random insertion for task {destroyed_task} in {second_chain} chain.")
            continue

        pos_second = random.sample(list(feasible_positions), 1)[0]
        insert_(destroyed_sequence_map, path_init_task_map, destroyed_task, pos_second, second_chain)

    return Solution(instance, destroyed_sequence_map, path_init_task_map)