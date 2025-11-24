import math
from Baseline.Util.parameter_ahasp import paramet_ahasp


# ==========================================
# Deep Copy Utilities
# ==========================================

def copy_set_int(original_set):
    """Deep copy for a set of integers."""
    return set(original_set)


def copy_dict_int_int(original_dict):
    """Shallow copy for dict {int: int}."""
    return original_dict.copy()


def copy_dict_int_list(original_dict):
    """Deep copy for dict {int: list}."""
    # List slicing [:] creates a copy of the list
    return {k: v[:] for k, v in original_dict.items()}


def copy_dict_int_dict(original_dict):
    """Deep copy for dict {int: dict}."""
    # .copy() creates a shallow copy of the inner dict
    return {k: v.copy() for k, v in original_dict.items()}


# ==========================================
# Geometry & Math Utilities
# ==========================================

def cal_distance(source_axis, destination_axis):
    """
    Calculate Manhattan distance between two points.
    Args:
        source_axis: [x, y]
        destination_axis: [x, y]
    """
    return abs(source_axis[0] - destination_axis[0]) + abs(source_axis[1] - destination_axis[1])


def get_T(instance):
    """
    Calculate a baseline time threshold based on average distance.
    """
    sum_dis = 0
    for info in instance:
        source_position = [info[1], info[2]]
        destination_position = [info[3], info[4]]
        sum_dis += cal_distance(source_position, destination_position) / paramet_ahasp.ROBOT_VELOCITY

    # Heuristic formula: Average time per robot * scaling factor
    T = sum_dis / (len(instance) * paramet_ahasp.ROBOT_NUM) * 10
    return T


# ==========================================
# Map Conversion Utilities
# ==========================================

def code2path_map(code):
    """
    Convert genetic algorithm code representation to path map.
    Args:
        code: [parent_code_list, child_code_list]
    Returns:
        path_map: {robot_id: [task_id, ...]}
    """
    parent_code = code[0]
    child_code = code[1]
    path_map = {}

    # Process Parent (Carrier) Code
    # Starts from 1
    parent_index = 1
    path = []
    for task_index, task in enumerate(parent_code):
        if task_index == 0: continue  # Skip dummy head

        if task == 0:
            path_map[parent_index] = path
            path = []
            parent_index += 1
            continue

        path.append(task)
        if task_index == len(parent_code) - 1:
            path_map[parent_index] = path

    # Process Child (Shuttle) Code
    # Starts after the last carrier
    child_index = paramet_ahasp.ROBOT_NUM_LIST[0] + 1
    path = []
    for task_index, task in enumerate(child_code):
        if task_index == 0: continue

        if task == 0:
            path_map[child_index] = path
            path = []
            child_index += 1
            continue

        path.append(task)
        if task_index == len(child_code) - 1:
            path_map[child_index] = path

    return path_map


def path_map2sequence_map(path_map):
    """
    Convert path map to doubly-linked sequence map.
    Returns:
        sequence_map: {task_id: {parent: id, child: id, pre/next pointers...}}
        path_init_task_map: {robot_id: first_task_id}
    """
    sequence_map = {}
    path_init_task_map = {}

    for agv_index in path_map.keys():
        path = path_map[agv_index]

        if not path:
            path_init_task_map[agv_index] = 0
            continue

        for task_index, task in enumerate(path):
            # Retrieve or initialize task info
            task_info = sequence_map.setdefault(task, {})

            is_carrier = agv_index in paramet_ahasp.carrier_list
            role_prefix = 'parent' if is_carrier else 'child'

            task_info[role_prefix] = agv_index

            # Handle Predecessor
            if task_index == 0:
                task_info[f'{role_prefix}_pre_task'] = 0
                path_init_task_map[agv_index] = task
            else:
                task_info[f'{role_prefix}_pre_task'] = path[task_index - 1]

            # Handle Successor
            if task_index == len(path) - 1:
                task_info[f'{role_prefix}_next_task'] = 0
            else:
                task_info[f'{role_prefix}_next_task'] = path[task_index + 1]

    return sequence_map, path_init_task_map


# ==========================================
# Core Algorithm Logic (Fitness & Operations)
# ==========================================

def cal_fitness(instance, sequence_map, path_init_task_map):
    """
    Calculate the fitness (objective value) of a solution using Discrete Event Simulation.
    Returns:
        fitness: Weighted sum of distance and tardiness
        total_distance: Sum of travel distances
        total_tardiness: Sum of tardiness
    """
    total_distance = 0
    total_tardiness = 0

    info_map = {task: {} for task in sequence_map.keys()}
    task_num = len(sequence_map)

    # Sets to track available tasks (Petri Net token logic)
    task_parent_ready_set = set()
    task_child_ready_set = set()

    # Initialize with the first task of each robot
    for path_index, first_task in path_init_task_map.items():
        if first_task != 0:
            if path_index in paramet_ahasp.carrier_list:
                info_map[first_task]['parent_pre_d_time'] = 0
                task_parent_ready_set.add(first_task)
            else:
                info_map[first_task]['child_pre_e_time'] = 0
                task_child_ready_set.add(first_task)

    tasks_processed_count = 0

    while tasks_processed_count < task_num:
        # Find a task ready in both chains (Intersection)
        ready_tasks = task_parent_ready_set.intersection(task_child_ready_set)

        if not ready_tasks:
            print("[Error] No enabled transition. Solution is infeasible (Deadlock).")
            return 1e6, 0, 0

        task_to_calculate = ready_tasks.pop()
        tasks_processed_count += 1

        # 1. Retrieve Predecessor Timing
        parent_pre_d_time = info_map[task_to_calculate]['parent_pre_d_time']
        child_pre_e_time = info_map[task_to_calculate]['child_pre_e_time']

        task_data = sequence_map[task_to_calculate]
        instance_task_data = instance[task_to_calculate - 1]

        # 2. Retrieve Positions
        # Parent Previous Position
        if task_data['parent_pre_task'] == 0:
            parent_pre_pos = list(paramet_ahasp.DEPOT)
        else:
            prev_idx = task_data['parent_pre_task'] - 1
            parent_pre_pos = [instance[prev_idx][3], instance[prev_idx][4]]

        # Child Previous Position
        if task_data['child_pre_task'] == 0:
            child_pre_pos = list(paramet_ahasp.DEPOT)
        else:
            prev_idx = task_data['child_pre_task'] - 1
            child_pre_pos = [instance[prev_idx][3], instance[prev_idx][4]]

        source_pos = [instance_task_data[1], instance_task_data[2]]
        dest_pos = [instance_task_data[3], instance_task_data[4]]

        # 3. Calculate Travel Times
        # t1: Carrier moves to Shuttle (Rendezvous)
        t_meet = cal_distance(parent_pre_pos, child_pre_pos) / paramet_ahasp.ROBOT_VELOCITY
        # t2: Coupled pair moves to Source (Pickup)
        t_pickup = cal_distance(child_pre_pos, source_pos) / paramet_ahasp.ROBOT_VELOCITY
        # t3: Coupled pair moves to Dest (Delivery)
        t_deliver = cal_distance(source_pos, dest_pos) / paramet_ahasp.ROBOT_VELOCITY

        # 4. Calculate Event Times
        # Wait time if Carrier arrives early
        t_idle_parent = max(0, child_pre_e_time - parent_pre_d_time - t_meet)

        # Decouple Time (Completion of Delivery)
        t_decouple = (parent_pre_d_time + t_idle_parent + t_meet +
                      paramet_ahasp.T_couple + t_pickup +
                      paramet_ahasp.T_load + t_deliver +
                      paramet_ahasp.T_decouple)

        # End Time (Shuttle Free)
        t_end = t_decouple + instance_task_data[6]

        # 5. Update Metrics
        task_dist = (t_meet + t_pickup + t_deliver) * paramet_ahasp.ROBOT_VELOCITY
        task_tardiness = max(0, t_end - instance_task_data[5])

        total_distance += task_dist
        total_tardiness += task_tardiness

        # 6. Propagate to Next Tasks
        parent_next = task_data['parent_next_task']
        child_next = task_data['child_next_task']

        if parent_next != 0:
            info_map[parent_next]['parent_pre_d_time'] = t_decouple
            task_parent_ready_set.add(parent_next)

        if child_next != 0:
            info_map[child_next]['child_pre_e_time'] = t_end
            task_child_ready_set.add(child_next)

        # Cleanup handled task from sets (already popped from intersection, but ensure consistency)
        if task_to_calculate in task_parent_ready_set: task_parent_ready_set.remove(task_to_calculate)
        if task_to_calculate in task_child_ready_set: task_child_ready_set.remove(task_to_calculate)

    fitness = total_distance * paramet_ahasp.WEIGHT + total_tardiness * (1 - paramet_ahasp.WEIGHT)
    return fitness, total_distance, total_tardiness


def get_all_position(destroyed_sequence_map, path_init_task_map, destroyed_task, chain):
    """
    Get all potential insertion positions for a task in a specific chain.
    Returns a set of tuples: (task_id, direction).
    direction: 0 = before task_id, 1 = after task_id.
    """
    # Find positions adjacent to existing tasks in the chain
    # Only consider tasks that are already inserted in this chain (is not None)
    candidates = [
        t for t in destroyed_sequence_map.keys()
        if t != destroyed_task and destroyed_sequence_map[t][chain] is not None
    ]

    feasible_position_set = set()
    for t in candidates:
        feasible_position_set.add((t, 0))  # Before t
        feasible_position_set.add((t, 1))  # After t

    # Remove duplicates where (B, 0) is equivalent to (A, 1) if A->B
    # Strategy: If A->B exists, remove (B, 0) and keep (A, 1) to standardize
    to_delete = []
    for position in feasible_position_set:
        task, direction = position
        if direction == 0:  # Check "Before" positions
            pre_task = destroyed_sequence_map[task][f'{chain}_pre_task']
            if pre_task and pre_task != 0:
                if (pre_task, 1) in feasible_position_set:
                    to_delete.append(position)  # Remove "Before current", keep "After previous"

    for pos in to_delete:
        feasible_position_set.discard(pos)

    # Add empty paths as valid positions (0, path_index)
    for path_index, first_task in path_init_task_map.items():
        is_empty = (first_task == 0)
        valid_robot = (chain == 'child' and path_index in paramet_ahasp.shuttle_list) or \
                      (chain == 'parent' and path_index in paramet_ahasp.carrier_list)

        if valid_robot and is_empty:
            feasible_position_set.add((0, path_index))

    return feasible_position_set


def get_feasible_insert_position(destroyed_sequence_map, path_init_task_map, destroyed_task, chain):
    """
    Determine valid insertion positions by filtering out those violating precedence constraints.
    Uses BFS to propagate constraints upstream and downstream.
    """
    feasible_position_set = get_all_position(destroyed_sequence_map, path_init_task_map, destroyed_task, chain)

    # --- Backward Propagation (Prune 'After' positions of predecessors) ---
    pre_to_explore = {destroyed_task}
    pre_explored = set()

    while pre_to_explore:
        current_task = pre_to_explore.pop()

        # Check predecessors in both chains
        predecessors = []
        if destroyed_sequence_map[current_task]['child_pre_task']:
            predecessors.append(('child', destroyed_sequence_map[current_task]['child_pre_task']))
        if destroyed_sequence_map[current_task]['parent_pre_task']:
            predecessors.append(('parent', destroyed_sequence_map[current_task]['parent_pre_task']))

        for chain_type, pre_task in predecessors:
            if pre_task == 0: continue

            # Cannot insert destroyed_task BEFORE its predecessor (Constraint logic)
            feasible_position_set.discard((pre_task, 0))

            # Also check the coupled task in the *current* insertion chain
            couple_key = f'{chain}_pre_task' if chain == 'child' else f'{chain}_pre_task'
            # Note: The logic in original code regarding couple_task seemed specific to finding
            # the corresponding node in the target chain.

            # Simplified logic based on original code structure:
            if chain == 'child':
                couple_task = destroyed_sequence_map[pre_task].get('child_pre_task')
            else:
                couple_task = destroyed_sequence_map[pre_task].get('parent_pre_task')

            if couple_task and couple_task != 0:
                feasible_position_set.discard((couple_task, 1))

            if pre_task not in pre_explored:
                pre_to_explore.add(pre_task)

        pre_explored.add(current_task)

    # --- Forward Propagation (Prune 'Before' positions of successors) ---
    next_to_explore = {destroyed_task}
    next_explored = set()

    while next_to_explore:
        current_task = next_to_explore.pop()

        successors = []
        if destroyed_sequence_map[current_task]['child_next_task']:
            successors.append(('child', destroyed_sequence_map[current_task]['child_next_task']))
        if destroyed_sequence_map[current_task]['parent_next_task']:
            successors.append(('parent', destroyed_sequence_map[current_task]['parent_next_task']))

        for chain_type, next_task in successors:
            if next_task == 0: continue

            feasible_position_set.discard((next_task, 1))

            if chain == 'child':
                couple_task = destroyed_sequence_map[next_task].get('child_next_task')
            else:
                couple_task = destroyed_sequence_map[next_task].get('parent_next_task')

            if couple_task and couple_task != 0:
                feasible_position_set.discard((couple_task, 0))

            if next_task not in next_explored:
                next_to_explore.add(next_task)

        next_explored.add(current_task)

    return feasible_position_set


def insert_(destroyed_sequence_map, path_init_task_map, destroyed_task, insert_position, chain):
    """
    Insert a task into the sequence map at the specified position.
    Args:
        insert_position: Tuple (task_id, direction) or (0, path_index)
        chain: 'parent' or 'child'
    """
    chain_opposite = 'child' if chain == 'parent' else 'parent'

    # Initialize task structure if new
    if destroyed_task not in destroyed_sequence_map:
        destroyed_sequence_map[destroyed_task] = {
            'parent': None, 'child': None,
            'parent_pre_task': None, 'parent_next_task': None,
            'child_pre_task': None, 'child_next_task': None
        }

    # Case 1: Insert relative to an existing task (After/Before)
    if insert_position[0] != 0:
        ref_task = insert_position[0]
        # direction 0 = pre, 1 = next
        direction_str = 'next' if insert_position[1] == 1 else 'pre'
        opposite_dir_str = 'pre' if insert_position[1] == 1 else 'next'

        path_index = destroyed_sequence_map[ref_task][chain]

        # Link Logic
        # old_neighbor is the task currently in the direction of insertion
        old_neighbor = destroyed_sequence_map[ref_task][f'{chain}_{direction_str}_task']

        # Update Reference Task -> New Task
        destroyed_sequence_map[ref_task][f'{chain}_{direction_str}_task'] = destroyed_task

        # Update New Task -> Reference Task
        destroyed_sequence_map[destroyed_task][chain] = path_index
        destroyed_sequence_map[destroyed_task][f'{chain}_{opposite_dir_str}_task'] = ref_task

        # Update New Task -> Old Neighbor (if exists)
        destroyed_sequence_map[destroyed_task][f'{chain}_{direction_str}_task'] = old_neighbor

        if old_neighbor != 0:
            destroyed_sequence_map[old_neighbor][f'{chain}_{opposite_dir_str}_task'] = destroyed_task
        else:
            # If inserting at the very beginning (pre) of the chain, update head map
            if direction_str == 'pre':
                path_init_task_map[path_index] = destroyed_task

    # Case 2: Insert into an empty path or as the first element of a path known by index
    else:
        path_index = insert_position[1]
        path_init_task_map[path_index] = destroyed_task

        destroyed_sequence_map[destroyed_task][chain] = path_index
        destroyed_sequence_map[destroyed_task][f'{chain}_pre_task'] = 0
        destroyed_sequence_map[destroyed_task][f'{chain}_next_task'] = 0

    return destroyed_sequence_map, path_init_task_map


def remove_(sequence_map, path_init_task_map, task):
    """
    Remove a task from the sequence map (doubly linked list deletion).
    Handles both parent and child chains.
    """
    # 1. Update Parent Chain Links
    p_pre = sequence_map[task]['parent_pre_task']
    p_next = sequence_map[task]['parent_next_task']

    if p_pre == 0:
        # Head of the list
        parent_id = sequence_map[task]['parent']
        path_init_task_map[parent_id] = p_next
    else:
        sequence_map[p_pre]['parent_next_task'] = p_next

    if p_next != 0:
        sequence_map[p_next]['parent_pre_task'] = p_pre

    # 2. Update Child Chain Links
    c_pre = sequence_map[task]['child_pre_task']
    c_next = sequence_map[task]['child_next_task']

    if c_pre == 0:
        # Head of the list
        child_id = sequence_map[task]['child']
        path_init_task_map[child_id] = c_next
    else:
        sequence_map[c_pre]['child_next_task'] = c_next

    if c_next != 0:
        sequence_map[c_next]['child_pre_task'] = c_pre

    # 3. Delete Task Data
    del sequence_map[task]

    return sequence_map, path_init_task_map