import argparse
import logging
import math
import os
import sys
from pathlib import Path

from gurobipy import Model, GRB, quicksum

# Make project root importable (assuming this file is under a subfolder)
CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
sys.path.append(str(PARENT_DIR))

from Baseline.Util.load_data import read_excel  # noqa: E402
from problems.ahasp.paramet_ahasp import paramet_ahasp  # noqa: E402

# Big-M constant for linearization
BIG_M = 10000


def cal_distance(source_axis, destination_axis):
    """Manhattan distance between two points."""
    sx, sy = source_axis
    dx, dy = destination_axis
    return math.fabs(sx - dx) + math.fabs(sy - dy)


def buildModel(instance_name: str,
               time_limit: int = 3600,
               log_file: str = "gurobi_log.txt"):
    """
    Build and solve the AHASP model for a given instance.

    Parameters
    ----------
    instance_name : str
        Path to the Excel instance file.
    time_limit : int, optional
        Gurobi time limit in seconds, by default 3600.
    log_file : str, optional
        Gurobi log file name, by default "gurobi_log.txt".

    Returns
    -------
    dict or None
        Solution summary with objective, distance, tardiness, status, etc.,
        or None if no feasible solution is found within the time limit.
    """
    # Logging header
    logging.info("---------- %s start ----------", instance_name)
    print("====================================================")
    print(f"Start solving instance: {instance_name}")
    print("====================================================")

    # Load instance
    instance = read_excel(instance_name)
    N = len(instance)  # number of real tasks (tasks are 1..N, 0 is dummy)

    # Robot sets
    num_carriers = paramet_ahasp.ROBOT_NUM_LIST[0]
    num_shuttles = paramet_ahasp.ROBOT_NUM_LIST[1]
    carriers = range(1, num_carriers + 1)
    shuttles = range(num_carriers + 1, num_carriers + num_shuttles + 1)
    all_robots = range(1, paramet_ahasp.ROBOT_NUM + 1)

    # Create model
    model = Model("AHASP")

    # Decision variables
    # x[i, j, r] = 1 if robot r goes from task i to task j (i, j in {0..N})
    x = model.addVars(
        range(0, N + 1),
        range(0, N + 1),
        all_robots,
        vtype=GRB.BINARY,
        name="x",
    )

    # t_d[i]: decoupling/start time of task i (including dummy 0)
    t_d = model.addVars(range(0, N + 1), vtype=GRB.CONTINUOUS, name="t_d")

    # t_e[i]: finish/ending time of task i (including dummy 0)
    t_e = model.addVars(range(0, N + 1), vtype=GRB.CONTINUOUS, name="t_e")

    # t_idle_parent[i]: idle time of parent (carrier) before executing task i
    t_idle_parent = model.addVars(
        range(0, N + 1), vtype=GRB.CONTINUOUS, name="t_idle_parent"
    )

    # tardiness[i]: tardiness of task i (i=1..N)
    tardiness = model.addVars(
        range(0, N + 1), vtype=GRB.CONTINUOUS, name="t_tardiness"
    )

    # distance[i]: travel distance cost attributed to task i
    distance = model.addVars(
        range(0, N + 1), vtype=GRB.CONTINUOUS, name="distance"
    )

    # intermediate[i, j, k] = 1 if:
    #   parent (carrier) last task is i,
    #   child (shuttle) last task is j,
    #   and current executed task is k
    intermediate = model.addVars(
        range(0, N + 1),
        range(0, N + 1),
        range(1, N + 1),
        vtype=GRB.BINARY,
        name="intermediate",
    )

    # Aggregate variables
    distance_total = model.addVar(vtype=GRB.CONTINUOUS, name="distance_total")
    tardiness_total = model.addVar(vtype=GRB.CONTINUOUS, name="tardiness_total")

    # ----------------------
    # Objective: weighted sum of distance and tardiness
    # ----------------------
    for i in range(1, N + 1):
        # tardiness[i] >= 0
        model.addConstr(tardiness[i] >= 0, name=f"tardiness_lb_{i}")
        # tardiness[i] >= completion - due_date
        due_date = instance[i - 1][5]
        model.addConstr(
            tardiness[i] >= t_e[i] - due_date,
            name=f"tardiness_def_{i}",
        )

    model.addConstr(
        distance_total == quicksum(distance[i] for i in range(1, N + 1)),
        name="distance_total_def",
    )
    model.addConstr(
        tardiness_total == quicksum(tardiness[i] for i in range(1, N + 1)),
        name="tardiness_total_def",
    )

    model.setObjective(
        distance_total * paramet_ahasp.WEIGHT
        + tardiness_total * (1 - paramet_ahasp.WEIGHT),
        GRB.MINIMIZE,
    )

    # ----------------------
    # Intermediate variable linking
    # ----------------------
    for i in range(0, N + 1):        # parent previous task
        for j in range(0, N + 1):    # child previous task
            for k in range(1, N + 1):  # current task
                # If intermediate[i,j,k] == 1, both carrier and shuttle must go to k
                model.addConstr(
                    intermediate[i, j, k]
                    <= quicksum(x[i, k, r] for r in carriers),
                    name=f"intermediate_parent_ub_{i}_{j}_{k}",
                )
                model.addConstr(
                    intermediate[i, j, k]
                    <= quicksum(x[j, k, r] for r in shuttles),
                    name=f"intermediate_child_ub_{i}_{j}_{k}",
                )
                model.addConstr(
                    intermediate[i, j, k]
                    >= quicksum(x[i, k, r] for r in carriers)
                    + quicksum(x[j, k, r] for r in shuttles)
                    - 1,
                    name=f"intermediate_lb_{i}_{j}_{k}",
                )

    # ----------------------
    # Routing constraints
    # ----------------------

    # No self loops on real tasks: x[i, i, r] = 0 for i=1..N
    for i in range(1, N + 1):
        for r in all_robots:
            model.addConstr(x[i, i, r] == 0, name=f"no_self_loop_{i}_{r}")

    # For each task j, exactly one carrier enters and one shuttle enters
    for j in range(1, N + 1):
        model.addConstr(
            quicksum(x[i, j, r] for i in range(0, N + 1) for r in carriers) == 1,
            name=f"one_carrier_in_{j}",
        )
        model.addConstr(
            quicksum(x[i, j, r] for i in range(0, N + 1) for r in shuttles) == 1,
            name=f"one_shuttle_in_{j}",
        )

    # For each task i, exactly one carrier leaves and one shuttle leaves
    for i in range(1, N + 1):
        model.addConstr(
            quicksum(x[i, j, r] for j in range(0, N + 1) for r in carriers) == 1,
            name=f"one_carrier_out_{i}",
        )
        model.addConstr(
            quicksum(x[i, j, r] for j in range(0, N + 1) for r in shuttles) == 1,
            name=f"one_shuttle_out_{i}",
        )

    # Flow conservation: for each real task and robot, flow in = flow out
    for j in range(1, N + 1):
        for r in all_robots:
            model.addConstr(
                quicksum(x[i, j, r] for i in range(0, N + 1))
                - quicksum(x[j, i, r] for i in range(0, N + 1))
                == 0,
                name=f"flow_conservation_{j}_{r}",
            )

    # Each robot starts from dummy 0 exactly once and returns to 0 exactly once
    for r in all_robots:
        model.addConstr(
            quicksum(x[0, j, r] for j in range(0, N + 1)) == 1,
            name=f"start_once_{r}",
        )
        model.addConstr(
            quicksum(x[i, 0, r] for i in range(0, N + 1)) == 1,
            name=f"end_once_{r}",
        )

    # Dummy task 0 starts at time 0
    model.addConstr(t_d[0] == 0, name="t_d_0")
    model.addConstr(t_e[0] == 0, name="t_e_0")

    # ----------------------
    # Time and distance constraints
    # ----------------------
    for i in range(0, N + 1):        # parent previous task
        for j in range(0, N + 1):    # child previous task
            for k in range(1, N + 1):  # actual executed task
                if i == 0:
                    carrier_position = [0, 0]
                else:
                    carrier_position = [instance[i - 1][3], instance[i - 1][4]]

                if j == 0:
                    shuttle_position = [0, 0]
                else:
                    shuttle_position = [instance[j - 1][3], instance[j - 1][4]]

                task_s_position = [instance[k - 1][1], instance[k - 1][2]]
                task_d_position = [instance[k - 1][3], instance[k - 1][4]]

                # Distance components
                dis_i2j = cal_distance(carrier_position, shuttle_position)
                dis_j2k_s = cal_distance(shuttle_position, task_s_position)
                dis_k_s2k_d = cal_distance(task_s_position, task_d_position)

                # Distance linking: distance[k] >= travel distance if this (i,j,k) is active
                model.addConstr(
                    distance[k]
                    >= dis_i2j + dis_j2k_s + dis_k_s2k_d
                    - BIG_M * (1 - intermediate[i, j, k]),
                    name=f"distance_link_{i}_{j}_{k}",
                )

                # Time components
                t_i2j = dis_i2j / paramet_ahasp.ROBOT_VELOCITY
                t_j2k_s = dis_j2k_s / paramet_ahasp.ROBOT_VELOCITY
                t_k_s2k_d = dis_k_s2k_d / paramet_ahasp.ROBOT_VELOCITY

                # Idle time of parent carrier before executing task k
                model.addConstr(
                    t_idle_parent[k]
                    >= t_e[j] - t_d[i] - t_i2j - BIG_M * (1 - intermediate[i, j, k]),
                    name=f"idle_parent_{i}_{j}_{k}",
                )

                # Time continuity between decoupling/ending of tasks
                model.addConstr(
                    t_d[i]
                    + t_idle_parent[k]
                    + t_i2j
                    + paramet_ahasp.T_couple
                    + t_j2k_s
                    + paramet_ahasp.T_load
                    + t_k_s2k_d
                    + paramet_ahasp.T_decouple
                    - BIG_M * (1 - intermediate[i, j, k])
                    <= t_d[k],
                    name=f"time_link_{i}_{j}_{k}",
                )

    # Parent idle time must be non-negative
    for i in range(1, N + 1):
        model.addConstr(t_idle_parent[i] >= 0, name=f"idle_parent_lb_{i}")

    # Within-task time relationship: ending time >= decoupling + service duration
    for i in range(1, N + 1):
        service_time = instance[i - 1][6]
        model.addConstr(
            t_e[i] >= t_d[i] + service_time,
            name=f"task_time_{i}",
        )

    # ----------------------
    # Gurobi parameters
    # ----------------------
    model.setParam("TimeLimit", time_limit)
    model.setParam("LogFile", log_file)
    model.setParam("MIPFocus", 1)        # feasibility focus
    model.setParam("Heuristics", 0.3)    # try some heuristic strength
    model.setParam("NoRelHeurTime", time_limit / 10)  # dedicate early time to heuristics
    model.setParam("RINS", 1)            # enable RINS heuristic

    # ----------------------
    # Solve
    # ----------------------
    model.optimize()

    if model.SolCount == 0:
        print("No feasible solution found within time limit.")
        logging.warning("No feasible solution for instance: %s", instance_name)
        return None

    # ----------------------
    # Collect results
    # ----------------------
    objective = model.ObjVal
    dist_val = distance_total.X
    tard_val = tardiness_total.X

    print("====================================================")
    print(f"Finished solving instance: {instance_name}")
    print(f"Model status : {model.Status}")
    print(f"Objective    : {objective:.4f}")
    print(f"Total distance  (weighted) part : {dist_val:.4f}")
    print(f"Total tardiness (weighted) part : {tard_val:.4f}")
    print("====================================================")

    logging.info("---------- %s end ----------", instance_name)
    logging.info("Objective: %f", objective)

    return {
        "instance": instance_name,
        "status": int(model.Status),
        "objective": float(objective),
        "total_distance": float(dist_val),
        "total_tardiness": float(tard_val),
        "num_tasks": N,
        "num_carriers": num_carriers,
        "num_shuttles": num_shuttles,
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Solve AHASP scheduling problem with Gurobi."
    )
    parser.add_argument(
        "-i",
        "--instance",
        required=True,
        help="Path to the Excel instance file.",
    )
    parser.add_argument(
        "-t",
        "--time_limit",
        type=int,
        default=3600,
        help="Time limit in seconds for Gurobi (default: 3600).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Basic logging config (INFO level)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename="python_log.txt",
        filemode="w",
    )

    args = parse_args()
    buildModel(
        instance_name=args.instance,
        time_limit=args.time_limit,
    )