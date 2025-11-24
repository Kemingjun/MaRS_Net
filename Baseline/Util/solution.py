from Baseline.Util.util import cal_distance, copy_dict_int_list, copy_dict_int_int, copy_dict_int_dict
from Baseline.Util.parameter_ahasp import paramet_ahasp


class Solution:
    def __init__(self, instance, sequence_map, path_init_task_map):
        """
        Initialize the Solution object.

        Args:
            instance (list): The problem instance data.
            sequence_map (dict): The linked list structure of tasks.
            path_init_task_map (dict): The starting task for each robot.
        """
        self.instance = instance
        self.sequence_map = sequence_map
        self.path_init_task_map = path_init_task_map

        # Evaluation metrics
        self.distance = None
        self.tardiness = None
        self.fitness = None
        self.feasible = None
        self.path_map = None

        # Determine number of tasks based on sequence map keys
        self.task_num = len(self.sequence_map)

        # Dictionary to store detailed timing and cost info for each task
        self.info_map = {task: {} for task in self.sequence_map.keys()}

        # Breakdown of distances (kept for compatibility with original logic)
        self.fixed_distance = 0
        self.unfixed_distance = 0

        # Generate genetic code representation and hash
        self.code = self.get_code()
        self.hash_key = hash(tuple(self.code[0] + self.code[1]))

    def get_path_map(self):
        """
        Reconstructs the full path (list of tasks) for each robot from the linked list.
        """
        if self.path_map is not None:
            return copy_dict_int_list(self.path_map)

        path_map = {}

        for path_index, init_task in self.path_init_task_map.items():
            if init_task == 0:
                # Empty path for this robot
                path_map[path_index] = []
                continue

            # Determine chain direction based on robot type
            if path_index in paramet_ahasp.carrier_list:
                next_key = 'parent_next_task'
            else:
                next_key = 'child_next_task'

            # Traverse the linked list
            path = [init_task]
            current_task = init_task
            while True:
                next_task = self.sequence_map[current_task][next_key]
                if next_task == 0:
                    break
                path.append(next_task)
                current_task = next_task

            path_map[path_index] = path

        self.path_map = path_map
        return path_map

    def get_code(self):
        """
        Generates the genetic code representation (two lists: one for carriers, one for shuttles).
        0 represents a separator between different robots' paths.
        """
        code_parent = [0]
        code_child = [0]

        for path_index, init_task in self.path_init_task_map.items():
            # Determine list and chain type based on robot
            if path_index in paramet_ahasp.carrier_list:
                target_list = code_parent
                next_key = 'parent_next_task'
            else:
                target_list = code_child
                next_key = 'child_next_task'

            if init_task != 0:
                path = [init_task]
                current_task = init_task
                while True:
                    next_task = self.sequence_map[current_task][next_key]
                    if next_task == 0:
                        break
                    path.append(next_task)
                    current_task = next_task
                target_list.extend(path)

            # Add separator (0) unless it's the last robot of that type
            # Note: Checking against ROBOT_NUM_LIST[0] (last carrier) and ROBOT_NUM (last shuttle)
            is_last_carrier = (path_index == paramet_ahasp.ROBOT_NUM_LIST[0])
            is_last_shuttle = (path_index == paramet_ahasp.ROBOT_NUM)

            if not is_last_carrier and not is_last_shuttle:
                target_list.append(0)

        return [code_parent, code_child]

    def get_fitness(self):
        """
        Calculates the fitness of the solution using a discrete event simulation approach.
        Computes total distance and total tardiness.
        """
        if self.fitness is not None:
            return self.fitness

        total_distance = 0
        total_tardiness = 0
        self.fixed_distance = 0
        self.unfixed_distance = 0

        # Sets to track available tasks for processing
        # A task is ready when it is available in both parent (carrier) and child (shuttle) flows
        task_parent_ready_set = set()
        task_child_ready_set = set()

        # Initialize ready sets with the first tasks of each robot
        for path_index, first_task in self.path_init_task_map.items():
            if first_task != 0:
                if path_index in paramet_ahasp.carrier_list:
                    self.info_map[first_task]['parent_pre_d_time'] = 0
                    task_parent_ready_set.add(first_task)
                else:
                    self.info_map[first_task]['child_pre_e_time'] = 0
                    task_child_ready_set.add(first_task)

        tasks_processed_count = 0

        while tasks_processed_count < self.task_num:
            # Find a task that is ready in both chains (Intersection of sets)
            ready_tasks = task_parent_ready_set.intersection(task_child_ready_set)

            if not ready_tasks:
                print("Error: No enabled transition found. Solution is infeasible/deadlocked.")
                self.feasible = False
                self.fitness = 1e6
                return self.fitness

            # Pop an arbitrary ready task
            task_to_calculate = ready_tasks.pop()

            # 1. Retrieve Predecessor Timing
            parent_pre_d_time = self.info_map[task_to_calculate][
                'parent_pre_d_time']  # Carrier finished previous decouple
            child_pre_e_time = self.info_map[task_to_calculate]['child_pre_e_time']  # Shuttle finished previous end

            # 2. Retrieve Locations
            # Data format assumption: instance[i] = [id, src_x, src_y, dst_x, dst_y, due, service]
            task_idx = task_to_calculate - 1
            task_data = self.instance[task_idx]

            source_position = [task_data[1], task_data[2]]
            dest_position = [task_data[3], task_data[4]]

            # Parent (Carrier) previous location
            prev_parent_task_id = self.sequence_map[task_to_calculate]['parent_pre_task']
            if prev_parent_task_id == 0:
                parent_pre_position = list(paramet_ahasp.DEPOT)
            else:
                p_data = self.instance[prev_parent_task_id - 1]
                parent_pre_position = [p_data[3], p_data[4]]  # Previous drop-off location

            # Child (Shuttle) previous location
            prev_child_task_id = self.sequence_map[task_to_calculate]['child_pre_task']
            if prev_child_task_id == 0:
                child_pre_position = list(paramet_ahasp.DEPOT)
            else:
                c_data = self.instance[prev_child_task_id - 1]
                child_pre_position = [c_data[3], c_data[4]]  # Previous drop-off location

            # 3. Calculate Travel Times
            # Segment 1: Carrier moves from prev location to Shuttle's location (rendezvous)
            dist_p_to_c = cal_distance(parent_pre_position, child_pre_position)
            t_meet = dist_p_to_c / paramet_ahasp.ROBOT_VELOCITY

            # Segment 2: Coupled moves to Source
            dist_c_to_src = cal_distance(child_pre_position, source_position)
            t_to_source = dist_c_to_src / paramet_ahasp.ROBOT_VELOCITY

            # Segment 3: Coupled moves Source to Dest
            dist_src_to_dst = cal_distance(source_position, dest_position)
            t_transport = dist_src_to_dst / paramet_ahasp.ROBOT_VELOCITY

            # 4. Calculate Event Times
            # Carrier waits if it arrives before Shuttle is ready
            t_idle_parent = max(0, child_pre_e_time - parent_pre_d_time - t_meet)

            # Decouple Time (Completion of transport)
            # = Start + Idle + Meet + Couple_Op + Move_Source + Load_Op + Move_Dest + Decouple_Op
            t_d = (parent_pre_d_time + t_idle_parent + t_meet +
                   paramet_ahasp.T_couple + t_to_source +
                   paramet_ahasp.T_load + t_transport +
                   paramet_ahasp.T_decouple)

            # End Time (Shuttle is free) = Decouple Time + Service Time
            t_e = t_d + task_data[6]

            # 5. Store Information
            self.info_map[task_to_calculate].update({
                'parent_start_time': parent_pre_d_time + t_idle_parent,
                'attach_time': parent_pre_d_time + t_idle_parent + t_meet,
                'parent_decouple_time': t_d,
                'child_end_time': t_e,
                'misalignment': abs(child_pre_e_time - parent_pre_d_time - t_meet)
            })

            # 6. Calculate Costs
            task_dist = (t_meet + t_to_source + t_transport) * paramet_ahasp.ROBOT_VELOCITY
            task_tardiness = max(0, t_e - task_data[5])  # max(0, end - due_date)

            task_cost = (task_dist * paramet_ahasp.WEIGHT +
                         task_tardiness * (1 - paramet_ahasp.WEIGHT))

            self.info_map[task_to_calculate]['distance'] = task_dist
            self.info_map[task_to_calculate]['tardiness'] = task_tardiness
            self.info_map[task_to_calculate]['cost'] = task_cost

            total_distance += task_dist
            total_tardiness += task_tardiness

            # Accumulate split distances (Fixed: Source->Dest, Unfixed: Empty travel)
            self.fixed_distance += (t_transport * paramet_ahasp.ROBOT_VELOCITY)
            self.unfixed_distance += ((t_meet + t_to_source) * paramet_ahasp.ROBOT_VELOCITY)

            # 7. Update State for Next Iteration
            parent_next_task = self.sequence_map[task_to_calculate]['parent_next_task']
            child_next_task = self.sequence_map[task_to_calculate]['child_next_task']

            if parent_next_task != 0:
                self.info_map[parent_next_task]['parent_pre_d_time'] = t_d
                task_parent_ready_set.add(parent_next_task)

            if child_next_task != 0:
                self.info_map[child_next_task]['child_pre_e_time'] = t_e
                task_child_ready_set.add(child_next_task)

            # Remove current task from sets (it was already popped from intersection, 
            # but we must ensure it's removed from the source sets if we used logic 
            # other than intersection-pop)
            if task_to_calculate in task_parent_ready_set:
                task_parent_ready_set.remove(task_to_calculate)
            if task_to_calculate in task_child_ready_set:
                task_child_ready_set.remove(task_to_calculate)

            tasks_processed_count += 1

        # Final Fitness Calculation
        self.fitness = total_distance * paramet_ahasp.WEIGHT + total_tardiness * (1 - paramet_ahasp.WEIGHT)
        self.distance = total_distance
        self.tardiness = total_tardiness
        self.feasible = True

        return self.fitness

    def get_path_init_task_map(self):
        """Returns a deep copy of the path initialization map."""
        return copy_dict_int_int(self.path_init_task_map)

    def get_sequence_map(self):
        """Returns a deep copy of the sequence map."""
        return copy_dict_int_dict(self.sequence_map)