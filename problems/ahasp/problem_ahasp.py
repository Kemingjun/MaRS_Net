from torch.utils.data import Dataset
import torch
import os
import pickle
from pathlib import Path
import pandas as pd

from problems.ahasp.state_ahasp import StateAHASP


class AHASP(object):
    NAME = 'ahasp'  # coupled (attachable) heterogeneous agent scheduling problem

    @staticmethod
    def get_costs(dataset, pi):
        pass

    @staticmethod
    def make_dataset(*args, **kwargs):
        return AHASPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateAHASP.initialize(*args, **kwargs)


class AHASPDataset(Dataset):
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(AHASPDataset, self).__init__()

        self.data = []

        self.data_set = []
        if filename is not None:
            if os.path.splitext(filename)[1] == '.pkl':
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                self.data = [make_instance(args) for args in data[offset:offset + num_samples]]
            else:
                if os.path.splitext(filename)[1] == '.xlsx':
                    file_name = str(Path(__file__).resolve().parent.parent.parent) + "/Instance/" + filename
                    df = pd.read_excel(file_name)
                    instance = [list(row) for index, row in df.iterrows()]

                    source = [[row[1], row[2]] for row in instance]
                    destination = [[row[3], row[4]] for row in instance]
                    deadline = [row[5] for row in instance]
                    operation_time = [row[6] for row in instance]

                    source_tensor = torch.tensor(source, dtype=torch.float)
                    destination_tensor = torch.tensor(destination, dtype=torch.float)
                    deadline_tensor = torch.tensor(deadline, dtype=torch.float)
                    operation_time_tensor = torch.tensor(operation_time, dtype=torch.float)

                    self.data = [{
                        'source': source_tensor,
                        'destination': destination_tensor,
                        'deadline': deadline_tensor,
                        'operation_time': operation_time_tensor,
                    }]
                else:
                    base_dir = Path(__file__).resolve().parent.parent.parent / f"Instance/{filename}"
                    all_files = list(base_dir.glob("*.xlsx"))
                    for file_path in all_files:
                        df = pd.read_excel(file_path)
                        instance = [list(row) for index, row in df.iterrows()]

                        source = [[row[1], row[2]] for row in instance]
                        destination = [[row[3], row[4]] for row in instance]
                        deadline = [row[5] for row in instance]
                        operation_time = [row[6] for row in instance]

                        source_tensor = torch.tensor(source, dtype=torch.float)
                        destination_tensor = torch.tensor(destination, dtype=torch.float)
                        deadline_tensor = torch.tensor(deadline, dtype=torch.float)
                        operation_time_tensor = torch.tensor(operation_time, dtype=torch.float)
                        self.data.append({
                            'source': source_tensor,
                            'destination': destination_tensor,
                            'deadline': deadline_tensor,
                            'operation_time': operation_time_tensor,
                        })


        else:
            ddl_base = 300
            task_indices = torch.arange(size, dtype=torch.float32)

            base = task_indices * 40 + ddl_base
            base_batch = base.unsqueeze(0).repeat(num_samples, 1)
            noise_batch = (torch.rand(num_samples, size) * 2 - 1) * 40
            deadline_unpermuted = base_batch + noise_batch
            permutations = torch.argsort(torch.rand(num_samples, size), dim=1)
            deadline = torch.gather(deadline_unpermuted, 1, permutations)

            '''取放货点坐标生成'''
            x_bound = [0, 100]
            y_bound = [0, 100]

            source_x = torch.rand(num_samples, size, dtype=torch.float32) * x_bound[1]
            source_y = torch.rand(num_samples, size, dtype=torch.float32) * y_bound[1]
            source = torch.stack([source_x, source_y], dim=-1)

            destination_x = torch.rand(num_samples, size, dtype=torch.float32) * x_bound[1]
            destination_y = torch.rand(num_samples, size, dtype=torch.float32) * y_bound[1]
            destination = torch.stack([destination_x, destination_y], dim=-1)  # shape: (num_samples, size, 2)

            operation_time = 60 + 5 * torch.randint(0, 9, (num_samples, size), dtype=torch.int32)

            self.data = []
            for i in range(num_samples):
                self.data.append({
                    'source': source[i],
                    'destination': destination[i],
                    'deadline': deadline[i],
                    'operation_time': operation_time[i],
                })

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


def make_instance(args):
    source, destination, deadline, operation_time, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        'source': torch.tensor(source, dtype=torch.float) / grid_size,
        'destination': torch.tensor(destination, dtype=torch.float) / grid_size,
        'deadline': torch.tensor(deadline, dtype=torch.float),
        'operation_time': torch.tensor(operation_time, dtype=torch.float)
    }
