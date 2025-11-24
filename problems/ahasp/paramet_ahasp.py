import torch


class paramet_ahasp:
    ROBOT_VELOCITY = 1.2

    DEPOT = torch.tensor([0, 0])

    ROBOT_TYPE_NUM = 2

    ROBOT_NUM = 12

    WEIGHT = 0.4

    ROBOT_NUM_LIST = torch.tensor([4, 8])

    T_couple = 8
    T_decouple = 8
    T_load = 30

    location_norm = 100
    time_norm = None  # the maximum value of the deadline
