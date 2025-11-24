class paramet_ahasp:
    """
    Configuration parameters for the AHASP (Carrier-Shuttle System).
    """

    # ==========================================
    # 1. Robot Configuration
    # ==========================================
    # Number of robots for each type: [Carriers, Shuttles]
    ROBOT_NUM_LIST = [4, 8]

    # Total number of robots (Calculated automatically)
    ROBOT_NUM = sum(ROBOT_NUM_LIST)

    # Number of robot types (Carrier and Shuttle)
    ROBOT_TYPE_NUM = len(ROBOT_NUM_LIST)

    # Robot Index Lists (1-based index)
    # Carriers: 1 to N_carrier
    carrier_list = list(range(1, ROBOT_NUM_LIST[0] + 1))

    # Shuttles: N_carrier + 1 to Total
    shuttle_list = list(range(ROBOT_NUM_LIST[0] + 1, ROBOT_NUM + 1))

    # Depot Location coordinates [x, y]
    DEPOT = [0, 0]

    # ==========================================
    # 2. Physical & Optimization Parameters
    # ==========================================
    # Robot moving velocity (m/s)
    ROBOT_VELOCITY = 1.2

    # Weight for Objective Function
    # Objective = WEIGHT * Distance + (1 - WEIGHT) * Tardiness
    WEIGHT = 0.4

    # ==========================================
    # 3. Operation Time Parameters (seconds)
    # ==========================================
    # Time required to couple (Carrier + Shuttle)
    T_couple = 8

    # Time required to decouple
    T_decouple = 8

    # Time required for loading operation
    T_load = 30
