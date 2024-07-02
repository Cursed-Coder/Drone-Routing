import copy
import pandas as pd
import pickle
import ILP
import algorithms as alg
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# N_DELIVERIES = [5, 10, 15, 20]

N_DELIVERIES = [5, 10, 15, 20, 40, 60, 80, 100]
# N_DELIVERIES = [ 100 ]
N_STOP_POINTS = [30]
ZIPF_PARAM = [2]
N_DRONES = [7]
E = 44.54
delta = -133.33

debug = 0
K = 3
uav_weight_capacity_range = [3, 5]
uav_battery_capacity_range = [3000, 5000]


class UAV:
    def __init__(self, id, battery_limit, weight_capacity, weight):
        self.id = id
        self.battery_limit = battery_limit
        self.weight_capacity = weight_capacity
        self.current_battery = battery_limit
        self.weight = weight
        self.route_ranges = []  # Keeps track of (start, end) ranges for this UAV's routes

    def __repr__(self):
        return f"UAV(id={self.id}, battery_limit={self.battery_limit}, weight_capacity={self.weight_capacity}, current_battery={self.current_battery})"


def generate_uavs(num_drones, weight_capacity_range, battery_capacity_range):
    uavs = []
    for i in range(num_drones):
        weight_capacity = random.randint(weight_capacity_range[0], weight_capacity_range[1])
        battery_limit = random.randint(battery_capacity_range[0], battery_capacity_range[1])
        uav = UAV(i, battery_limit, weight_capacity, 6.3)
        uavs.append(uav)
    return uavs


def algo_tests():
    for num_drones in N_DRONES:
        global N_DELIVERIES, N_STOP_POINTS, ZIPF_PARAM
        uavs = generate_uavs(num_drones, uav_weight_capacity_range, uav_battery_capacity_range)
        for n_deliveries in N_DELIVERIES:
            for num_stop_points in N_STOP_POINTS:
                for theta in ZIPF_PARAM:
                    load_name = "problems-2/problem_n" + str(n_deliveries) + "_t" + str(theta) + "_s" + str(
                        num_stop_points) + ".dat"
                    instances = []
                    with open(load_name, 'rb') as f:
                        while True:
                            try:
                                loaded_item = pickle.load(f)
                                instances.append(loaded_item)
                            except EOFError:
                                break
                    # print("instances:", instances)
                    iter_count = 0
                    iter_total = len(instances)
                    results = pd.DataFrame(
                        columns=["DRA-1_profit", "DRA-2_profit", "DRA-3_profit", "ILP_profit"])

                    for prob in instances:

                        uav_s = copy.deepcopy(uavs)

                        print("Generated UAVs:", uavs)

                        print("***************algo1 starts****************")
                        output_1 = alg.DRA_1(prob[0][0], prob[0][1], uav_s, E, delta, K, debug)
                        print("***************algo1 ends****************")
                        uav_s = copy.deepcopy(uavs)

                        output_2 = alg.DRA_2(prob[0][0], prob[0][1], uav_s, E, delta, K, debug)
                        uav_s = copy.deepcopy(uavs)
                        print("***************algo2 ends****************")

                        print("***************algo3 starts****************")
                        output_3 = alg.DRA_3(prob[0][0], prob[0][1], uav_s, E, delta, K, debug)
                        print("***************algo3 ends****************")
                        uav_s = copy.deepcopy(uavs)
                        if num_drones > 1:
                            max_count = 15
                        else:
                            max_count = 20

                        if n_deliveries < max_count:
                            print("***************optimal starts****************")
                            print("Running iteration ", iter_count, "/", iter_total, "for drone count", num_drones,
                                  "and for stop points count", num_stop_points, "and num deliveries", n_deliveries)
                            iter_count = iter_count + 1
                            output_ILP = ILP.opt_algo_cplex(prob[0][0], prob[0][1], uav_s, E, delta, K, debug)
                        else:
                            output_ILP = None
                        print("***************optimal ends****************")

                        #
                        # print(
                        #     "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&7")
                        # print("output1", output_1)
                        # print("output2", output_2)
                        # print("output3", output_3)
                        # print("outputILP", output_ILP)
                        # # print("outputILP_2", output_ILP_2)
                        # print(
                        #     "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&7")

                        # results = [{
                        #     "Instance": prob,
                        #     "N_Deliveries": n_deliveries,
                        #     "Algorithm_1_Reward": output_1,
                        #     "Algorithm_2_Reward": output_2,
                        #     "Algorithm_3_Reward": output_3,
                        #     "ILP_Reward": output_ILP
                        # }]
                        to_append = [output_1, output_2, output_3, output_ILP]
                        results.loc[len(results)] = to_append
                        # a_series = pd.Series(to_append, index=results.columns)
                        # results = results.append(a_series, ignore_index=True)
                    save_name = "results-3/result_d" + str(num_drones) + "_n" + str(n_deliveries) + "_s" + str(
                        num_stop_points) + ".csv"
                    # df_results = pd.DataFrame(results)
                    results.to_csv(save_name)
                    print(f"Results saved to {save_name}")


def load_results(N_DELIVERIES, num_stop_points, num_drones):
    rewards = {
        "N_Deliveries": [],
        "Algorithm_1_Reward": [],
        "Algorithm_2_Reward": [],
        "Algorithm_3_Reward": [],
        "ILP_Reward": []
    }
    compact_csv = pd.DataFrame(
        columns=["x", "DRA-1_mean", "DRA-1_std", "DRA-2_mean", "DRA-2_std", "DRA-3_mean",
                 "DRA-3_std", "ILP_mean", "ILP_std"])

    for n_deliveries in N_DELIVERIES:
        file_name = f"results-3/result_d{num_drones}_n{n_deliveries}_s{num_stop_points}.csv"
        if os.path.exists(file_name):
            df = pd.read_csv(file_name)
            DRA_1_mean = df["DRA-1_profit"].mean()
            DRA_2_mean = df["DRA-2_profit"].mean()
            DRA_3_mean = df["DRA-3_profit"].mean()
            DRA_1_std = df["DRA-1_profit"].std()
            DRA_2_std = df["DRA-2_profit"].std()
            DRA_3_std = df["DRA-3_profit"].std()

            ILP_mean = df["ILP_profit"].mean() if df["ILP_profit"].notna().any() else None
            ILP_std = df["ILP_profit"].std() if df["ILP_profit"].notna().any() else None

            to_append = [n_deliveries, DRA_1_mean, DRA_1_std, DRA_2_mean, DRA_2_std, DRA_3_mean, DRA_3_std]

            if ILP_mean is not None and ILP_std is not None:
                to_append.extend([ILP_mean, ILP_std])
            else:
                to_append.extend([None, None])

            compact_csv.loc[len(compact_csv)] = to_append

    # print(compact_csv)
    return compact_csv


def plot_means_and_stds(compact_csv):
    line_len = 0.8  # Length of the horizontal line for std deviation caps
    marker_size = 20  # Marker size

    plt.figure(figsize=(10, 10))

    # Markers for different algorithms
    markers = {
        'DRA-1': 'o',
        'DRA-2': 's',
        'DRA-3': 'D',
        'ILP': 'x'
    }

    # Plot DRA-1 mean and vertical lines for std
    plt.plot(compact_csv['x'], compact_csv['DRA-1_mean'], label='DRA-1 Mean', marker=markers['DRA-1'],
             markersize=marker_size)
    for i, x in enumerate(compact_csv['x']):
        mean = compact_csv['DRA-1_mean'][i]
        std = compact_csv['DRA-1_std'][i]
        plt.plot([x, x], [mean - std, mean + std], color='blue', alpha=0.5)
        plt.hlines(mean - std, x - line_len, x + line_len, color='blue')
        plt.hlines(mean + std, x - line_len, x + line_len, color='blue')

    # Plot DRA-2 mean and vertical lines for std
    plt.plot(compact_csv['x'], compact_csv['DRA-2_mean'], label='DRA-2 Mean', marker=markers['DRA-2'],
             markersize=marker_size)
    for i, x in enumerate(compact_csv['x']):
        mean = compact_csv['DRA-2_mean'][i]
        std = compact_csv['DRA-2_std'][i]
        plt.plot([x, x], [mean - std, mean + std], color='orange', alpha=0.5)
        plt.hlines(mean - std, x - line_len, x + line_len, color='orange')
        plt.hlines(mean + std, x - line_len, x + line_len, color='orange')

    # Plot DRA-3 mean and vertical lines for std
    plt.plot(compact_csv['x'], compact_csv['DRA-3_mean'], label='DRA-3 Mean', marker=markers['DRA-3'],
             markersize=marker_size)
    for i, x in enumerate(compact_csv['x']):
        mean = compact_csv['DRA-3_mean'][i]
        std = compact_csv['DRA-3_std'][i]
        plt.plot([x, x], [mean - std, mean + std], color='green', alpha=0.5)
        plt.hlines(mean - std, x - line_len, x + line_len, color='green')
        plt.hlines(mean + std, x - line_len, x + line_len, color='green')

    # Plot ILP mean and vertical lines for std if ILP data is available
    if compact_csv['ILP_mean'].notnull().any():
        plt.plot(compact_csv['x'], compact_csv['ILP_mean'], label='ILP Mean', marker=markers['ILP'], color='black',
                 markersize=marker_size)
        for i, x in enumerate(compact_csv['x']):
            if pd.notna(compact_csv['ILP_mean'][i]) and pd.notna(compact_csv['ILP_std'][i]):
                mean = compact_csv['ILP_mean'][i]
                std = compact_csv['ILP_std'][i]
                plt.plot([x, x], [mean - std, mean + std], color='black', alpha=0.5)
                plt.hlines(mean - std, x - line_len, x + line_len, color='black')
                plt.hlines(mean + std, x - line_len, x + line_len, color='black')

    plt.xlabel('Number of Deliveries', fontsize=35)
    plt.ylabel('Reward', fontsize=35)
    plt.title('Mean Rewards and Standard Deviations of Algorithms', fontsize=16)

    plt.legend(fontsize=22)
    plt.grid(True)

    plt.tick_params(axis='both', which='major', labelsize=35)
    plt.tick_params(axis='both', which='minor', labelsize=35)

    plt.show()


# algo_tests()
compact_csv = load_results(N_DELIVERIES, N_STOP_POINTS[0], N_DRONES[0])
# plot_means(compact_csv)
plot_means_and_stds(compact_csv)
