import copy
import pandas as pd
import pickle
import ILP
import algorithms as alg
import numpy as np
import matplotlib.pyplot as plt
import os
import random

N_DELIVERIES = [5, 10, 15, 20,30,  40, 60, 80 , 100]
N_STOP_POINTS = [10]
ZIPF_PARAM = [2]
N_DRONES = [3]
E = 44.54
I = -133.33

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
    global N_DELIVERIES, N_STOP_POINTS, ZIPF_PARAM
    for n_deliveries in N_DELIVERIES:
        for num_stop_points in N_STOP_POINTS:
            for num_drones in N_DRONES:
                for theta in ZIPF_PARAM:
                    load_name = "problems/problem_n" + str(n_deliveries) + "_t" + str(theta) + "_s" + str(
                        num_stop_points) + ".dat"
                    instances = []
                    with open(load_name, 'rb') as f:
                        while True:
                            try:
                                loaded_item = pickle.load(f)
                                instances.append(loaded_item)
                            except EOFError:
                                break
                    uavs = generate_uavs(num_drones, uav_weight_capacity_range, uav_battery_capacity_range)

                    for prob in instances:
                        uav_s = copy.deepcopy(uavs)

                        print("Generated UAVs:", uavs)

                        print("***************algo1 starts****************")
                        output_1 = alg.DRA_1(prob[0][0], prob[0][1], uav_s, E, I, K, debug)
                        print("***************algo1 ends****************")
                        uav_s = copy.deepcopy(uavs)

                        output_2 = alg.DRA_2(prob[0][0], prob[0][1], uav_s, E, I, K, debug)
                        uav_s = copy.deepcopy(uavs)
                        print("***************algo2 ends****************")

                        print("***************algo3 starts****************")
                        output_3 = alg.DRA_3(prob[0][0], prob[0][1], uav_s, E, I, K, debug)
                        print("***************algo3 ends****************")
                        uav_s = copy.deepcopy(uavs)

                        if n_deliveries < 20:
                            print("***************optimal starts****************")
                            output_ILP = ILP.opt_algo_cplex(prob[0][0], prob[0][1], uav_s, E, I, K, debug)
                        else:
                            output_ILP = None
                        print("***************optimal ends****************")
                        uav_s = copy.deepcopy(uavs)

                        # print("***************optimal starts****************")
                        # output_ILP_2 = ILP.opt_algo_cplex_2(prob[0][0], prob[0][1], uav_s, E, I, K, debug)
                        # print("***************optimal ends****************")

                        results = [{
                            "Instance": prob,
                            "N_Deliveries": n_deliveries,
                            "Algorithm_1_Reward": output_1,
                            "Algorithm_2_Reward": output_2,
                            "Algorithm_3_Reward": output_3,
                            "ILP_Reward": output_ILP
                        }]

                        print(
                            "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&7")
                        print("output1", output_1)
                        print("output2", output_2)
                        print("output3", output_3)
                        print("outputILP", output_ILP)
                        # print("outputILP_2", output_ILP_2)
                        print(
                            "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&7")

                        results = [{
                            "Instance": prob,
                            "N_Deliveries": n_deliveries,
                            "Algorithm_1_Reward": output_1,
                            "Algorithm_2_Reward": output_2,
                            "Algorithm_3_Reward": output_3,
                            "ILP_Reward": output_ILP
                        }]
                        save_name = "results/result_n" + str(n_deliveries) + "_d" + str(num_drones) + "_s" + str(
                            num_stop_points) + ".csv"
                        df_results = pd.DataFrame(results)
                        df_results.to_csv(save_name, index=False)
                        print(f"Results saved to {save_name}")


# def load_results(N_DELIVERIES, num_stop_points, num_drones):
#     rewards = {
#         "N_Deliveries": [],
#         "Algorithm_1_Reward": [],
#         "Algorithm_2_Reward": [],
#         "Algorithm_3_Reward": [],
#         "ILP_Reward": []
#     }
#     for n_deliveries in N_DELIVERIES:
#         file_name = f"results/result_n{n_deliveries}_d{num_drones}_s{num_stop_points}.csv"
#         if os.path.exists(file_name):
#             df = pd.read_csv(file_name)
#             rewards["N_Deliveries"].append(n_deliveries)
#             rewards["Algorithm_1_Reward"].append(df["Algorithm_1_Reward"].mean())
#             rewards["Algorithm_2_Reward"].append(df["Algorithm_2_Reward"].mean())
#             rewards["Algorithm_3_Reward"].append(df["Algorithm_3_Reward"].mean())
#             rewards["ILP_Reward"].append(df["ILP_Reward"].mean())
#     return rewards

def load_results(N_DELIVERIES, num_stop_points, num_drones):
    rewards = {
        "N_Deliveries": [],
        "Algorithm_1_Reward": [],
        "Algorithm_2_Reward": [],
        "Algorithm_3_Reward": [],
        "ILP_Reward": []
    }
    for n_deliveries in N_DELIVERIES:
        file_name = f"results/result_n{n_deliveries}_d{num_drones}_s{num_stop_points}.csv"
        if os.path.exists(file_name):
            df = pd.read_csv(file_name)
            rewards["N_Deliveries"].append(n_deliveries)
            rewards["Algorithm_1_Reward"].append(df["Algorithm_1_Reward"].mean())
            rewards["Algorithm_2_Reward"].append(df["Algorithm_2_Reward"].mean())
            rewards["Algorithm_3_Reward"].append(df["Algorithm_3_Reward"].mean())
            ilp_rewards = df["ILP_Reward"].dropna()  # Remove None (NaN) values
            if not ilp_rewards.empty:
                rewards["ILP_Reward"].append(ilp_rewards.mean())
            else:
                rewards["ILP_Reward"].append(None)
    return rewards


# def plot_all_rewards(rewards):
#     plt.figure(figsize=(6, 6))
#
#     # Scatter and line plots for Algorithm 1
#     plt.scatter(rewards["N_Deliveries"], rewards["Algorithm_1_Reward"], label="Algorithm 1", alpha=0.6)
#     plt.plot(sorted(rewards["N_Deliveries"]),
#              [rewards["Algorithm_1_Reward"][i] for i in np.argsort(rewards["N_Deliveries"])], linestyle='-', alpha=0.6)
#
#     # Scatter and line plots for Algorithm 2
#     plt.scatter(rewards["N_Deliveries"], rewards["Algorithm_2_Reward"], label="Algorithm 2", alpha=0.6)
#     plt.plot(sorted(rewards["N_Deliveries"]),
#              [rewards["Algorithm_2_Reward"][i] for i in np.argsort(rewards["N_Deliveries"])], linestyle='-', alpha=0.6)
#
#     # Scatter and line plots for Algorithm 3
#     plt.scatter(rewards["N_Deliveries"], rewards["Algorithm_3_Reward"], label="Algorithm 3", alpha=0.6)
#     plt.plot(sorted(rewards["N_Deliveries"]),
#              [rewards["Algorithm_3_Reward"][i] for i in np.argsort(rewards["N_Deliveries"])], linestyle='-', alpha=0.6)
#
#     # Scatter and line plots for ILP
#     plt.scatter(rewards["N_Deliveries"], rewards["ILP_Reward"], label="ILP", alpha=0.6)
#     plt.plot(sorted(rewards["N_Deliveries"]), [rewards["ILP_Reward"][i] for i in np.argsort(rewards["N_Deliveries"])],
#              linestyle='-', alpha=0.6)
#
#     plt.xlabel("Number of Deliveries")
#     plt.ylabel("Reward")
#     plt.title("Rewards vs Number of Deliveries")
#     plt.legend()
#     plt.grid(True)
#     plt.show()
def plot_all_rewards(rewards):
    plt.figure(figsize=(6, 6))

    # Scatter and line plots for Algorithm 1
    plt.scatter(rewards["N_Deliveries"], rewards["Algorithm_1_Reward"], label="Algorithm 1", alpha=0.6)
    plt.plot(sorted(rewards["N_Deliveries"]),
             [rewards["Algorithm_1_Reward"][i] for i in np.argsort(rewards["N_Deliveries"])], linestyle='-', alpha=0.6)

    # Scatter and line plots for Algorithm 2
    plt.scatter(rewards["N_Deliveries"], rewards["Algorithm_2_Reward"], label="Algorithm 2", alpha=0.6)
    plt.plot(sorted(rewards["N_Deliveries"]),
             [rewards["Algorithm_2_Reward"][i] for i in np.argsort(rewards["N_Deliveries"])], linestyle='-', alpha=0.6)

    # Scatter and line plots for Algorithm 3
    plt.scatter(rewards["N_Deliveries"], rewards["Algorithm_3_Reward"], label="Algorithm 3", alpha=0.6)
    plt.plot(sorted(rewards["N_Deliveries"]),
             [rewards["Algorithm_3_Reward"][i] for i in np.argsort(rewards["N_Deliveries"])], linestyle='-', alpha=0.6)

    # Scatter and line plots for ILP
    non_none_indices = [i for i, reward in enumerate(rewards["ILP_Reward"]) if reward is not None]
    plt.scatter([rewards["N_Deliveries"][i] for i in non_none_indices],
                [rewards["ILP_Reward"][i] for i in non_none_indices], label="ILP", alpha=0.6)
    plt.plot(sorted([rewards["N_Deliveries"][i] for i in non_none_indices]),
             [rewards["ILP_Reward"][i] for i in sorted(non_none_indices)], linestyle='-', alpha=0.6)

    plt.xlabel("Number of Deliveries")
    plt.ylabel("Reward")
    plt.title("Rewards vs Number of Deliveries")
    plt.legend()
    plt.grid(True)
    plt.show()


# Plot all rewards against number of drones
def plot_all_rewards_drones(rewards):
    plt.figure(figsize=(10, 6))

    # Scatter and line plots for Algorithm 1
    plt.scatter(rewards["N_Drones"], rewards["Algorithm_1_Reward"], label="Algorithm 1", alpha=0.6)
    plt.plot(sorted(rewards["N_Drones"]), [rewards["Algorithm_1_Reward"][i] for i in np.argsort(rewards["N_Drones"])],
             linestyle='-', alpha=0.6)

    # Scatter and line plots for Algorithm 2
    plt.scatter(rewards["N_Drones"], rewards["Algorithm_2_Reward"], label="Algorithm 2", alpha=0.6)
    plt.plot(sorted(rewards["N_Drones"]), [rewards["Algorithm_2_Reward"][i] for i in np.argsort(rewards["N_Drones"])],
             linestyle='-', alpha=0.6)

    # Scatter and line plots for Algorithm 3
    plt.scatter(rewards["N_Drones"], rewards["Algorithm_3_Reward"], label="Algorithm 3", alpha=0.6)
    plt.plot(sorted(rewards["N_Drones"]), [rewards["Algorithm_3_Reward"][i] for i in np.argsort(rewards["N_Drones"])],
             linestyle='-', alpha=0.6)

    # Scatter and line plots for ILP
    plt.scatter(rewards["N_Drones"], rewards["ILP_Reward"], label="ILP", alpha=0.6)
    plt.plot(sorted(rewards["N_Drones"]), [rewards["ILP_Reward"][i] for i in np.argsort(rewards["N_Drones"])],
             linestyle='-', alpha=0.6)

    plt.xlabel("Number of Drones")
    plt.ylabel("Reward")
    plt.title("Rewards vs Number of Drones")
    plt.legend()
    plt.grid(True)
    plt.show()


# Run the tests and plot the results
# algo_tests()
# rewards = load_all_results(N_DELIVERIES[0], N_STOP_POINTS[0], N_DRONES)
# Run the tests and plot the results
algo_tests()
rewards = load_results(N_DELIVERIES, N_STOP_POINTS[0], N_DRONES[0])
plot_all_rewards(rewards)

# def load_all_results(n_deliveries, num_stop_points, N_DRONES):
#     rewards = {
#         "N_Drones": [],
#         "Algorithm_1_Reward": [],
#         "Algorithm_2_Reward": [],
#         "Algorithm_3_Reward": [],
#         "ILP_Reward": []
#     }
#     for num_drones in N_DRONES:
#         file_name = f"results/result_n{n_deliveries}_d{num_drones}_s{num_stop_points}.csv"
#         if os.path.exists(file_name):
#             df = pd.read_csv(file_name)
#             rewards["N_Drones"].extend([num_drones] * len(df))
#             rewards["Algorithm_1_Reward"].extend(df["Algorithm_1_Reward"])
#             rewards["Algorithm_2_Reward"].extend(df["Algorithm_2_Reward"])
#             rewards["Algorithm_3_Reward"].extend(df["Algorithm_3_Reward"])
#             rewards["ILP_Reward"].extend(df["ILP_Reward"])
#     return rewards
