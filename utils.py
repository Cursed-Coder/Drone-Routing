import math


def select_uav_with_most_battery(uav_s):
    # Sort UAVs by current battery in descending order and return the one with the most battery left
    if not uav_s:
        return None

    sorted_uav_s = sorted(uav_s, key=lambda uav: uav.current_battery, reverse=True)
    return sorted_uav_s[0]


# # Helper function to compute Euclidean distance
def euclidean_distance(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)


# IMPORTS
# import pandas as pd
# import pickle
#
# import ILP
# import algorithms as alg
# # import ILP as ilp
# import numpy as np
# import matplotlib.pyplot as plt
# import os
#
# N_DELIVERIES = [5]
# N_STOP_POINTS = [10]
# ZIPF_PARAM = [2]
# xi = 650
# w_u = [0.25, 0.5, 1]
# # B = 700  # J
# # W = 20  # KG
# N_DRONES = [1, 3, 5]
# E = 36
# debug = 0
# K = 3
# uav_weight_capacity_range = [3, 5]
#
#
# class UAV:
#     def __init__(self, id, battery_limit, weight_capacity, weight):
#         self.id = id
#         self.battery_limit = battery_limit
#         self.weight_capacity = weight_capacity
#         self.current_battery = battery_limit
#         self.weight = weight
#         self.route_ranges = []  # Keeps track of (start, end) ranges for this UAV's routes
#
#     def __repr__(self):
#         return f"UAV(id={self.id}, battery_limit={self.battery_limit}, weight_capacity={self.weight_capacity}, current_battery={self.current_battery})"
#
#
# # Define the list of number of deliveries you are varying
# # N_DELIVERIES = [10]
# # N_STOP_POINTS = [10]
# # ZIPF_PARAM = [2]
#
# def algo_tests():
#     global N_DELIVERIES, N_STOP_POINTS, ZIPF_PARAM
#     for n_deliveries in N_DELIVERIES:
#         for num_stop_points in N_STOP_POINTS:
#             for num_drones in N_DRONES:
#
#                 for theta in ZIPF_PARAM:
#                     load_name = "problems/problem_n" + str(n_deliveries) + "_t" + str(theta) + "_s" + str(
#                         num_stop_points) + ".dat"
#                     # file = open(name, 'rb')
#                     # instances = pickle.load(file)
#                     instances = []
#                     with open(load_name, 'rb') as f:
#                         # loaded_data = []
#
#                         # Continue loading data until the end of file
#                         while True:
#                             try:
#                                 loaded_item = pickle.load(f)
#                                 instances.append(loaded_item)
#                             except EOFError:
#                                 break  # Break the loop when end of file is reached
#
#                         # print(instances)
#                     #
#                     # print(instances)
#                     # for en in E:
#                     #     for w in W:
#                     # print(len(instances))
#                     for prob in instances:
#                         print(prob[0][1])
#                         # print("len of instances is ",len(prob))
#
#                         print("***************8lgo1 starts****************")
#                         output_1 = alg.DRA_1(prob[0][0], prob[0][1], num_drones, B, W, E, 1, K, debug)
#                         print("***************algo1 ends****************")
#                         print("***************algo2 starts****************")
#                         # #
#
#                         output_2 = alg.DRA_2(prob[0][0], prob[0][1], num_drones, B, W, E, 1, K, debug)
#                         output_3 = alg.DRA_3(prob[0][0], prob[0][1], num_drones, B, W, E, 1, K, debug)
#
#                         print("***************algo2 ends****************")
#                         output_ILP = ILP.opt_algo_cplex(prob[0][0], prob[0][1], num_drones, B, W, E, 1, K, debug)
#
#                         print(
#                             "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&7")
#                         print("output1", output_1)
#                         print("output2", output_2)
#                         print("output3", output_3)
#                         print("outputILP", output_ILP)
#                         print(
#                             "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&7")
#
#                         # output_ILP = ILP.opt_algo_cplex(prob[0], prob[1], num_drones, B, W, E, K, debug)
#                         results = [{
#                             "Instance": prob,
#                             "Algorithm_1_Output": output_1,
#                             "Algorithm_2_Output": output_2,
#                             "ILP_Output": output_ILP
#                         }]
#                         save_name = "results/result_n" + str(n_deliveries) + "_t" + str(theta) + "_s" + str(
#                             num_stop_points) + ".csv"
#                         df_results = pd.DataFrame(results)
#                         df_results.to_csv(save_name, index=False)
#                         print(f"Results saved to {save_name}")


# def plot_rewards(rewards):
#     plt.figure(figsize=(10, 6))
#     plt.plot(rewards["N_Deliveries"], rewards["Algorithm_1_Reward"], label="Algorithm 1", marker='o')
#     plt.plot(rewards["N_Deliveries"], rewards["Algorithm_2_Reward"], label="Algorithm 2", marker='o')
#     plt.plot(rewards["N_Deliveries"], rewards["Algorithm_3_Reward"], label="Algorithm 3", marker='o')
#     plt.plot(rewards["N_Deliveries"], rewards["ILP_Reward"], label="ILP", marker='o')
#     plt.xlabel("Number of Deliveries")
#     plt.ylabel("Average Reward")
#     plt.title("Rewards vs Number of Deliveries")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# def plot_all_rewards(rewards):
#     plt.figure(figsize=(10, 6))
#     plt.scatter(rewards["N_Deliveries"], rewards["Algorithm_1_Reward"], label="Algorithm 1", alpha=0.6)
#     plt.scatter(rewards["N_Deliveries"], rewards["Algorithm_2_Reward"], label="Algorithm 2", alpha=0.6)
#     plt.scatter(rewards["N_Deliveries"], rewards["Algorithm_3_Reward"], label="Algorithm 3", alpha=0.6)
#     plt.scatter(rewards["N_Deliveries"], rewards["ILP_Reward"], label="ILP", alpha=0.6)
#     plt.xlabel("Number of Deliveries")
#     plt.ylabel("Reward")
#     plt.title("Rewards vs Number of Deliveries")
#     plt.legend()
#     plt.grid(True)
#     plt.show()
# import copy
#
# import pandas as pd
# import pickle
#
# import ILP
# import algorithms as alg
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import random
#
# N_DELIVERIES = [5,8]
# N_STOP_POINTS = [7]
# ZIPF_PARAM = [2]
# # xi = 650
# # w_u = [5]
# N_DRONES = [1]
# E = 44
# I =  27
#
# debug = 0
# K = 3
# uav_weight_capacity_range = [3, 5]
# uav_battery_capacity_range = [3000, 6000]
#
#
# class UAV:
#     def __init__(self, id, battery_limit, weight_capacity, weight):
#         self.id = id
#         self.battery_limit = battery_limit
#         self.weight_capacity = weight_capacity
#         self.current_battery = battery_limit
#         self.weight = weight
#         self.route_ranges = []  # Keeps track of (start, end) ranges for this UAV's routes
#
#     def __repr__(self):
#         return f"UAV(id={self.id}, battery_limit={self.battery_limit}, weight_capacity={self.weight_capacity}, current_battery={self.current_battery})"
#
#
# def generate_uavs(num_drones, weight_capacity_range, battery_capacity_range):
#     uavs = []
#     for i in range(num_drones):
#         weight_capacity = random.uniform(weight_capacity_range[0],
#                                          weight_capacity_range[1])  # Random value between 3 and 5
#         # battery_weight = random.choice(battery_weights)  # Randomly choose from [0.25, 0.5, 1]
#         battery_limit = random.uniform(battery_capacity_range[0], battery_capacity_range[
#             1])  # Randomly choose from [0.25, 0.5, 1] # Calculate battery limit
#         uav = UAV(i, battery_limit, weight_capacity, 2)  # Create UAV object
#         uavs.append(uav)  # Add UAV to the list
#     return uavs
#
#
# def algo_tests():
#     global N_DELIVERIES, N_STOP_POINTS, ZIPF_PARAM
#     for n_deliveries in N_DELIVERIES:
#         for num_stop_points in N_STOP_POINTS:
#             for num_drones in N_DRONES:
#                 for theta in ZIPF_PARAM:
#                     load_name = "problems/problem_n" + str(n_deliveries) + "_t" + str(theta) + "_s" + str(
#                         num_stop_points) + ".dat"
#                     instances = []
#                     with open(load_name, 'rb') as f:
#                         while True:
#                             try:
#                                 loaded_item = pickle.load(f)
#                                 instances.append(loaded_item)
#                             except EOFError:
#                                 break
#                     uavs = generate_uavs(num_drones, uav_weight_capacity_range, uav_battery_capacity_range)
#                     # print("########################################################################")
#                     # print(uavs)
#                     # print("#########################################################################")
#
#                     for prob in instances:
#                         print(prob[0][1])
#                         uav_s = copy.deepcopy(uavs)
#                         # print("########################################################################")
#                         # print(uavs)
#                         # print("#########################################################################")
#
#                         # Generate UAVs
#
#                         print("Generated UAVs:", uavs)
#
#                         print("***************algo1 starts****************")
#                         output_1 = alg.DRA_1(prob[0][0], prob[0][1], uav_s, E, I, K, debug)
#                         print("***************algo1 ends****************")
#                         uav_s = copy.deepcopy(uavs)
#                         # print("***************algo2 starts****************")
#                         # # print("###lsmldsmlsmdl#####################################################################")
#                         # # print(uavs)
#                         # # print("#####skdnksn#skdnlksdnkdsnk###################################################################")
#                         #
#                         output_2 = alg.DRA_2(prob[0][0], prob[0][1], uav_s, E, I, K, debug)
#                         uav_s = copy.deepcopy(uavs)
#                         output_3 = alg.DRA_3(prob[0][0], prob[0][1], uav_s, E, I, K, debug)
#                         # #
#                         print("***************algo2 ends****************")
#                         uav_s = copy.deepcopy(uavs)
#                         output_ILP = ILP.opt_algo_cplex(prob[0][0], prob[0][1], uav_s, E, I, K, debug)
#
# print(
#     "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&7")
# print("output1", output_1)
# print("output2", output_2)
# print("output3", output_3)
# print("outputILP", output_ILP)
# print(
#     "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&7")

#                         results = [{
#                             "Instance": prob,
#                             "Algorithm_1_Output": output_1,
#                             "Algorithm_2_Output": output_2,
#                             "Algorithm_3_Output": output_3,
#                             "ILP_Output": output_ILP
#                         }]
#                         save_name = "results/result_n" + str(n_deliveries) + "_d" + str(num_drones)  + "_s" + str(
#                             num_stop_points) + ".csv"
#                         df_results = pd.DataFrame(results)
#                         df_results.to_csv(save_name, index=False)
#                         print(f"Results saved to {save_name}")
#
# # if __name__ == "__main__":
# #     algo_tests()