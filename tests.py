# IMPORTS
import pandas as pd
import pickle

import ILP
import algorithms as alg
# import ILP as ilp
import numpy as np

N_DELIVERIES = [4]
N_STOP_POINTS = [4]
ZIPF_PARAM = [2]
B = 100  # J
W = 20  # KG
N_DRONES = [1]
E = 1


def algo_tests():
    global N_DELIVERIES, N_STOP_POINTS, ZIPF_PARAM
    for n_deliveries in N_DELIVERIES:
        for num_stop_points in N_STOP_POINTS:
            for num_drones in N_DRONES:
                for theta in ZIPF_PARAM:
                    load_name = "problems/problem_n" + str(n_deliveries) + "_t" + str(theta) + "_s" + str(
                        num_stop_points) + ".dat"
                    # file = open(name, 'rb')
                    # instances = pickle.load(file)
                    instances = []
                    with open(load_name, 'rb') as f:
                        # loaded_data = []

                        # Continue loading data until the end of file
                        while True:
                            try:
                                loaded_item = pickle.load(f)
                                instances.append(loaded_item)
                            except EOFError:
                                break  # Break the loop when end of file is reached

                        # print(instances)
#
                    # print(instances)
                    # for en in E:
                    #     for w in W:
                    # print(len(instances))
                    for prob in instances:
                        print(prob[0][1])
                        # print("len of instances is ",len(prob))

                        print("***************8lgo1 starts****************")
                        output_1 = alg.DRA_1(prob[0][0], prob[0][1], num_drones, B, W, E)
                        print("***************algo1 ends****************")
                        print("***************algo2 starts****************")
                        # #

                        output_2 = alg.DRA_2(prob[0][0], prob[0][1], num_drones, B, W, E)
                        print("***************algo2 ends****************")
                        output_ILP =  alg.DRA_2(prob[0][0], prob[0][1], num_drones, B, W, E)
                            # ILP.opt_algo_cplex(prob[0], prob[1], num_drones, B, W, E))
                        results = [{
                            "Instance": prob,
                            "Algorithm_1_Output": output_1,
                            "Algorithm_2_Output": output_2,
                            "ILP_Output": output_ILP
                        }]
                        save_name = "results/result_n" + str(n_deliveries) + "_t" + str(theta) + "_s" + str(
                            num_stop_points) + ".dat"
                        df_results = pd.DataFrame(results)
                        df_results.to_csv(save_name, index=False)
                        print(f"Results saved to {save_name}")

                        # print(output)
# def algo_tests():
#     for n_deliveries in N_DELIVERIES:
#         for num_stop_points in N_STOP_POINTS:
#             for theta in ZIPF_PARAM:
#                 name = "problems/problem_n" + str(n_deliveries) + "_t" + str(theta) + "_s" + str(
#                     num_stop_points) + ".pkl"
#                 with open(name, 'rb') as file:
#                     instances = pickle.load(file)
#                 print(instances)
# for prob in instances:
#     print("Deliveries in instance:")
#     # for delivery_id, delivery_info in prob[1].items():
#     #     print(f"Delivery ID: {delivery_id}, Profit: {delivery_info[0]}, Weight: {delivery_info[1]}, Coordinates: {delivery_info[2]}")
#     # print()  # Print a blank line between instances


# import pickle
#
# # Sample data to append
# data_to_append = [1, 2, 3, 4, 5]
#
# # Loop to append data
# for item in data_to_append:
#     with open('your_file.pkl', 'ab') as file:  # Open the file in binary append mode
#         pickle.dump(item, file)  # Dump each item into the file
# print("Data appended successfully.")
#
#

#
