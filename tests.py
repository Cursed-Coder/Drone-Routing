# IMPORTS
import pandas as pd
import pickle

import ILP
import algorithms as alg
# import ILP as ilp
import numpy as np
import matplotlib.pyplot as plt
import os

N_DELIVERIES = [10]
N_STOP_POINTS = [10]
ZIPF_PARAM = [2]
B = 500  # J
W = 10  # KG
N_DRONES = [1]
E = 1
debug = 0
K = 3


# Define the list of number of deliveries you are varying
# N_DELIVERIES = [10]
# N_STOP_POINTS = [10]
# ZIPF_PARAM = [2]

def load_results(n_deliveries, num_stop_points, theta):
    results = []
    load_name = f"results/result_n{n_deliveries}_t{theta}_s{num_stop_points}.csv"
    if os.path.exists(load_name):
        df = pd.read_csv(load_name)
        results.append(df)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def calculate_average_profits(results):
    avg_profits = {
        "n_deliveries": [],
        "DRA_1_avg_profit": [],
        "DRA_2_avg_profit": [],
        "ILP_avg_profit": [],
        "DRA_1_normalized_profit": [],
        "DRA_2_normalized_profit": []
    }

    for n_deliveries in N_DELIVERIES:
        filtered_results = results[results['Instance'].apply(lambda x: len(eval(x)[0][1]) == n_deliveries)]

        if not filtered_results.empty:
            avg_profit_dra_1 = filtered_results['Algorithm_1_Output'].mean()
            avg_profit_dra_2 = filtered_results['Algorithm_2_Output'].mean()
            avg_profit_ilp = filtered_results['ILP_Output'].mean()

            normalized_profit_dra_1 = avg_profit_dra_1 / avg_profit_ilp if avg_profit_ilp else 0
            normalized_profit_dra_2 = avg_profit_dra_2 / avg_profit_ilp if avg_profit_ilp else 0

            avg_profits["n_deliveries"].append(n_deliveries)
            avg_profits["DRA_1_avg_profit"].append(avg_profit_dra_1)
            avg_profits["DRA_2_avg_profit"].append(avg_profit_dra_2)
            avg_profits["ILP_avg_profit"].append(avg_profit_ilp)
            avg_profits["DRA_1_normalized_profit"].append(normalized_profit_dra_1)
            avg_profits["DRA_2_normalized_profit"].append(normalized_profit_dra_2)

    return pd.DataFrame(avg_profits)


def plot_average_profits(avg_profits):
    plt.figure(figsize=(10, 6))
    plt.plot(avg_profits['n_deliveries'], avg_profits['DRA_1_avg_profit'], label='DRA_1 Avg Profit', marker='o')
    plt.plot(avg_profits['n_deliveries'], avg_profits['DRA_2_avg_profit'], label='DRA_2 Avg Profit', marker='o')
    plt.plot(avg_profits['n_deliveries'], avg_profits['ILP_avg_profit'], label='ILP Avg Profit', marker='o')

    plt.xlabel('Number of Deliveries')
    plt.ylabel('Average Profit')
    plt.title('Average Profits of DRA_1, DRA_2, and ILP varying Number of Deliveries')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_normalized_profits(avg_profits):
    plt.figure(figsize=(10, 6))
    plt.plot(avg_profits['n_deliveries'], avg_profits['DRA_1_normalized_profit'], label='DRA_1 Normalized Profit',
             marker='o')
    plt.plot(avg_profits['n_deliveries'], avg_profits['DRA_2_normalized_profit'], label='DRA_2 Normalized Profit',
             marker='o')

    plt.xlabel('Number of Deliveries')
    plt.ylabel('Normalized Profit (relative to ILP)')
    plt.title('Normalized Profits of DRA_1 and DRA_2 relative to ILP')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    all_results = []
    for n_deliveries in N_DELIVERIES:
        for num_stop_points in N_STOP_POINTS:
            for theta in ZIPF_PARAM:
                results = load_results(n_deliveries, num_stop_points, theta)
                if not results.empty:
                    all_results.append(results)

    if all_results:
        all_results_df = pd.concat(all_results, ignore_index=True)
        avg_profits = calculate_average_profits(all_results_df)

        plot_average_profits(avg_profits)
        plot_normalized_profits(avg_profits)
    else:
        print("No results found.")


if __name__ == "__main__":
    main()


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
                        output_1 = alg.DRA_1(prob[0][0], prob[0][1], num_drones, B, W, E, 1, K, debug)
                        print("***************algo1 ends****************")
                        print("***************algo2 starts****************")
                        # #

                        output_2 = alg.DRA_2(prob[0][0], prob[0][1], num_drones, B, W, E, 1, K, debug)
                        output_3 = alg.DRA_3(prob[0][0], prob[0][1], num_drones, B, W, E, 1, K, debug)

                        print("***************algo2 ends****************")
                        output_ILP = ILP.opt_algo_cplex(prob[0][0], prob[0][1], num_drones, B, W, E, 1, K, debug)

                        print(
                            "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&7")
                        print("output1", output_1)
                        print("output2", output_2)
                        print("output3", output_3)
                        print("outputILP", output_ILP)
                        print(
                            "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&7")

                        # output_ILP = ILP.opt_algo_cplex(prob[0], prob[1], num_drones, B, W, E, K, debug)
                        results = [{
                            "Instance": prob,
                            "Algorithm_1_Output": output_1,
                            "Algorithm_2_Output": output_2,
                            "ILP_Output": output_ILP
                        }]
                        save_name = "results/result_n" + str(n_deliveries) + "_t" + str(theta) + "_s" + str(
                            num_stop_points) + ".csv"
                        df_results = pd.DataFrame(results)
                        df_results.to_csv(save_name, index=False)
                        print(f"Results saved to {save_name}")
