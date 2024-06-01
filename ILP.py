# Initialize the model
from docplex.mp.model import Model
import numpy as np  # For

import math
import itertools


# class UAV:
#     def __init__(self, id, battery_limit, weight_capacity):
#         self.id = id
#         self.battery_limit = battery_limit
#         self.weight_capacity = weight_capacity
#         self.current_battery = battery_limit
#         self.route_ranges = []  # Keeps track of (start, end) ranges for this UAV's routes
#
#     def __repr__(self):
#         return f"UAV(id={self.id}, battery_limit={self.battery_limit}, weight_capacity={self.weight_capacity}, current_battery={self.current_battery})"
#

def euclidean_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


def opt_algo_cplex(stop_coordinates, deliveries, uav_s, E, I, K, debug):
    model = Model(name="uav_delivery_optimization")
    U = [f"u{uav.id + 1}" for uav in uav_s]
    S = [f"s{idx + 1}" for idx in range(len(stop_coordinates))]
    P = [f"p{idx + 1}" for idx in range(len(deliveries))]

    rewards = {}
    weights = {}
    coordinates = {}

    uav_map = {}
    for i in range(len(uav_s)):
        uav_map[U[i]] = uav_s[i]

    # Iterate over delivery_data dictionary
    for key, value in deliveries.items():
        rewards[key] = value[0]
        weights[key] = value[1]
        coordinates[key] = value[2]
    for key, value in stop_coordinates.items():
        coordinates[key] = value[1]

    len_p = len(P)
    A = S + P
    M = 10000
    # Function to compute Euclidean distance between two points

    # Create a list of all points (stops + deliveries)
    points = list(coordinates.keys())

    # Create a distance matrix (dictionary)
    distance_matrix = {
        (point1, point2): euclidean_distance(coordinates[point1], coordinates[point2])
        for point1, point2 in itertools.product(points, repeat=2)
    }

    # Define binary variables
    alpha = {(a, b, u): model.binary_var(name=f"alpha_{a}_{b}_{u}") for a in S + P for b in S + P for u in U}
    z = {(p, a, b, u): model.binary_var(name=f"z_{p}_{a}_{b}_{u}") for p in P for a in S for b in S for u in U}
    tau = {(a, b): model.binary_var(name=f"tau_{a}_{b}") for a in S + ["depot"] for b in S + ["depot"]}

    w = {(a, b, u): model.binary_var(name=f"w_{a}_{b}_{u}") for a in S for b in S for u in U}

    q = {(a, b, u): model.binary_var(name=f"q_{a}_{b}_{u}") for a in S for b in S for u in U}

    # Define other variables
    # t = {s: model.integer_var(lb=1, ub=len(S), name=f"t_{s}") for s in S}

    y = {(p, a, b, u): model.integer_var(lb=1, ub=len_p, name=f"y_{p}_{a}_{b}_{u}") for p in P for a in S for b in S for
         u in U}

    v = {(p_1, p_2, a, b, u): model.binary_var(name=f"v_{p_1}_{p_2}_{a}_{b}_{u}") for p_1 in P for p_2 in P for a in S
         for b in S for u in U}

    i = {(a_1, a_2, u): model.continuous_var(lb=0, name=f"i_{u}_{a_1}_{a_2}") for a_1 in A for a_2 in A for u in U}
    h = {(a, u): model.continuous_var(lb=0, name=f"h_{a}_{u}") for a in A for u in U}

    # Objective: Maximize profit from deliveries
    model.maximize(model.sum(z[p, a, b, u] * rewards[p] for u in U for p in P for a in S for b in S))
    model.add_constraints(
        model.sum(z[p, a, b, u] for a in S for b in S for u in U) <= 1 for p in P
    )
    model.add_constraints(
        model.sum(z[p, a, b, u] for p in P) <= K for a in S for b in S for u in U
    )

    # model.add_constraints(
    #     model.sum(E * alpha[a, b, u] * distance_matrix[(a, b)] for a in A for b in A) <= B_u for u in U
    # )



    model.add_constraints(
        i[a_1, a_2, u] >= h[a_1, u] - (1 - alpha[a_1, a_2, u]) * M for a_1 in A for a_2 in A for u in U)
    model.add_constraints(i[a_1, a_2, u] <= h[a_1, u] for a_1 in A for a_2 in A for u in U)
    model.add_constraints(i[a_1, a_2, u] <= alpha[a_1, a_2, u] * M for a_1 in A for a_2 in A for u in U)




    #
    #
    #
    #
    model.add_constraints(model.sum(
        (i[a_1, a_2, u] + uav_map[u].weight * alpha[a_1, a_2, u]) * distance_matrix[(a_1, a_2)] * E for a_1 in A for a_2 in
        A) <= uav_map[u].battery_limit for u in U)

    model.add_constraints(h[a, u] == model.sum(z[p, a, b, u] * weights[p] for b in S for p in P) for a in S for u in U)

    model.add_constraints(
        h[a_1, u] <= i[a_2, a_1, u] - alpha[a_2, a_1, u] * weights[a_1] + (1 - alpha[a_2, a_1, u]) * M for a_1 in P for
        a_2 in A for u in U)
    # #
    model.add_constraints(
        h[a_1, u] >= i[a_2, a_1, u] - alpha[a_2, a_1, u] * weights[a_1] for a_1 in P for a_2 in A for u in U)

    model.add_constraints(
        model.sum(z[p, a, b, u] * weights[p] for p in P) <= uav_map[u].weight_capacity for u in U for a in S for b in S
    )

    # Additionally, you may want to ensure the balance between incoming and outgoing flights for a given delivery point
    model.add_constraints(
        model.sum(alpha[p, a, u] for a in A) == model.sum(alpha[b, p, u] for b in A) for p in P for u in U
    )
    # Corrected constraint to ensure the total outgoing and incoming flights are at most one for each UAV and delivery point
    model.add_constraints(
        model.sum(alpha[p, a, u] for a in A) <= 1 for p in P for u in U
    )
    # Corrected constraint to ensure the total outgoing and incoming flights are at most one for each UAV and delivery point
    model.add_constraints(
        model.sum(alpha[b, p, u] for b in A) <= 1 for p in P for u in U
    )
    # Corrected constraint to ensure the total outgoing and incoming flights are at most one for each UAV and delivery point
    model.add_constraints(
        model.sum(alpha[p, a, u] for a in A) == model.max(z[p, a, b, u] for a in S for b in S) for p in P for u in U
    )
    # Corrected constraint to ensure the total outgoing and incoming flights are at most one for each UAV and delivery point
    model.add_constraints(
        model.sum(alpha[b, p, u] for b in A) == model.max(z[p, a, b, u] for a in S for b in S) for p in P for u in U
    )

    model.add_constraints(w[a, b, u] <= tau[a, b] for a in S for b in S for u in U)
    model.add_constraints(w[a, b, u] <= model.sum(alpha[p, b, u] for p in P) for a in S for b in S for u in U)
    model.add_constraints \
        (w[a, b, u] >= (tau[a, b] + model.sum(alpha[p, b, u] for p in P) - 1) for a in S for b in S for u in U
         )

    model.add_constraints(
        model.sum(alpha[a, p, u] for p in P) <= (1 - tau[a, b]) * M + w[a, b, u] for a in S for b in S for u in U
    )

    model.add_constraints(
        model.sum(alpha[a, p, u] for p in P) >= w[a, b, u] for a in S for b in S for u in U
    )

    model.add_constraints(q[a, b, u] <= tau[a, b] for a in S for b in S for u in U)
    model.add_constraints(q[a, b, u] <= model.max(z[p, a, b, u] for p in P) for a in S for b in S for u in U)
    model.add_constraints \
        (q[a, b, u] >= (tau[a, b] + model.max(z[p, a, b, u] for p in P) - 1) for a in S for b in S for u in U
         )

    model.add_constraints(
        model.sum(alpha[a, p, u] for p in P) <= (1 - tau[a, b]) * M + q[a, b, u] for a in S for b in S for u in U
    )

    model.add_constraints(
        model.sum(alpha[a, p, u] for p in P) >= q[a, b, u] for a in S for b in S for u in U
    )

    model.add_constraints(
        v[p_1, p_2, a, b, u] <= z[p_1, a, b, u] for p_1 in P for p_2 in P for a in S for b in S for u in U
    )

    model.add_constraints(
        v[p_1, p_2, a, b, u] <= z[p_2, a, b, u] for p_1 in P for p_2 in P for a in S for b in S for u in U
    )

    model.add_constraints(
        v[p_1, p_2, a, b, u] >= z[p_1, a, b, u] + z[p_2, a, b, u] - 1 for p_1 in P for p_2 in P for a in S for b in S
        for u in U
    )

    model.add_constraints(
        y[p_1, a, b, u] - y[p_2, a, b, u] - len_p * (1 - alpha[p_1, p_2, u]) + 1 <= M * (1 - v[p_1, p_2, a, b, u]) for
        p_1
        in P for p_2 in P for a in S for b in S for u in U
    )

    # Corrected constraint to ensure the total outgoing and incoming flights are at most one for each UAV and delivery point
    model.add_constraints(
        model.sum(z[p, a, b, u] for b in S) >= alpha[a, p, u] for p in P for u in U for a in S
    )

    # Corrected constraint to ensure the total outgoing and incoming flights are at most one for each UAV and delivery point
    model.add_constraints(
        model.sum(z[p, a, b, u] for a in S) >= alpha[p, b, u] for p in P for u in U for b in S
    )
    # Corrected constraint to ensure the total outgoing and incoming flights are at most one for each UAV and delivery point
    model.add_constraints(
        z[p, a, b, u] <= tau[a, b] for a in S for b in S for p in P for u in U
    )

    model.add_constraints(
        2 * alpha[p_1, p_2, u] <= model.max(z[p_1, a, b, u] + z[p_2, a, b, u] for a in S for b in S) for p_1 in P for
        p_2 in
        P for u in U
    )

    model.add_constraints(alpha[a, a, u] == 0 for a in A for u in U)
    model.add_constraints(alpha[a, b, u] == 0 for a in S for b in S for u in U)

    model.add_constraints(tau[a, a] == 0 for a in S + ["depot"])

    # Given the same model structure as before, let's add new constraints

    for i in range(len(S)):
        if i == 0:
            model.add_constraint(tau["depot", S[i]] == 1)
        else:
            model.add_constraint(tau["depot", S[i]] == 0)

    for i in range(len(S)):
        for j in range(len(S)):
            if j == i + 1:
                model.add_constraint(tau[S[i], S[j]] == 1)
            else:
                model.add_constraint(tau[S[i], S[j]] == 0)

        # model.add_constraint(tau[stops[i], stops[i+1]] == 1)
    model.add_constraint(tau[S[-1], "depot"] == 1)

    # Optional additional constraints to ensure specific sequences

    # if debug:
        # model.set_log_output(True)  # Enable solver logging
    model.set_time_limit(10)
    try:
        solution = model.solve()
        solve_details = model.solve_details
        if solve_details.status_code == 107:
            return None

        print("code: ",solve_details.status_code)

        if solution:
            # print("sdmlsmlslmdllslmdlmssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss")
            print("Optimal solution found!")
            print("Objective value:", solution.objective_value)

            # Display relevant information
            for key in z:
                if solution[z[key]] > 0:
                    print(f"Delivery {key[0]} is conducted by UAV {key[3]} between stops {key[1]} and {key[2]}.")

            for key in alpha:
                if solution[alpha[key]] > 0:
                    print(f"UAV {key[2]} flies from {key[0]} to {key[1]}.")
            for key in tau:
                if solution[tau[key]] > 0:
                    print(f"truck travels between {key[0]} and {key[1]}.")
            for key in h:
                if solution[h[key]] > 0:
                    print(f"UAV flies from {key[0]} carrying {solution[h[key]]} Kgs.")

            return solution.objective_value
        else:
            # No solution found, examine the solver status
            status = model.solution.get_status()

            if status == 103:
                print("The problem might be infeasible or unbounded.")
            elif status == 105:
                print("Solver terminated early due to a time limit or other constraints.")
            else:
                print("No solution found. Status:", status)
            return None

    except Exception as e:
        print("An error occurred while solving the model:", str(e))
