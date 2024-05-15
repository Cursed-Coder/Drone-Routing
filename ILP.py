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


def opt_algo_cplex(stop_coordinates, deliveries, num_drones, B_u, W_u, E):
    model = Model(name="uav_delivery_optimization")
    U = [f"u{idx + 1}" for idx in range(num_drones)]
    S = [f"s{idx + 1}" for idx in range(len(stop_coordinates))]
    P = [f"p{idx + 1}" for idx in range(len(deliveries))]

    # Unmanned Aerial Vehicles (UAVs)
    # U = ["u1", "u2", "u3"]  # Adding an extra UAV for variety

    # List of stop points
    # S = ["s1", "s2", "s3", "s4", "s5"]  # Adding one more stop point

    # List of delivery points
    # P = ["p1", "p2", "p3", "p4", "p5"]  # Adding one more delivery point

    # Rewards for each delivery point
    # rewards = {
    #     "p1": 15,
    #     "p2": 20,
    #     "p3": 30,
    #     "p4": 35,
    #     "p5": 40  # New reward for the additional delivery point
    # }

    # # Weights for each delivery point
    # weights = {
    #     "p1": 5,
    #     "p2": 7,
    #     "p3": 9,
    #     "p4": 12,
    #     "p5": 14  # Weight for the additional delivery point
    # }
    #
    # # Coordinates for stop points and delivery points
    # coordinates = {
    #     "s1": (0, 0),
    #     "s2": (2, 3),
    #     "s3": (6, 7),
    #     "s4": (10, 12),
    #     "s5": (15, 17),  # Coordinate for the new stop point
    #
    #     "p1": (1, 1),
    #     "p2": (4, 5),
    #     "p3": (7, 8),
    #     "p4": (11, 13),
    #     "p5": (16, 18)  # Coordinate for the new delivery point
    # }
    rewards = {}
    weights = {}
    coordinates = {}

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
    t = {s: model.integer_var(lb=1, ub=len(S), name=f"t_{s}") for s in S}

    y = {(p, a, b, u): model.integer_var(lb=1, ub=len_p, name=f"y_{p}_{a}_{b}_{u}") for p in P for a in S for b in S for
         u in U}

    v = {(p_1, p_2, a, b, u): model.binary_var(name=f"v_{p_1}_{p_2}_{a}_{b}_{u}") for p_1 in P for p_2 in P for a in S
         for b in S for u in U}

    # Objective: Maximize profit from deliveries
    model.maximize(model.sum(z[p, a, b, u] * rewards[p] for u in U for p in P for a in S for b in S))
    model.add_constraints(
        model.sum(z[p, a, b, u] for a in S for b in S for u in U) <= 1 for p in P
    )

    model.add_constraints(
        model.sum(E * alpha[a, b, u] * distance_matrix[(a, b)] for a in A for b in A) <= B_u for u in U
    )

    model.add_constraints(
        model.sum(z[p, a, b, u] * weights[p] for p in P) <= W_u for u in U for a in S for b in S
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

    for i in range(len(S) - 1):
        for j in range(len(S)):
            if j == i + 1:
                model.add_constraint(tau[S[i], S[j]] == 1)
            else:
                model.add_constraint(tau[S[i], S[j]] == 0)

        # model.add_constraint(tau[stops[i], stops[i+1]] == 1)
    model.add_constraint(tau[S[-1], "depot"] == 1)

    # Optional additional constraints to ensure specific sequences

    model.set_log_output(True)  # Enable solver logging

    try:
        solution = model.solve()

        if solution:
            print("Optimal solution found!")
            print("Objective value:", solution.objective_value)

            # Display relevant information
            for key in z:
                if solution[z[key]] > 0:
                    print(f"Delivery {key[0]} is conducted by UAV {key[3]} between stops {key[1]} and {key[2]}.")

            for key in alpha:
                if solution[alpha[key]] > 0:
                    print(f"UAV {key[2]} flies from {key[0]} to {key[1]}.")
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

    except Exception as e:
        print("An error occurred while solving the model:", str(e))
