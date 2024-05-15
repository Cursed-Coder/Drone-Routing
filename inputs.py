import numpy as np
import pandas as pd
import sympy as sp
import matplotlib.pyplot as plt
import itertools as it
import math
import scipy.stats as stats
import pickle
import networkx as nx
import numpy as np
import random
import pickle
from scipy.stats import zipf
from shapely.geometry import LineString, Point
from collections import namedtuple



# E = [2500000]  # J
# S = [4000]  # MB


def generate_random_coordinates(num_nodes):
    """
    Generate random coordinates for a given number of nodes.
    """
    return {f"Node_{i}": (np.random.uniform(0, 100), np.random.uniform(0, 100)) for i in range(num_nodes)}


def create_complete_graph_with_weights(coordinates):
    """
    Create a complete graph with weights based on Euclidean distances.
    """
    node_names = list(coordinates.keys())
    graph = nx.complete_graph(len(node_names))
    nx.relabel_nodes(graph, {i: node_names[i] for i in range(len(node_names))}, copy=False)

    # Assign weights based on Euclidean distances
    for edge in graph.edges:
        node1, node2 = edge
        coord1 = coordinates[node1]
        coord2 = coordinates[node2]
        distance = np.linalg.norm(np.array(coord1) - np.array(coord2))
        graph[edge[0]][edge[1]]['weight'] = distance

    return graph


def generate_random_cycle(graph_2, node_names, target_distance, tolerance, max_iterations=50000):
    """
    Generate a random directed cycle (closed loop) with a target distance and tolerance.
    """
    iteration = 0
    random_order = None
    edges_in_path = None
    total_distance = None

    while iteration < max_iterations:
        random_order = random.sample(node_names, len(node_names))
        edges_in_path = [(random_order[i], random_order[i + 1]) for i in range(len(random_order) - 1)]
        edges_in_path.append((random_order[-1], random_order[0]))  # Complete the cycle

        total_distance = sum(graph_2[edge[0]][edge[1]]['weight'] for edge in edges_in_path)

        iteration += 1

    # if abs(total_distance - target_distance) <= tolerance:
    return random_order, edges_in_path, total_distance

    # raise ValueError("Could not generate a cycle within the specified tolerance and iterations.")


def define_stop_points(edges_in_path, coordinates, graph, min_distance, num_stop_points):
    """
    Define stop points along the path with a minimum distance constraint.
    """
    cumulative_lengths = []
    current_length = 0
    for edge in edges_in_path:
        current_length += graph[edge[0]][edge[1]]['weight']
        cumulative_lengths.append((edge, current_length))

    total_path_length = cumulative_lengths[-1][1]
    stop_points = []
    attempts = 0

    while len(stop_points) < num_stop_points and attempts < 50000:
        random_position = np.random.uniform(0, total_path_length)
        if all(abs(random_position - sp) >= min_distance for sp in stop_points):
            stop_points.append(random_position)
        attempts += 1

    stop_points.sort()  # Sort the stop points
    print("Here ", coordinates)

    # Map stop points to edges
    stop_point_coords = []
    for stop_point in stop_points:
        for i, (edge, cum_length) in enumerate(cumulative_lengths):
            if cum_length > stop_point:
                previous_cum_length = cumulative_lengths[i - 1][1] if i > 0 else 0
                edge_length = cum_length - previous_cum_length
                position_along_edge = (stop_point - previous_cum_length) / edge_length

                node1, node2 = edge
                coord1 = np.array(coordinates[node1])
                coord2 = np.array(coordinates[node2])

                # # Calculate the coordinates of the stop point
                stop_coord = coord1 + position_along_edge * (coord2 - coord1)
                stop_point_coords.append((stop_point, stop_coord))
                break

    return stop_point_coords


def is_point_far_enough(point, existing_deliveries, min_distance):
    """
    Check if a point is far enough from existing deliveries.
    """
    return all(Point(delivery['coordinates']).distance(point) >= min_distance for delivery in existing_deliveries)


def define_deliveries(edges_in_path_lines, num_deliveries, min_edge_distance, max_edge_distance, min_delivery_distance,
                      profit_distribution, weights_range):
    """
    Generate random deliveries with certain constraints, including profit distribution and distance to path.
    """
    deliveries = []
    num_attempts = 0

    while len(deliveries) < num_deliveries and num_attempts < 100000:
        # Random x, y coordinates
        x = np.random.uniform(0, 100)
        y = np.random.uniform(0, 100)

        # Create a Point for the delivery
        point = Point(x, y)

        # Calculate distance to each edge
        distances = [line.distance(point) for line in edges_in_path_lines]

        # Check if the point is within acceptable distance from at least one edge
        if all(min_edge_distance <= distance <= max_edge_distance for distance in distances) and is_point_far_enough(
                point, deliveries, min_delivery_distance):
            # Random profit from Zipf distribution
            profit = profit_distribution.rvs()
            closest_value = min([5, 10, 20], key=lambda x: abs(x - profit))

            # Random weight within the specified range
            weight = np.random.uniform(weights_range[0], weights_range[1])

            deliveries.append({
                'id': f"Delivery_{len(deliveries) + 1}",
                'profit': closest_value,
                'weight': weight,
                'coordinates': [x, y]  # Convert coordinates to list
            })

        num_attempts += 1  # Prevent endless loop

    # Ensure all deliveries meet the constraints
    return [delivery for delivery in deliveries if any(
        min_edge_distance <= line.distance(Point(delivery['coordinates'])) <= max_edge_distance for line in
        edges_in_path_lines)]


def save_data_to_file(data, filename):
    """
    Save the problem instance data to a file.
    """
    with open(filename, 'ab') as f:
        pickle.dump(data, f)


# Example usage
def generate_problem_instance():
    global N_DELIVERIES, ZIPF_PARAM, N_STOP_POINTS
    for n_deliveries in N_DELIVERIES:
        for theta in ZIPF_PARAM:
            for num_stop_points in N_STOP_POINTS:
                for i in range(iterations):
                    instances = []
                    num_nodes = 5
                    target_distance = 400.0
                    tolerance = 50
                    min_dist_between_stop_points = 30
                    profit_distribution = zipf(theta)
                    weights_range = (2, 4)

                    # Step 1: Generate coordinates for nodes
                    coordinates = generate_random_coordinates(num_nodes)
                    # print("Here2 ",coordinates)

                    # Step 2: Create a complete graph with weights
                    graph = create_complete_graph_with_weights(coordinates)

                    # Step 3: Generate a random directed cycle
                    node_names = list(coordinates.keys())
                    random_order, edges_in_path, total_distance = generate_random_cycle(graph, node_names, target_distance,
                                                                                        tolerance)

                    # Step 4: Define stop points along the path
                    stop_point_coords = define_stop_points(edges_in_path, coordinates, graph, min_dist_between_stop_points,
                                                           num_stop_points)

                    # Step 5: Define deliveries with constraints
                    edges_in_path_lines = [LineString([coordinates[edge[0]], coordinates[edge[1]]]) for edge in
                                           edges_in_path]

                    deliveries = define_deliveries(
                        edges_in_path_lines,
                        n_deliveries,
                        min_edge_distance=5,
                        max_edge_distance=100,
                        min_delivery_distance=5,
                        profit_distribution=profit_distribution,
                        weights_range=weights_range,
                    )

                    # # Step 6: Output the minimum closest distance between deliveries and edges
                    delivery_points = [Point(delivery['coordinates']) for delivery in deliveries]
                    closest_distances = [min([line.distance(delivery_point) for line in edges_in_path_lines]) for
                                         delivery_point
                                         in delivery_points]
                    min_closest_distance = min(closest_distances) if closest_distances else None

                    deliveries_2 = {}
                    for index, delivery in enumerate(deliveries):
                        # Create a new key for each delivery (D1, D2, D3, etc.)
                        key = f"p{index + 1}"

                        # Map the original profit to reward, and keep the other properties
                        deliveries_2[key] = {
                            0: delivery["profit"],  # Example reward mapping
                            1: delivery["weight"],  # Example weight adjustment
                            2: tuple(delivery["coordinates"]),  # Convert to tuple
                        }

                    stop_coordinates = {}
                    for index, s in enumerate(stop_point_coords):
                        # Generate the key using the ASCII value of 'A' + index
                        key = f"s{index + 1}"
                        # Convert the array to a tuple
                        # coordinates = tuple(array)
                        # Add the key and coordinates to the dictionary
                        stop_coordinates[key] = s

                    # Step 7: Save data to file
                    instances.append([stop_coordinates, deliveries_2])
                    # data_to_save = {
                    #     "coordinates": coordinates,
                    #     "graph_edges": [(edge[0], edge[1], graph[edge[0]][edge[1]]['weight']) for edge in graph.edges],
                    #     "edges_in_path": edges_in_path,
                    #     "stop_points": stop_point_coords,
                    #     "deliveries": deliveries
                    # }
                    # print("Hello")
                    name = "problems/problem_n" + str(n_deliveries) + "_t" + str(theta) + "_s" + str(
                        num_stop_points) + ".dat"

                    save_data_to_file(instances, name)

                    print("Truck's Path:", " -> ".join(random_order))
                    print("Coordinates of Nodes:")
                    for node, coord in coordinates.items():
                        print(f"{node}: {coord}")

                    print("Deliveries (Profit, Weight, Coordinates):")
                    for delivery in deliveries:
                        print(
                            f"{delivery['id']} - Profit: {delivery['profit']}, Weight: {delivery['weight']}, Coordinates: {delivery['coordinates']}")

                    print("Minimum distance between any delivery and any edge:", min_closest_distance)
